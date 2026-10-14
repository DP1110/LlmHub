"""FastAPI Application providing OpenAI-compatible endpoints and LLMHub system APIs.

Endpoints:
- POST /v1/chat/completions (SSE streaming and JSON across concurrent models)
- GET  /v1/models (OpenAI model catalog)
- GET  /v1/models/{model_id} (OpenAI individual model metadata)
- GET  /api/ps (Active running models matrix, pinned flags, memory, and tok/s)
- GET  /api/models (Extended model registry manifest)
- POST /api/models/{model_spec}/load (Manual load with context & pinning)
- POST /api/models/{model_spec}/unload (Manual unload)
- POST /api/models/{model_spec}/pin (Pin model against LRU eviction)
- POST /api/models/{model_spec}/unpin (Unpin model)
- GET  /healthz (Uptime and residency status)
"""

import asyncio
from contextlib import asynccontextmanager
import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from urllib.parse import unquote
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llmhub.config import LLMHubConfig
from llmhub.core.registry import ModelRegistry
from llmhub.core.template import ChatTemplateEngine
from llmhub.hardware.detector import HardwareDetector
from llmhub.server.backend import Backend
from llmhub.server.llamacpp import LlamaCppBackend
from llmhub.server.model_manager import LoadedModel, MemoryAdmissionError, ModelManager

START_TIME = time.time()

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None

class ServerState:
    def __init__(self) -> None:
        self.config = LLMHubConfig()
        self.registry = ModelRegistry(self.config.db_path)
        self.model_manager = ModelManager(config=self.config, registry=self.registry)
        self.lock = asyncio.Lock()

state = ServerState()

def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.model_manager.start_reaper()
        yield
        state.model_manager.stop_reaper()

    app = FastAPI(
        title="LLMHub Inference Server",
        version="0.4.0",
        description="Offline-first multi-model GGUF inference platform with OpenAI API compatibility",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "uptime_seconds": time.time() - START_TIME,
            "is_model_loaded": len(state.model_manager.loaded_models) > 0,
            "resident_models": list(state.model_manager.loaded_models.keys()),
            "resident_models_count": len(state.model_manager.loaded_models),
        }

    @app.get("/v1/models")
    async def list_v1_models() -> Dict[str, Any]:
        records = state.registry.list_models()
        data = [
            {
                "id": r.tag,
                "object": "model",
                "created": int(time.time()),
                "owned_by": r.org,
                "permission": [],
                "root": r.name,
                "parent": None
            }
            for r in records
        ]
        return {"object": "list", "data": data}

    @app.get("/v1/models/{model_id:path}")
    async def get_v1_model(model_id: str) -> Dict[str, Any]:
        clean_id = unquote(model_id)
        record = state.registry.find_model(clean_id) or state.registry.find_model(model_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
        return {
            "id": record.tag,
            "object": "model",
            "created": int(time.time()),
            "owned_by": record.org,
            "permission": [],
            "root": record.name,
            "parent": None
        }

    @app.get("/api/models")
    async def list_api_models() -> Dict[str, Any]:
        records = state.registry.list_models()
        return {
            "models": [r.to_dict() for r in records]
        }

    @app.get("/api/ps")
    async def get_ps() -> Dict[str, Any]:
        return state.model_manager.get_ps_info()

    @app.post("/api/models/{model_spec:path}/load")
    async def api_load_model(
        model_spec: str,
        n_ctx: Optional[int] = None,
        pinned: bool = False,
        keep_alive: Optional[float] = None
    ) -> Dict[str, Any]:
        try:
            m = await state.model_manager.load_model(
                model_spec,
                n_ctx=n_ctx,
                pinned=pinned,
                keep_alive=keep_alive
            )
            return {
                "status": "loaded",
                "tag": m.tag,
                "pinned": m.pinned,
                "n_ctx": m.n_ctx,
                "total_mb": round(m.total_memory_bytes / (1024 * 1024), 1)
            }
        except MemoryAdmissionError as e:
            raise HTTPException(status_code=507, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/models/{model_spec:path}/unload")
    async def api_unload_model(model_spec: str, force: bool = False) -> Dict[str, Any]:
        try:
            success = await state.model_manager.unload_model(model_spec, force=force)
            if not success:
                raise HTTPException(status_code=404, detail=f"Model '{model_spec}' is not resident in memory.")
            return {"status": "unloaded", "model": model_spec}
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/api/models/{model_spec:path}/pin")
    async def api_pin_model(model_spec: str) -> Dict[str, Any]:
        try:
            await state.model_manager.pin_model(model_spec)
            return {"status": "pinned", "model": model_spec}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/models/{model_spec:path}/unpin")
    async def api_unpin_model(model_spec: str) -> Dict[str, Any]:
        success = await state.model_manager.unpin_model(model_spec)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model '{model_spec}' is not resident in memory.")
        return {"status": "unpinned", "model": model_spec}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        try:
            loaded_model = await state.model_manager.get_or_load_model(req.model)
        except MemoryAdmissionError as e:
            raise HTTPException(status_code=507, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Render prompt from messages
        messages_dict = [{"role": m.role, "content": m.content or ""} for m in req.messages]
        prompt = loaded_model.template_engine.render(messages_dict, add_generation_prompt=True)

        stop_criteria = [req.stop] if isinstance(req.stop, str) else req.stop
        session_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        loaded_model.active_sessions += 1
        loaded_model.last_active_time = time.monotonic()
        try:
            loaded_model.backend.new_session(
                session_id=session_id,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=stop_criteria
            )
        except Exception:
            loaded_model.active_sessions = max(0, loaded_model.active_sessions - 1)
            raise

        if req.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                created_ts = int(time.time())
                try:
                    # Emit initial assistant role chunk for strict OpenAI client compliance
                    role_chunk = {
                        "id": session_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(role_chunk)}\n\n"

                    # Run chunked prefill loop
                    is_complete = False
                    while not is_complete:
                        async with loaded_model.inference_lock:
                            _, is_complete = await asyncio.to_thread(loaded_model.backend.prefill_chunk, session_id, 512)
                        await asyncio.sleep(0)

                    while True:
                        # Decode 1 token per step; thread-offloaded and locked against context corruption
                        async with loaded_model.inference_lock:
                            res = await asyncio.to_thread(loaded_model.backend.decode_step, session_id, 1)

                        for delta in res.text_deltas:
                            chunk_data = {
                                "id": session_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"

                        if res.done:
                            sess = loaded_model.backend.get_session(session_id)
                            prompt_len = len(sess.prompt_tokens) if sess else 0
                            final_chunk = {
                                "id": session_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": res.stop_reason or "stop"
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": prompt_len,
                                    "completion_tokens": res.total_tokens_generated,
                                    "total_tokens": prompt_len + res.total_tokens_generated,
                                    "ttft_ms": round(res.ttft_ms, 2),
                                    "decode_tok_s": round(res.decode_tok_s, 2),
                                }
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        await asyncio.sleep(0)
                finally:
                    try:
                        loaded_model.backend.close_session(session_id)
                    finally:
                        loaded_model.active_sessions = max(0, loaded_model.active_sessions - 1)
                        loaded_model.last_active_time = time.monotonic()

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        else:
            # Non-streaming JSON response
            created_ts = int(time.time())
            try:
                # Complete prompt prefill
                is_complete = False
                while not is_complete:
                    async with loaded_model.inference_lock:
                        _, is_complete = await asyncio.to_thread(loaded_model.backend.prefill_chunk, session_id, 1024)
                    await asyncio.sleep(0)

                full_text = ""
                finish_reason = "stop"

                while True:
                    async with loaded_model.inference_lock:
                        res = await asyncio.to_thread(loaded_model.backend.decode_step, session_id, 4)
                    full_text += "".join(res.text_deltas)
                    if res.done:
                        finish_reason = res.stop_reason or "stop"
                        break
                    await asyncio.sleep(0)

                sess = loaded_model.backend.get_session(session_id)
                prompt_tokens = len(sess.prompt_tokens) if sess else 0
                completion_tokens = len(sess.generated_tokens) if sess else 0

                return {
                    "id": session_id,
                    "object": "chat.completion",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": full_text
                            },
                            "finish_reason": finish_reason
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
            finally:
                try:
                    loaded_model.backend.close_session(session_id)
                finally:
                    loaded_model.active_sessions = max(0, loaded_model.active_sessions - 1)
                    loaded_model.last_active_time = time.monotonic()

    return app

app = create_app()
