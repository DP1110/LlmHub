"""FastAPI Application providing OpenAI-compatible endpoints and LLMHub system APIs.

Endpoints:
- POST /v1/chat/completions (SSE streaming and JSON)
- GET  /v1/models (OpenAI model catalog)
- GET  /api/ps (Active running models, active sessions, memory, and tok/s)
- GET  /api/models (Extended model registry manifest)
- GET  /healthz (Uptime and status)
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
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
    stop: Optional[List[str]] = None

class ServerState:
    def __init__(self) -> None:
        self.config = LLMHubConfig()
        self.registry = ModelRegistry(self.config.db_path)
        self.backend: Backend = LlamaCppBackend()
        self.current_model_tag: Optional[str] = None
        self.template_engine: Optional[ChatTemplateEngine] = None
        self.lock = asyncio.Lock()
        self.inference_lock = asyncio.Lock()

state = ServerState()

def create_app() -> FastAPI:
    app = FastAPI(
        title="LLMHub Inference Server",
        version="0.3.0",
        description="Offline-first GGUF inference platform with OpenAI API compatibility"
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
            "is_model_loaded": state.backend.is_loaded(),
            "loaded_model": state.current_model_tag,
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
        record = state.registry.find_model(model_id)
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
        info = state.backend.get_info()
        hw = HardwareDetector.get_profile()
        return {
            "model": state.current_model_tag,
            "backend_status": info.get("status", "unloaded"),
            "active_sessions": info.get("active_sessions", 0),
            "memory": info.get("memory", {}),
            "bandwidth": info.get("bandwidth", {}),
            "system_ram_free_mb": hw.free_ram_bytes / (1024 * 1024),
            "suggested_threads": hw.suggested_threads,
        }

    async def _ensure_model_loaded(model_spec: str) -> None:
        async with state.lock:
            record = state.registry.find_model(model_spec)
            if not record:
                # Check direct GGUF file path
                cand = Path(model_spec).resolve()
                if cand.is_file() and cand.suffix.lower() == ".gguf":
                    file_path = cand
                    tag = cand.stem
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model '{model_spec}' not found in registry. Pull it first via 'llmhub pull {model_spec}'."
                    )
            else:
                file_path = Path(record.file_path)
                tag = record.tag

            if state.current_model_tag == tag and state.backend.is_loaded():
                if state.template_engine is None:
                    metadata = getattr(state.backend, "metadata", None)
                    template_str = metadata.chat_template if metadata else None
                    arch = metadata.architecture if metadata else "llama"
                    state.template_engine = ChatTemplateEngine(template_str=template_str, architecture=arch)
                return

            # Check if active inference sessions are currently running
            if state.current_model_tag != tag and state.backend.is_loaded():
                info = state.backend.get_info()
                active = info.get("active_sessions", 0)
                if active > 0:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Cannot switch model to '{tag}' while active inference session(s) ({active}) are running."
                    )
                state.backend.unload()

            # Load model
            state.backend.load(
                model_path=file_path,
                n_ctx=state.config.settings.get("runtime", {}).get("default_context_size", 2048),
                use_mmap=state.config.settings.get("runtime", {}).get("use_mmap", True),
                use_mlock=state.config.settings.get("runtime", {}).get("use_mlock", False),
            )
            state.current_model_tag = tag

            # Instantiate template engine
            metadata = getattr(state.backend, "metadata", None)
            template_str = metadata.chat_template if metadata else None
            arch = metadata.architecture if metadata else "llama"
            state.template_engine = ChatTemplateEngine(template_str=template_str, architecture=arch)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        await _ensure_model_loaded(req.model)
        if not state.template_engine:
            raise HTTPException(status_code=500, detail="Template engine not initialized.")

        # Render prompt from messages
        messages_dict = [{"role": m.role, "content": m.content or ""} for m in req.messages]
        prompt = state.template_engine.render(messages_dict, add_generation_prompt=True)

        session_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        state.backend.new_session(
            session_id=session_id,
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop
        )

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
                        async with state.inference_lock:
                            _, is_complete = await asyncio.to_thread(state.backend.prefill_chunk, session_id, 512)
                        await asyncio.sleep(0)

                    while True:
                        # Decode 1 token per step; thread-offloaded and locked against context corruption
                        async with state.inference_lock:
                            res = await asyncio.to_thread(state.backend.decode_step, session_id, 1)

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
                            sess = state.backend.get_session(session_id)
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
                    state.backend.close_session(session_id)

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        else:
            # Non-streaming JSON response
            created_ts = int(time.time())
            try:
                # Complete prompt prefill
                is_complete = False
                while not is_complete:
                    async with state.inference_lock:
                        _, is_complete = await asyncio.to_thread(state.backend.prefill_chunk, session_id, 1024)
                    await asyncio.sleep(0)

                full_text = ""
                finish_reason = "stop"

                while True:
                    async with state.inference_lock:
                        res = await asyncio.to_thread(state.backend.decode_step, session_id, 4)
                    full_text += "".join(res.text_deltas)
                    if res.done:
                        finish_reason = res.stop_reason or "stop"
                        break
                    await asyncio.sleep(0)

                sess = state.backend.get_session(session_id)
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
                state.backend.close_session(session_id)

    return app

app = create_app()
