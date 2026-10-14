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
from llmhub.scheduler.policy import ThreadingMode
from llmhub.scheduler.scheduler import QueueFullError, TokenRoundRobinScheduler
from llmhub.scheduler.session import SchedSession, SessionPhase
from llmhub.scheduler.slo import SLOViolationError
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
    priority: Optional[int] = 5
    weight: Optional[float] = 1.0
    tenant_id: Optional[str] = None
    user: Optional[str] = None
    slo_target_ttft_ms: Optional[float] = None
    slo_target_tbt_ms: Optional[float] = None
    strict_slo: bool = False
    use_speculative: bool = False
    draft_model: Optional[str] = None

class ServerState:
    def __init__(self) -> None:
        self.config = LLMHubConfig()
        self.registry = ModelRegistry(self.config.db_path)
        self.model_manager = ModelManager(config=self.config, registry=self.registry)
        self.scheduler = TokenRoundRobinScheduler(model_manager=self.model_manager)
        self.lock = asyncio.Lock()

state = ServerState()

def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.model_manager.start_reaper()
        state.scheduler.start()
        yield
        state.scheduler.stop()
        state.model_manager.stop_reaper()

    app = FastAPI(
        title="LLMHub Inference Server",
        version="0.7.0",
        description="Offline-first multi-model GGUF inference platform with token-level round-robin scheduling",
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
        info = state.model_manager.get_ps_info()
        info["scheduler"] = state.scheduler.get_telemetry()
        batch_summary = {}
        for tag, lm in state.model_manager.loaded_models.items():
            if hasattr(lm, "batch_engine") and lm.batch_engine:
                st = lm.batch_engine.get_status()
                batch_summary[tag] = {
                    "max_slots": st["max_slots"],
                    "active_slots": st["active_slots"],
                    "utilization_pct": st["utilization_pct"],
                    "prefix_cache_hit_rate": st["prefix_cache"]["hit_rate_pct"],
                }
        info["batch_summary"] = batch_summary
        return info

    @app.get("/api/batch")
    async def get_batch_status() -> Dict[str, Any]:
        batch_status = {}
        for tag, lm in state.model_manager.loaded_models.items():
            if hasattr(lm, "batch_engine") and lm.batch_engine:
                batch_status[tag] = lm.batch_engine.get_status()
        return {
            "loaded_models_count": len(state.model_manager.loaded_models),
            "batch_engines": batch_status,
        }

    @app.get("/api/scheduler")
    async def get_scheduler_telemetry() -> Dict[str, Any]:
        return state.scheduler.get_telemetry()

    @app.get("/api/scheduler/mlfq")
    async def get_scheduler_mlfq() -> Dict[str, Any]:
        return state.scheduler.mlfq.get_stats()

    @app.get("/api/scheduler/slo")
    async def get_scheduler_slo() -> Dict[str, Any]:
        return state.scheduler.slo_controller.get_stats()

    @app.get("/api/scheduler/tenants")
    async def get_scheduler_tenants() -> Dict[str, Any]:
        return state.scheduler.fair_share.get_stats()

    @app.get("/api/speculative")
    async def get_speculative_status() -> Dict[str, Any]:
        return {k: eng.get_stats() for k, eng in state.scheduler.speculative_engines.items()}

    @app.post("/api/scheduler/mode")
    async def set_scheduler_mode(mode: str) -> Dict[str, Any]:
        try:
            tmode = ThreadingMode(mode)
            state.scheduler.threading_strategy.mode = tmode
            return {"status": "updated", "threading_mode": tmode.value}
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid threading mode '{mode}'. Valid modes: time-slice, partition, hybrid"
            )

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
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
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
        tenant_id = req.tenant_id or req.user or request.headers.get("X-Tenant-ID") or "default"

        sched_session = SchedSession(
            session_id=session_id,
            model_tag=req.model,
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=stop_criteria,
            priority=req.priority or 5,
            weight=req.weight or 1.0,
            tenant_id=tenant_id,
            slo_target_ttft_ms=req.slo_target_ttft_ms,
            slo_target_tbt_ms=req.slo_target_tbt_ms,
            strict_slo=req.strict_slo,
            use_speculative=req.use_speculative,
            draft_model_tag=req.draft_model,
        )

        # Ensure scheduler loop is running on current event loop
        state.scheduler.start()

        try:
            await state.scheduler.submit_session(sched_session)
        except QueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e))
        except SLOViolationError as e:
            raise HTTPException(
                status_code=429,
                detail=f"SLO admission rejected: predicted TTFT ({e.estimated_ttft_ms:.1f}ms) exceeds target ({e.target_ttft_ms:.1f}ms)"
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

                    while True:
                        msg = await sched_session.token_queue.get()
                        if msg is None:
                            break
                        if msg.get("error"):
                            err_chunk = {"error": msg["error"]}
                            yield f"data: {json.dumps(err_chunk)}\n\n"
                            break
                        if "delta" in msg:
                            chunk_data = {
                                "id": session_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": msg["delta"]},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                        if msg.get("done"):
                            final_chunk = {
                                "id": session_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": msg.get("stop_reason") or "stop"
                                    }
                                ],
                                "usage": msg.get("usage", {})
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                finally:
                    if not sched_session.is_done:
                        await state.scheduler.cancel_session(session_id)

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        else:
            # Non-streaming JSON response
            created_ts = int(time.time())
            full_text = ""
            finish_reason = "stop"
            usage = {}
            try:
                while True:
                    msg = await sched_session.token_queue.get()
                    if msg is None:
                        break
                    if msg.get("error"):
                        status_code = 504 if msg.get("stop_reason") == "timeout" else 500
                        raise HTTPException(status_code=status_code, detail=msg["error"])
                    if "delta" in msg:
                        full_text += msg["delta"]
                    if msg.get("done"):
                        finish_reason = msg.get("stop_reason") or "stop"
                        usage = msg.get("usage", {})
                        break

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
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", len(full_text.split())),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
                }
            finally:
                if not sched_session.is_done:
                    await state.scheduler.cancel_session(session_id)

    return app

app = create_app()
