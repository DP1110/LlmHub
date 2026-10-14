"""FastAPI Application providing OpenAI-compatible endpoints and LLMHub system APIs.

Features (Phase 9 - Observability and Hardening):
- Prometheus Metrics Exporter (GET /metrics) tracking active sessions, TTFT, tok/s, KV bytes, thermals, and memory
- Structured JSON logging with request tracing (X-Request-ID) across async contexts
- Crash recovery, PID lifecycle tracking, and stale lock reclamation in ~/.llmhub/run/
- Graceful model unloading and request draining (HTTP 503 during drain, timeout-based completion)
- API key Bearer token authentication and per-key Token Bucket rate limiting (RPM/TPM)
- Request size limits (HTTP 413) and input sanitization (null bytes, control chars, prompt bounds)
- Client disconnect detection aborting orphaned inference sessions
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from urllib.parse import unquote
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llmhub.config import LLMHubConfig, is_termux
from llmhub.core.lifecycle import PIDManager
from llmhub.core.registry import ModelRegistry
from llmhub.core.template import ChatTemplateEngine
from llmhub.hardware.detector import HardwareDetector
from llmhub.observability.logger import current_request_id, get_logger, setup_logging
from llmhub.observability.metrics import MetricsRegistry, get_metrics_registry
from llmhub.scheduler.policy import ThreadingMode
from llmhub.scheduler.scheduler import QueueFullError, TokenRoundRobinScheduler
from llmhub.scheduler.session import SchedSession, SessionPhase
from llmhub.scheduler.slo import SLOViolationError
from llmhub.security.auth import APIKeyAuthenticator, RateLimiter
from llmhub.security.sanitizer import InputSanitizer, PayloadTooLargeError, ValidationError
from llmhub.server.backend import Backend
from llmhub.server.llamacpp import LlamaCppBackend
from llmhub.server.model_manager import (
    LoadedModel,
    MemoryAdmissionError,
    ModelDrainingError,
    ModelManager,
)
from llmhub.termux.controller import TermuxController
from llmhub.cluster.node import NodeInfo, NodeRole, NodeStatus
from llmhub.cluster.manager import ClusterManager
from llmhub.cluster.router import ClusterRouter, RoutingDecision
from llmhub.cluster.pipeline import PipelineCoordinator, PipelinePlan

logger = get_logger("server")
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
        self.termux_controller = TermuxController(
            config=self.config,
            base_threads=self.config.settings.get("runtime", {}).get("default_threads", 4),
        )
        self.termux_controller.set_thread_callback(self.scheduler.set_thermal_thread_limit)

        def _on_memory_pressure(level: Any) -> None:
            from llmhub.termux.memory_pressure import MemoryPressureLevel
            if level == MemoryPressureLevel.CRITICAL:
                for lm in self.model_manager.loaded_models.values():
                    if hasattr(lm, "batch_engine") and lm.batch_engine and lm.batch_engine.prefix_cache:
                        lm.batch_engine.prefix_cache.prune_memory_pressure(ratio=0.5)
            elif level == MemoryPressureLevel.MODERATE:
                for lm in self.model_manager.loaded_models.values():
                    if hasattr(lm, "batch_engine") and lm.batch_engine and lm.batch_engine.prefix_cache:
                        lm.batch_engine.prefix_cache.prune_memory_pressure(ratio=0.25)

        self.termux_controller.set_memory_pressure_callback(_on_memory_pressure)
        self.lock = asyncio.Lock()

        # Phase 9: Observability & Security components
        self.metrics: MetricsRegistry = get_metrics_registry()
        sec_cfg = self.config.settings.get("security", {})
        self.authenticator = APIKeyAuthenticator(
            enabled=sec_cfg.get("auth_enabled", False),
            api_keys=sec_cfg.get("api_keys", []),
            key_configs=sec_cfg.get("key_configs", {}),
            exempt_localhost=sec_cfg.get("exempt_localhost", True),
        )
        self.rate_limiter = RateLimiter(
            default_rpm=sec_cfg.get("default_rpm", 60),
            default_tpm=sec_cfg.get("default_tpm", 30000),
        )
        self.pid_manager = PIDManager(self.config.run_dir)

        # Register live dynamic Prometheus metrics collector
        def _collect_live_metrics() -> None:
            loaded = self.model_manager.loaded_models
            self.metrics.loaded_models_count.set(len(loaded))
            self.metrics.memory_allocated_bytes.set(sum(m.total_memory_bytes for m in loaded.values()))
            self.metrics.active_sessions.set(sum(m.active_sessions for m in loaded.values()))
            self.metrics.queued_sessions.set(len([s for s in self.scheduler.runnable_queue if not s.is_done]))

            try:
                import psutil
                vm = psutil.virtual_memory()
                self.metrics.system_memory_used_ratio.set(vm.percent / 100.0)
            except Exception:
                pass

            try:
                t_reading = self.termux_controller.thermal.last_reading
                if t_reading:
                    self.metrics.thermal_temperature_celsius.set(t_reading.temperature_c)
                m_probe = self.termux_controller.memory.last_reading
                if m_probe:
                    from llmhub.termux.memory_pressure import MemoryPressureLevel
                    lvl = 0
                    if m_probe.level == MemoryPressureLevel.MODERATE:
                        lvl = 1
                    elif m_probe.level == MemoryPressureLevel.CRITICAL:
                        lvl = 2
                    self.metrics.memory_pressure_level.set(lvl)
                self.metrics.wakelock_held.set(1 if self.termux_controller.wakelock.is_held else 0)
            except Exception:
                pass

        self.metrics.register_collector(_collect_live_metrics)

        # Phase 10: Cluster & Distributed Scale-Out components
        cl_cfg = self.config.settings.get("cluster", {})
        srv_cfg = self.config.settings.get("server", {})
        local_node_id = cl_cfg.get("node_id") or f"node-{uuid.uuid4().hex[:8]}"
        local_node_name = cl_cfg.get("node_name") or f"node-{local_node_id[-4:]}"
        self.local_node = NodeInfo(
            node_id=local_node_id,
            name=local_node_name,
            host=srv_cfg.get("host", "127.0.0.1"),
            port=srv_cfg.get("port", 11434),
            api_url=f"http://{srv_cfg.get('host', '127.0.0.1')}:{srv_cfg.get('port', 11434)}",
            is_termux=is_termux(),
            cluster_secret=cl_cfg.get("cluster_secret") or None,
        )
        self.cluster_manager = ClusterManager(
            local_node=self.local_node,
            enabled=cl_cfg.get("enabled", False),
            static_peers=cl_cfg.get("peers", []),
            heartbeat_interval_s=float(cl_cfg.get("heartbeat_interval_s", 2.0)),
            node_timeout_s=float(cl_cfg.get("node_timeout_s", 6.0)),
            dead_timeout_s=float(cl_cfg.get("dead_timeout_s", 30.0)),
            beacon_enabled=cl_cfg.get("beacon_enabled", True),
            beacon_port=int(cl_cfg.get("beacon_port", 11435)),
            cluster_secret=cl_cfg.get("cluster_secret") or None,
        )
        self.cluster_router = ClusterRouter(self.cluster_manager)

        def _get_cluster_node_state() -> Dict[str, Any]:
            loaded = self.model_manager.loaded_models
            free_mb = 0.0
            tot_mb = 0.0
            try:
                import psutil
                vm = psutil.virtual_memory()
                free_mb = vm.available / (1024 * 1024)
                tot_mb = vm.total / (1024 * 1024)
            except Exception:
                pass
            return {
                "free_ram_mb": round(free_mb, 1),
                "total_ram_mb": round(tot_mb, 1),
                "active_sessions": sum(m.active_sessions for m in loaded.values()),
                "resident_models": {
                    tag: {"memory_bytes": m.total_memory_bytes, "active_sessions": m.active_sessions}
                    for tag, m in loaded.items()
                },
                "is_termux": is_termux(),
            }

        self.cluster_manager.set_state_provider(_get_cluster_node_state)


state = ServerState()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Setup structured logging and validate configuration
        setup_logging(
            log_level=state.config.settings.get("observability", {}).get("log_level", "INFO"),
            logs_dir=state.config.logs_dir,
            json_format=state.config.settings.get("observability", {}).get("log_json", True),
        )
        state.config.validate(strict=False)

        # 2. Cleanup stale run-dir artifacts and acquire PID
        state.pid_manager.cleanup_run_dir()
        srv_cfg = state.config.settings.get("server", {})
        try:
            state.pid_manager.acquire(
                host=srv_cfg.get("host", "127.0.0.1"),
                port=srv_cfg.get("port", 11434),
                version="1.0.0",
            )
        except Exception as e:
            logger.warning("PID acquisition issue: %s", e)

        # 3. Start background loops
        state.model_manager.start_reaper()
        state.scheduler.start()
        await state.termux_controller.start()
        await state.cluster_manager.start()

        yield

        # 4. Graceful shutdown: drain models, stop loops, release PID
        drain_timeout = float(state.config.settings.get("lifecycle", {}).get("drain_timeout_s", 30.0))
        try:
            await state.model_manager.drain_all(timeout=drain_timeout)
            await state.cluster_manager.stop()
            await state.cluster_router.aclose()
            await state.termux_controller.stop()
            state.scheduler.stop()
            state.model_manager.stop_reaper()
        finally:
            state.pid_manager.release()

    app = FastAPI(
        title="LLMHub Inference Server",
        version="0.9.0",
        description="Offline-first multi-model GGUF inference platform with token-level round-robin scheduling",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Security Authentication, Rate Limiting, and Payload Size Guard
    # Registered before tracing so that request_tracing_middleware executes outermost.
    @app.middleware("http")
    async def security_and_rate_limit_middleware(request: Request, call_next: Any) -> Response:
        # 1. Payload size check
        max_bytes = state.config.settings.get("security", {}).get("max_request_body_bytes", 4 * 1024 * 1024)
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Payload too large. Request body exceeds limit of {max_bytes} bytes.",
                        "type": "payload_too_large",
                    }
                },
            )

        # 1b. Chunked transfer encoding / missing Content-Length body size guard
        if request.method in ("POST", "PUT", "PATCH") and (
            cl is None or request.headers.get("transfer-encoding", "").lower() == "chunked"
        ):
            chunks = []
            total_bytes = 0
            exceeded = False
            async for chunk in request.stream():
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    exceeded = True
                    break
                chunks.append(chunk)

            if exceeded:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "message": f"Payload too large. Request body exceeds limit of {max_bytes} bytes.",
                            "type": "payload_too_large",
                        }
                    },
                )
            request._body = b"".join(chunks)

        # 2. Authentication check
        auth_header = request.headers.get("authorization")
        client_host = request.client.host if request.client else "unknown"
        is_forwarded = bool(request.headers.get("x-forwarded-for") or request.headers.get("forwarded"))
        auth_ok, client_key, auth_err = state.authenticator.authenticate_header(
            auth_header=auth_header,
            client_host=client_host,
            path=request.url.path,
            is_forwarded=is_forwarded,
        )
        if not auth_ok:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": auth_err or "Unauthorized",
                        "type": "authentication_error",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        request.state.client_id = client_key or client_host

        # 3. Rate limiting check (skip exempt paths)
        sec_cfg = state.config.settings.get("security", {})
        if sec_cfg.get("rate_limit_enabled", True) and not state.authenticator.is_exempt_path(request.url.path):
            custom_rpm, custom_tpm = state.authenticator.get_key_limits(client_key or "")
            allowed, retry_after, remaining, limit = state.rate_limiter.check_request(
                client_id=client_key or client_host,
                est_tokens=1,
                custom_rpm=custom_rpm,
                custom_tpm=custom_tpm,
            )
            if not allowed:
                state.metrics.rate_limit_exceeded_total.inc(labels={"key": client_key or "anonymous"})
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "Rate limit exceeded. Please retry after specified delay.",
                            "type": "rate_limit_error",
                        }
                    },
                    headers={
                        "Retry-After": str(int(retry_after) + 1),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": str(remaining),
                    },
                )

        return await call_next(request)

    # Middleware: Request Tracing & Duration Histogram (outermost layer)
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next: Any) -> Response:
        req_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        token = current_request_id.set(req_id)
        start_t = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            dur = time.perf_counter() - start_t
            # Record histogram for API endpoints
            state.metrics.request_duration_seconds.observe(
                dur,
                labels={"method": request.method, "status": str(status_code)},
            )
            current_request_id.reset(token)

    # Built-in Web GUI and Static Files
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=FileResponse)
    @app.get("/dashboard", response_class=FileResponse)
    async def get_dashboard() -> Response:
        """Serve offline-first single-page Web GUI dashboard."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file), media_type="text/html")
        return HTMLResponse("<html><body><h1>LLMHub Server Running</h1></body></html>")

    @app.get("/metrics")
    async def get_metrics() -> Response:
        """Expose Prometheus 0.0.4 metrics exposition."""
        text = state.metrics.export()
        return Response(content=text, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "uptime_seconds": time.time() - START_TIME,
            "is_model_loaded": len(state.model_manager.loaded_models) > 0,
            "resident_models": list(state.model_manager.loaded_models.keys()),
            "resident_models_count": len(state.model_manager.loaded_models),
            "termux": {
                "is_termux": is_termux(),
                "thermal_state": state.termux_controller.thermal.current_state.value,
                "memory_pressure": state.termux_controller.memory.probe().level.value,
                "low_memory_mode": state.config.is_low_memory_mode,
            },
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
                "parent": None,
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
            "parent": None,
        }

    @app.get("/api/models")
    async def list_api_models() -> Dict[str, Any]:
        records = state.registry.list_models()
        return {"models": [r.to_dict() for r in records]}

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

    @app.get("/api/termux/status")
    async def get_termux_status() -> Dict[str, Any]:
        return state.termux_controller.get_status()

    @app.post("/api/termux/wakelock/acquire")
    async def acquire_wakelock() -> Dict[str, Any]:
        success = await asyncio.to_thread(state.termux_controller.wakelock.acquire)
        return {
            "status": "acquired" if success else "unsupported_or_failed",
            "is_held": state.termux_controller.wakelock.is_held,
            "is_supported": state.termux_controller.wakelock.is_supported,
        }

    @app.post("/api/termux/wakelock/release")
    async def release_wakelock() -> Dict[str, Any]:
        success = await asyncio.to_thread(state.termux_controller.wakelock.release)
        return {
            "status": "released" if success else "failed",
            "is_held": state.termux_controller.wakelock.is_held,
        }

    @app.post("/api/system/low-memory")
    async def set_low_memory_mode(enabled: bool) -> Dict[str, Any]:
        state.config.set_low_memory_mode(enabled)
        return {
            "low_memory_mode": state.config.is_low_memory_mode,
            "status": "updated",
        }

    @app.post("/api/scheduler/mode")
    async def set_scheduler_mode(mode: str) -> Dict[str, Any]:
        try:
            tmode = ThreadingMode(mode)
            state.scheduler.threading_strategy.mode = tmode
            return {"status": "updated", "threading_mode": tmode.value}
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid threading mode '{mode}'. Valid modes: time-slice, partition, hybrid",
            )

    @app.post("/api/models/{model_spec:path}/load")
    async def api_load_model(
        model_spec: str,
        n_ctx: Optional[int] = None,
        pinned: bool = False,
        keep_alive: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            m = await state.model_manager.load_model(
                model_spec, n_ctx=n_ctx, pinned=pinned, keep_alive=keep_alive
            )
            return {
                "status": "loaded",
                "tag": m.tag,
                "pinned": m.pinned,
                "n_ctx": m.n_ctx,
                "total_mb": round(m.total_memory_bytes / (1024 * 1024), 1),
            }
        except MemoryAdmissionError as e:
            raise HTTPException(status_code=507, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/models/{model_spec:path}/unload")
    async def api_unload_model(
        model_spec: str,
        force: bool = False,
        drain: bool = False,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        try:
            success = await state.model_manager.unload_model(
                model_spec, force=force, drain=drain, timeout=timeout
            )
            if not success:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model_spec}' is not resident in memory."
                )
            return {"status": "unloaded", "model": model_spec, "drained": drain}
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
            raise HTTPException(
                status_code=404, detail=f"Model '{model_spec}' is not resident in memory."
            )
        return {"status": "unpinned", "model": model_spec}

    # Phase 10: Cluster & Distributed Scale-Out Endpoints
    @app.get("/api/cluster/status")
    async def get_cluster_status() -> Dict[str, Any]:
        """Aggregate health, capacity, and model distribution across all cluster nodes."""
        return state.cluster_manager.get_cluster_status()

    @app.get("/api/cluster/nodes")
    async def get_cluster_nodes() -> Dict[str, Any]:
        """Return list of known cluster nodes and their live capabilities."""
        st = state.cluster_manager.get_cluster_status()
        return {"nodes": st.get("nodes", [])}

    @app.post("/api/cluster/heartbeat")
    async def post_cluster_heartbeat(data: Dict[str, Any], request: Request) -> Dict[str, Any]:
        """Receive peer heartbeat update and reply with local node status."""
        sec = state.cluster_manager.cluster_secret
        if sec:
            import secrets
            req_sec = request.headers.get("X-Cluster-Secret") or data.get("secret") or ""
            if not secrets.compare_digest(req_sec, sec):
                raise HTTPException(status_code=403, detail="Invalid cluster secret")
        return await state.cluster_manager.handle_heartbeat_received(data)

    @app.post("/api/cluster/join")
    async def post_cluster_join(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to an existing cluster peer node and synchronize cluster state."""
        peer_url = payload.get("peer_url")
        if not peer_url:
            raise HTTPException(status_code=400, detail="Missing peer_url")
        ok, msg = await state.cluster_manager.join_peer(peer_url)
        if not ok:
            raise HTTPException(status_code=502, detail=msg)
        return {"status": "joined", "message": msg, "cluster": state.cluster_manager.get_cluster_status()}

    @app.post("/api/cluster/plan")
    async def post_cluster_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compute optimal pipeline layer partition plan across active cluster nodes."""
        model_tag = payload.get("model")
        if not model_tag:
            raise HTTPException(status_code=400, detail="Missing model tag")
        total_layers = int(payload.get("total_layers", 32))
        hidden_dim = int(payload.get("hidden_dim", 4096))
        weights_mb = float(payload.get("weights_mb", 4000.0))
        all_nodes = [state.cluster_manager.local_node] + list(state.cluster_manager.peers.values())
        plan = PipelineCoordinator.plan_pipeline(
            model_tag=model_tag,
            total_layers=total_layers,
            hidden_dim=hidden_dim,
            total_weights_mb=weights_mb,
            nodes=all_nodes,
        )
        return plan.to_dict()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
        req_start_t = time.perf_counter()

        # 1. Input sanitization & schema validation
        try:
            sanitized_messages = InputSanitizer.validate_chat_messages(req.messages)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # 1b. Locality-Aware Distributed Cluster Routing Check
        if state.cluster_manager.enabled and state.config.settings.get("cluster", {}).get("auto_route", True):
            routing_dec = state.cluster_router.route_model(req.model)
            if not routing_dec.is_local:
                logger.info(
                    "Forwarding inference request for model '%s' to cluster node %s (%s) [reason=%s]",
                    req.model, routing_dec.target, routing_dec.target_url, routing_dec.reason,
                )
                raw_payload = req.dict()
                if req.stream:
                    return await state.cluster_router.forward_chat_completion_stream(
                        routing_dec.target_url, raw_payload, request, dict(request.headers)
                    )
                else:
                    fwd_resp = await state.cluster_router.forward_chat_completion_json(
                        routing_dec.target_url, raw_payload, dict(request.headers)
                    )
                    # Deduct tokens from rate limiter if successful JSON response
                    if fwd_resp.status_code == 200:
                        try:
                            body_data = json.loads(fwd_resp.body.decode("utf-8"))
                            usage = body_data.get("usage", {})
                            c_toks = usage.get("completion_tokens", 0)
                            p_toks = usage.get("prompt_tokens", 0)
                            tot = p_toks + c_toks
                            state.metrics.tokens_generated_total.inc(c_toks, labels={"model": req.model})
                            state.metrics.prompt_tokens_total.inc(p_toks, labels={"model": req.model})
                            cid = getattr(request.state, "client_id", None)
                            if cid and tot > 1:
                                state.rate_limiter.consume_tokens(cid, tot - 1)
                        except Exception:
                            pass
                    return fwd_resp

        # 2. Model retrieval / loading with draining check
        try:
            loaded_model = await state.model_manager.get_or_load_model(req.model)
        except ModelDrainingError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except MemoryAdmissionError as e:
            raise HTTPException(status_code=507, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # 3. Render prompt from messages and validate bounds
        prompt = loaded_model.template_engine.render(sanitized_messages, add_generation_prompt=True)
        try:
            InputSanitizer.validate_prompt_bounds(prompt, loaded_model.n_ctx)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

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

        state.scheduler.start()

        try:
            await state.scheduler.submit_session(sched_session)
        except QueueFullError as e:
            state.metrics.requests_total.inc(
                labels={"model": req.model, "status": "rejected_queue_full", "tenant": tenant_id}
            )
            raise HTTPException(status_code=429, detail=str(e))
        except SLOViolationError as e:
            state.metrics.requests_total.inc(
                labels={"model": req.model, "status": "rejected_slo", "tenant": tenant_id}
            )
            raise HTTPException(
                status_code=429,
                detail=f"SLO admission rejected: predicted TTFT ({e.estimated_ttft_ms:.1f}ms) exceeds target ({e.target_ttft_ms:.1f}ms)",
            )

        if req.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                created_ts = int(time.time())
                ttft_observed = False
                try:
                    role_chunk = {
                        "id": session_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(role_chunk)}\n\n"

                    while True:
                        # Client disconnect detection: abort orphaned session
                        if await request.is_disconnected():
                            logger.info("Client disconnected from streaming session %s; aborting", session_id)
                            await state.scheduler.cancel_session(session_id)
                            state.metrics.requests_total.inc(
                                labels={"model": req.model, "status": "canceled_disconnect", "tenant": tenant_id}
                            )
                            break

                        try:
                            msg = await asyncio.wait_for(sched_session.token_queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            continue

                        if msg is None:
                            break
                        if msg.get("error"):
                            err_chunk = {"error": msg["error"]}
                            yield f"data: {json.dumps(err_chunk)}\n\n"
                            state.metrics.requests_total.inc(
                                labels={"model": req.model, "status": "error", "tenant": tenant_id}
                            )
                            break
                        if "delta" in msg:
                            if not ttft_observed:
                                ttft_s = time.perf_counter() - req_start_t
                                state.metrics.ttft_seconds.observe(ttft_s, labels={"model": req.model})
                                ttft_observed = True

                            chunk_data = {
                                "id": session_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": msg["delta"]},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                        if msg.get("done"):
                            usage = msg.get("usage", {})
                            final_chunk = {
                                "id": session_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": msg.get("stop_reason") or "stop",
                                    }
                                ],
                                "usage": usage,
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"

                            # Metrics tracking
                            state.metrics.requests_total.inc(
                                labels={"model": req.model, "status": "success", "tenant": tenant_id}
                            )
                            p_toks = usage.get("prompt_tokens", 0)
                            c_toks = usage.get("completion_tokens", 0)
                            state.metrics.tokens_generated_total.inc(
                                c_toks, labels={"model": req.model}
                            )
                            state.metrics.prompt_tokens_total.inc(
                                p_toks, labels={"model": req.model}
                            )
                            # Deduct actual tokens from TPM bucket (minus 1 token already accounted during admission check)
                            cid = getattr(request.state, "client_id", None)
                            if cid and (p_toks + c_toks) > 1:
                                state.rate_limiter.consume_tokens(cid, (p_toks + c_toks) - 1)
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
            usage: Dict[str, Any] = {}
            ttft_observed = False
            try:
                while True:
                    if await request.is_disconnected():
                        logger.info("Client disconnected from session %s; aborting", session_id)
                        await state.scheduler.cancel_session(session_id)
                        state.metrics.requests_total.inc(
                            labels={"model": req.model, "status": "canceled_disconnect", "tenant": tenant_id}
                        )
                        raise HTTPException(status_code=499, detail="Client Closed Request")

                    try:
                        msg = await asyncio.wait_for(sched_session.token_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    if msg is None:
                        break
                    if msg.get("error"):
                        status_code = 504 if msg.get("stop_reason") == "timeout" else 500
                        state.metrics.requests_total.inc(
                            labels={"model": req.model, "status": "error", "tenant": tenant_id}
                        )
                        raise HTTPException(status_code=status_code, detail=msg["error"])
                    if "delta" in msg:
                        if not ttft_observed:
                            ttft_s = time.perf_counter() - req_start_t
                            state.metrics.ttft_seconds.observe(ttft_s, labels={"model": req.model})
                            ttft_observed = True
                        full_text += msg["delta"]
                    if msg.get("done"):
                        finish_reason = msg.get("stop_reason") or "stop"
                        usage = msg.get("usage", {})
                        break

                prompt_toks = usage.get("prompt_tokens", 0)
                comp_toks = usage.get("completion_tokens", len(full_text.split()))
                tot_toks = usage.get("total_tokens", prompt_toks + comp_toks)

                state.metrics.requests_total.inc(
                    labels={"model": req.model, "status": "success", "tenant": tenant_id}
                )
                state.metrics.tokens_generated_total.inc(comp_toks, labels={"model": req.model})
                state.metrics.prompt_tokens_total.inc(prompt_toks, labels={"model": req.model})

                # Deduct actual tokens from TPM bucket (minus 1 token already accounted during admission check)
                cid = getattr(request.state, "client_id", None)
                if cid and tot_toks > 1:
                    state.rate_limiter.consume_tokens(cid, tot_toks - 1)

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
                                "content": full_text,
                            },
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_toks,
                        "completion_tokens": comp_toks,
                        "total_tokens": tot_toks,
                    },
                }
            finally:
                if not sched_session.is_done:
                    await state.scheduler.cancel_session(session_id)

    return app


app = create_app()
