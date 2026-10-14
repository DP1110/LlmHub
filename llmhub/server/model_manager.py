"""Multi-model resident runtime manager with memory admission control and LRU eviction.

Features:
- Concurrent residence of multiple GGUF models in memory
- Mathematical memory admission control (weights + KV cache + overhead against free RAM safety margin)
- LRU (Least-Recently-Used) eviction of idle models
- Configurable keep-alive expiration timer (default 5 minutes / 300 seconds)
- Background async reaper task
- Model pinning (pinned models are immune to LRU eviction and keep-alive expiry)
- Zero-copy OS page cache sharing via mmap
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from llmhub.config import LLMHubConfig
from llmhub.core.gguf_parser import GGUFMetadata, GGUFParser
from llmhub.core.registry import ModelRecord, ModelRegistry
from llmhub.core.template import ChatTemplateEngine
from llmhub.hardware.detector import HardwareDetector
from llmhub.server.backend import Backend, DecodeResult, SessionState
from llmhub.server.llamacpp import LlamaCppBackend
from llmhub.termux.downshift import DownshiftEngine
from llmhub.utils.math_calc import compute_memory_breakdown, estimate_decode_throughput

class MemoryAdmissionError(Exception):
    """Raised when a model cannot be loaded due to memory admission constraints."""
    def __init__(self, message: str, required_bytes: int, available_bytes: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        self.details = details or {}


class ModelDrainingError(RuntimeError):
    """Raised when a request targets a model that is currently draining for unloading."""
    pass


@dataclass
class LoadedModel:
    tag: str
    file_path: Path
    backend: Backend
    metadata: GGUFMetadata
    template_engine: ChatTemplateEngine
    n_ctx: int
    n_threads: int
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    total_memory_bytes: int
    pinned: bool = False
    keep_alive_seconds: float = 300.0
    last_active_time: float = field(default_factory=time.monotonic)
    active_sessions: int = 0
    inference_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    batch_engine: Optional[Any] = None
    is_draining: bool = False
    drain_event: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        if self.batch_engine is None:
            from llmhub.batch.engine import ContinuousBatchEngine
            self.batch_engine = ContinuousBatchEngine(model_tag=self.tag, max_slots=4)
        if self.active_sessions == 0:
            self.drain_event.set()

    def inc_active_sessions(self) -> None:
        self.active_sessions += 1
        self.drain_event.clear()

    def dec_active_sessions(self) -> None:
        self.active_sessions = max(0, self.active_sessions - 1)
        if self.active_sessions == 0:
            self.drain_event.set()

    @property
    def is_idle(self) -> bool:
        return self.active_sessions == 0

    @property
    def idle_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.last_active_time)

    @property
    def is_expired(self) -> bool:
        if self.pinned or not self.is_idle:
            return False
        return self.idle_seconds >= self.keep_alive_seconds

class ModelManager:
    """Manages concurrent residence, memory admission, LRU eviction, and pinning of models."""

    def __init__(
        self,
        config: Optional[LLMHubConfig] = None,
        registry: Optional[ModelRegistry] = None,
        safety_margin: float = 0.85,
        default_keep_alive: float = 300.0,
        backend_factory: Optional[Callable[[], Backend]] = None
    ) -> None:
        self.config = config or LLMHubConfig()
        self.registry = registry or ModelRegistry(self.config.db_path)
        self.safety_margin = safety_margin
        self.default_keep_alive = default_keep_alive
        self.backend_factory = backend_factory or (lambda: LlamaCppBackend())
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.lock = asyncio.Lock()
        self._reaper_task: Optional[asyncio.Task] = None

    def get_free_ram_bytes(self) -> int:
        """Query system for currently available unallocated physical RAM (fast-path via psutil with fallback)."""
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            return HardwareDetector.get_profile().free_ram_bytes

    def calculate_model_memory(
        self,
        file_path: Path,
        metadata: GGUFMetadata,
        n_ctx: int
    ) -> Tuple[int, int, int, int]:
        """Compute exact memory footprint breakdown: (weights, kv_cache, overhead, total)."""
        file_size = file_path.stat().st_size
        breakdown = compute_memory_breakdown(
            weights_bytes=file_size,
            n_layers=metadata.block_count,
            n_kv_heads=metadata.head_count_kv,
            head_dim=metadata.head_dim,
            n_ctx=n_ctx
        )
        return (
            int(breakdown.weights_mb * 1024 * 1024),
            int(breakdown.kv_cache_mb * 1024 * 1024),
            int(breakdown.overhead_mb * 1024 * 1024),
            int(breakdown.total_mb * 1024 * 1024)
        )

    def check_admission(self, required_bytes: int, exclude_tag: Optional[str] = None) -> Tuple[bool, int, int]:
        """Check if required memory fits within free RAM safety margin minus resident models.
        
        Returns: (can_admit, usable_free_ram_bytes, shortfall_bytes)
        """
        free_ram = self.get_free_ram_bytes()
        usable_ram = int(free_ram * self.safety_margin)
        resident_bytes = sum(
            m.total_memory_bytes
            for tag, m in self.loaded_models.items()
            if tag != exclude_tag
        )
        available_bytes = usable_ram - resident_bytes
        if required_bytes <= available_bytes:
            return True, available_bytes, 0
        return False, max(0, available_bytes), required_bytes - available_bytes

    def get_eviction_candidates(self) -> List[LoadedModel]:
        """Return idle unpinned models sorted by LRU (least recently used first)."""
        candidates = [
            m for m in self.loaded_models.values()
            if m.is_idle and not m.pinned
        ]
        # Sort ascending by last_active_time (oldest active time first)
        candidates.sort(key=lambda m: m.last_active_time)
        return candidates

    def evict_for_admission(self, required_bytes: int) -> List[str]:
        """Evict idle unpinned models using LRU policy until memory fits.
        
        Raises MemoryAdmissionError if memory cannot be satisfied even after evicting all idle models.
        Returns list of evicted model tags.
        """
        can_admit, available_ram, shortfall = self.check_admission(required_bytes)
        if can_admit:
            return []

        candidates = self.get_eviction_candidates()
        total_reclaimable = sum(m.total_memory_bytes for m in candidates)

        # Non-destructive pre-check: If even evicting ALL idle candidates cannot satisfy memory requirement, fail immediately
        if available_ram + total_reclaimable < required_bytes:
            resident_summary = {
                m.tag: {
                    "pinned": m.pinned,
                    "active_sessions": m.active_sessions,
                    "ram_mb": round(m.total_memory_bytes / (1024 * 1024), 1)
                }
                for m in self.loaded_models.values()
            }
            raise MemoryAdmissionError(
                f"Insufficient memory to load model ({required_bytes / (1024**2):.1f} MB required). "
                f"Available RAM ({available_ram / (1024**2):.1f} MB) plus all idle models ({total_reclaimable / (1024**2):.1f} MB reclaimable) is insufficient. "
                f"No models were evicted.",
                required_bytes=required_bytes,
                available_bytes=available_ram,
                details={
                    "shortfall_mb": round((required_bytes - available_ram - total_reclaimable) / (1024**2), 1),
                    "evicted_models": [],
                    "resident_models": resident_summary,
                }
            )

        evicted_tags: List[str] = []
        freed_bytes = 0

        for candidate in candidates:
            # Unload candidate model
            tag = candidate.tag
            candidate.backend.unload()
            freed_bytes += candidate.total_memory_bytes
            evicted_tags.append(tag)
            self.loaded_models.pop(tag, None)

            # Re-check admission
            can_admit, available_ram, shortfall = self.check_admission(required_bytes)
            if can_admit:
                break

        return evicted_tags

    async def load_model(
        self,
        model_spec: str,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        keep_alive: Optional[float] = None,
        pinned: bool = False
    ) -> LoadedModel:
        """Load a model with memory admission control and mmap page caching."""
        async with self.lock:
            # Check if already loaded
            if model_spec in self.loaded_models:
                loaded = self.loaded_models[model_spec]
                loaded.last_active_time = time.time()
                if pinned:
                    loaded.pinned = True
                return loaded

            # Resolve model record or direct GGUF file
            record = self.registry.find_model(model_spec)
            if not record:
                cand = Path(model_spec).resolve()
                if cand.is_file() and cand.suffix.lower() == ".gguf":
                    file_path = cand
                    tag = cand.stem
                else:
                    raise FileNotFoundError(
                        f"Model '{model_spec}' not found in registry. Pull it first via 'llmhub pull {model_spec}'."
                    )
            else:
                file_path = Path(record.file_path)
                tag = record.tag

            if not file_path.is_file():
                raise FileNotFoundError(f"Model binary file missing from disk: {file_path}")

            # Check if resolved tag is already loaded
            if tag in self.loaded_models:
                loaded = self.loaded_models[tag]
                loaded.last_active_time = time.time()
                if pinned:
                    loaded.pinned = True
                return loaded

            # Parse GGUF metadata
            metadata = GGUFParser.parse_file(file_path)
            ctx = n_ctx or self.config.settings.get("runtime", {}).get("default_context_size", 2048)
            threads = n_threads or self.config.settings.get("runtime", {}).get("default_threads", 4)
            keep_alive_sec = keep_alive if keep_alive is not None else self.default_keep_alive

            # Calculate exact memory requirement
            w_b, kv_b, ov_b, total_b = self.calculate_model_memory(file_path, metadata, ctx)

            # Check if downshifting context is needed to fit under memory pressure
            allow_downshift = self.config.settings.get("termux", {}).get("allow_ctx_downshift", True)
            can_admit, avail_ram, shortfall = self.check_admission(total_b)
            if not can_admit and allow_downshift:
                candidates = self.get_eviction_candidates()
                total_reclaimable = sum(m.total_memory_bytes for m in candidates)
                total_avail_with_eviction = avail_ram + total_reclaimable

                fitting_ctx, rationale = DownshiftEngine.find_fitting_context(
                    weights_bytes=w_b,
                    n_layers=metadata.block_count,
                    n_kv_heads=metadata.head_count_kv,
                    head_dim=metadata.head_dim,
                    desired_ctx=ctx,
                    available_ram_bytes=total_avail_with_eviction,
                    safety_margin=1.0,  # Already factored in check_admission
                    min_ctx=self.config.settings.get("termux", {}).get("min_ctx_size", 512),
                )
                if fitting_ctx and fitting_ctx < ctx:
                    ctx = fitting_ctx
                    w_b, kv_b, ov_b, total_b = self.calculate_model_memory(file_path, metadata, ctx)

            # Memory admission control: evict idle models if necessary
            self.evict_for_admission(total_b)

            # Initialize backend
            backend = self.backend_factory()
            use_mlock = self.config.settings.get("runtime", {}).get("use_mlock", False)
            if HardwareDetector.is_termux() and HardwareDetector.is_non_root():
                use_mlock = False

            # Offload heavy C++ model weight loading to worker thread
            await asyncio.to_thread(
                backend.load,
                model_path=file_path,
                n_ctx=ctx,
                n_threads=threads,
                use_mmap=True,  # Strictly enforce mmap for page cache sharing
                use_mlock=use_mlock
            )

            # Chat template engine
            template_str = metadata.chat_template
            arch = metadata.architecture or "llama"
            template_engine = ChatTemplateEngine(template_str=template_str, architecture=arch)

            loaded_model = LoadedModel(
                tag=tag,
                file_path=file_path,
                backend=backend,
                metadata=metadata,
                template_engine=template_engine,
                n_ctx=ctx,
                n_threads=threads,
                weights_bytes=w_b,
                kv_cache_bytes=kv_b,
                overhead_bytes=ov_b,
                total_memory_bytes=total_b,
                pinned=pinned,
                keep_alive_seconds=keep_alive_sec,
                last_active_time=time.monotonic(),
                active_sessions=0
            )

            self.loaded_models[tag] = loaded_model
            return loaded_model

    async def unload_model(
        self,
        model_tag: str,
        force: bool = False,
        drain: bool = False,
        timeout: float = 30.0,
    ) -> bool:
        """Unload a resident model.
        
        If drain=True and active sessions exist, marks the model as draining and waits up to
        timeout seconds for all in-flight sessions to finish.
        If force=False and active sessions remain, raises RuntimeError.
        """
        async with self.lock:
            loaded = self.get_model(model_tag)
            if not loaded:
                return False
            actual_tag = loaded.tag
            if drain and loaded.active_sessions > 0:
                loaded.is_draining = True

        if drain and loaded.active_sessions > 0:
            try:
                if loaded.active_sessions > 0:
                    await asyncio.wait_for(loaded.drain_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                if not force:
                    async with self.lock:
                        loaded.is_draining = False
                    raise RuntimeError(
                        f"Timed out draining {loaded.active_sessions} active sessions from model '{actual_tag}' after {timeout:.1f}s."
                    )
            finally:
                if not force and loaded.active_sessions > 0:
                    async with self.lock:
                        loaded.is_draining = False

        async with self.lock:
            loaded = self.loaded_models.get(actual_tag)
            if not loaded:
                return False

            if loaded.active_sessions > 0 and not force:
                raise RuntimeError(
                    f"Cannot unload model '{actual_tag}': {loaded.active_sessions} active sessions are in progress."
                )

            async with loaded.inference_lock:
                loaded.backend.unload()
                self.loaded_models.pop(actual_tag, None)
            return True

    async def drain_all(self, timeout: float = 30.0) -> None:
        """Gracefully drain and unload all resident models before server shutdown."""
        tags = list(self.loaded_models.keys())
        for tag in tags:
            try:
                await self.unload_model(tag, drain=True, timeout=timeout, force=True)
            except Exception:
                pass

    def get_model(self, model_spec: str) -> Optional[LoadedModel]:
        """Retrieve loaded model if present."""
        if model_spec in self.loaded_models:
            return self.loaded_models[model_spec]
        record = self.registry.find_model(model_spec)
        if record and record.tag in self.loaded_models:
            return self.loaded_models[record.tag]
        cand = Path(model_spec).stem
        return self.loaded_models.get(cand)

    async def get_or_load_model(self, model_spec: str) -> LoadedModel:
        """Retrieve loaded model or trigger dynamic load via admission controller."""
        model = self.get_model(model_spec)
        if model:
            if model.is_draining:
                raise ModelDrainingError(
                    f"Model '{model.tag}' is currently draining for unload and cannot accept new requests."
                )
            model.last_active_time = time.monotonic()
            return model
        return await self.load_model(model_spec)

    async def pin_model(self, model_spec: str) -> bool:
        """Pin a model into memory, protecting it from LRU eviction and keep-alive expiration."""
        model = self.get_model(model_spec)
        if not model:
            model = await self.load_model(model_spec, pinned=True)
            return True
        async with self.lock:
            model.pinned = True
            return True

    async def unpin_model(self, model_spec: str) -> bool:
        """Unpin a model, allowing it to be evicted when idle."""
        async with self.lock:
            model = self.get_model(model_spec)
            if not model:
                return False
            model.pinned = False
            model.last_active_time = time.monotonic()
            return True

    async def reap_expired_models(self) -> List[str]:
        """Evict models whose keep_alive duration has expired and have zero active sessions."""
        expired_tags: List[str] = []
        for tag, model in list(self.loaded_models.items()):
            if model.is_expired:
                async with model.inference_lock:
                    if model.is_expired:
                        model.backend.unload()
                        self.loaded_models.pop(tag, None)
                        expired_tags.append(tag)
        return expired_tags

    async def _reaper_loop(self, interval_seconds: float = 10.0) -> None:
        """Background coroutine periodically running keep_alive expiration reaper."""
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                async with self.lock:
                    await self.reap_expired_models()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def start_reaper(self, interval_seconds: float = 10.0) -> None:
        """Start the background keep-alive reaper task."""
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self._reaper_loop(interval_seconds))

    def stop_reaper(self) -> None:
        """Stop the background keep-alive reaper task."""
        if self._reaper_task and not self._reaper_task.done():
            self._reaper_task.cancel()
            self._reaper_task = None

    def get_ps_info(self) -> Dict[str, Any]:
        """Generate comprehensive system and multi-model status report for /api/ps."""
        prof = HardwareDetector.get_profile()
        models_data = []

        # Snapshot resident models safely to prevent dict iteration mutation race
        models_snapshot = list(self.loaded_models.values())
        total_resident_mb = 0.0
        for m in models_snapshot:
            mb_tot = m.total_memory_bytes / (1024 * 1024)
            total_resident_mb += mb_tot
            bw = estimate_decode_throughput(m.weights_bytes)

            models_data.append({
                "tag": m.tag,
                "pinned": m.pinned,
                "active_sessions": m.active_sessions,
                "idle_seconds": round(m.idle_seconds, 1),
                "keep_alive_seconds": m.keep_alive_seconds,
                "n_ctx": m.n_ctx,
                "n_threads": m.n_threads,
                "memory": {
                    "weights_mb": round(m.weights_bytes / (1024 * 1024), 1),
                    "kv_cache_mb": round(m.kv_cache_bytes / (1024 * 1024), 1),
                    "overhead_mb": round(m.overhead_bytes / (1024 * 1024), 1),
                    "total_mb": round(mb_tot, 1),
                },
                "bandwidth": {
                    "theoretical_max_tok_s": round(bw.theoretical_max_tok_s, 1),
                    "bottleneck": bw.bottleneck_type
                }
            })

        return {
            "backend_status": "loaded" if len(models_snapshot) > 0 else "idle",
            "active_sessions": sum(m.active_sessions for m in models_snapshot),
            "loaded_model": models_snapshot[0].tag if models_snapshot else None,
            "resident_models_count": len(models_snapshot),
            "total_resident_mb": round(total_resident_mb, 1),
            "system_ram_total_mb": round(prof.total_ram_bytes / (1024 * 1024), 1),
            "system_ram_free_mb": round(prof.free_ram_bytes / (1024 * 1024), 1),
            "safety_margin": self.safety_margin,
            "models": models_data
        }
