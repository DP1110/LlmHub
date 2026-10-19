"""Token-level Round-Robin Scheduler with Chunked Prefill and Preemption Points."""

import asyncio
from collections import deque
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Set

from llmhub.scheduler.fair_share import TenantFairShareManager
from llmhub.scheduler.mlfq import MultilevelFeedbackQueue
from llmhub.scheduler.policy import (
    DeficitRoundRobinPolicy,
    ThreadingMode,
    ThreadingStrategy,
)
from llmhub.scheduler.session import SchedSession, SessionPhase
from llmhub.scheduler.slo import SLOAdmissionController, SLOTarget, SLOViolationError
from llmhub.server.backend import DecodeResult
from llmhub.server.model_manager import LoadedModel, ModelManager
from llmhub.speculative.engine import SpeculativeEngine

logger = logging.getLogger(__name__)

class QueueFullError(Exception):
    """Raised when the runnable queue exceeds maximum backpressure limit."""
    pass

class SessionTimeoutError(Exception):
    """Raised when a session exceeds its configured timeout duration."""
    pass

class TokenRoundRobinScheduler:
    """Core token-level round-robin scheduler across multiple models and sessions."""

    def __init__(
        self,
        model_manager: ModelManager,
        default_quantum: int = 4,
        default_chunk_size: int = 256,
        max_queue_depth: int = 128,
        threading_mode: ThreadingMode = ThreadingMode.HYBRID,
        enable_continuous_batching: bool = True,
        enable_mlfq: bool = True,
        enable_slo: bool = True,
        enable_fair_share: bool = True,
    ):
        self.model_manager = model_manager
        self.default_quantum = default_quantum
        self.default_chunk_size = default_chunk_size
        self.max_queue_depth = max_queue_depth
        self.enable_continuous_batching = enable_continuous_batching
        self.enable_mlfq = enable_mlfq
        self.enable_slo = enable_slo
        self.enable_fair_share = enable_fair_share

        self.threading_strategy = ThreadingStrategy(mode=threading_mode)
        self.policy = DeficitRoundRobinPolicy(default_quantum=default_quantum)
        self.mlfq = MultilevelFeedbackQueue(boost_interval_seconds=10.0, enable_mlfq=enable_mlfq)
        self.slo_controller = SLOAdmissionController(default_target_ttft_ms=2000.0, default_target_tbt_ms=50.0)
        self.fair_share = TenantFairShareManager()
        self.speculative_engines: Dict[str, SpeculativeEngine] = {}

        self.runnable_queue: List[SchedSession] = []
        self.completed_sessions: Deque[SchedSession] = deque(maxlen=500)
        self.lock = asyncio.Lock()
        self.running = False
        self._loop_task: Optional[asyncio.Task] = None

        # Statistics & Context-switch telemetry
        self.last_scheduled_model: Optional[str] = None
        self.total_context_switches: int = 0
        self.total_quanta_executed: int = 0
        self.total_tokens_generated: int = 0

    def _emit_terminal_packet(self, session: SchedSession, packet: Dict[str, Any]) -> None:
        """Safely deliver terminal packet to session queue without blocking scheduler loop."""
        try:
            session.token_queue.put_nowait(packet)
        except asyncio.QueueFull:
            try:
                session.token_queue.get_nowait()
            except Exception:
                pass
            try:
                session.token_queue.put_nowait(packet)
            except Exception:
                pass

    async def _cleanup_session_resources(self, session: SchedSession) -> None:
        """Safely release backend session KV cache, decrement active session counters, and prune queue."""
        loaded = self.model_manager.get_model(session.model_tag)
        if loaded:
            async with loaded.inference_lock:
                if hasattr(loaded, "batch_engine") and loaded.batch_engine:
                    loaded.batch_engine.release_session(session.session_id)
                try:
                    await asyncio.to_thread(loaded.backend.close_session, session.session_id)
                except Exception:
                    pass

            if session.admitted:
                if hasattr(loaded, "dec_active_sessions"):
                    loaded.dec_active_sessions()
                else:
                    loaded.active_sessions = max(0, loaded.active_sessions - 1)
                session.admitted = False
            session.slot_id = None

        if session.use_speculative and session.draft_model_tag:
            draft_loaded = self.model_manager.get_model(session.draft_model_tag)
            if draft_loaded:
                if hasattr(draft_loaded, "dec_active_sessions"):
                    draft_loaded.dec_active_sessions()
                else:
                    draft_loaded.active_sessions = max(0, draft_loaded.active_sessions - 1)
                async with draft_loaded.inference_lock:
                    try:
                        await asyncio.to_thread(draft_loaded.backend.close_session, session.session_id)
                    except Exception:
                        pass

        if self.enable_slo:
            target = None
            if session.slo_target_ttft_ms is not None or session.slo_target_tbt_ms is not None:
                target = SLOTarget(
                    target_ttft_ms=session.slo_target_ttft_ms or self.slo_controller.default_target_ttft_ms,
                    target_tbt_ms=session.slo_target_tbt_ms or self.slo_controller.default_target_tbt_ms,
                    strict_enforcement=session.strict_slo,
                )
            self.slo_controller.record_session_completion(session, target=target)

        async with self.lock:
            if self.enable_mlfq:
                self.mlfq.remove_session(session)
            if self.enable_fair_share:
                self.fair_share.register_session_completion(session)
            if session in self.runnable_queue:
                self.runnable_queue.remove(session)
            if session not in self.completed_sessions:
                self.completed_sessions.append(session)

    async def _step_continuous_batch(self, loaded_model: LoadedModel, primary_session: SchedSession) -> bool:
        """Step active slots in parallel for loaded_model using continuous batching."""
        engine = loaded_model.batch_engine
        assert engine is not None

        from llmhub.batch.engine import BatchSlotState

        # 1. Gather all runnable sessions targeting this model
        async with self.lock:
            model_sessions = []
            for s in self.runnable_queue:
                if s.is_done or s.cancelled:
                    continue
                if s.model_tag == loaded_model.tag:
                    model_sessions.append(s)
                elif hasattr(self.model_manager, "get_model") and self.model_manager.get_model(s.model_tag) == loaded_model:
                    s.model_tag = loaded_model.tag  # Canonicalize once to eliminate redundant queries
                    model_sessions.append(s)

        if not model_sessions:
            return False

        # 2. Admit waiting sessions into free slots in batch_engine
        for s in model_sessions:
            slot = engine.get_slot_by_session(s.session_id)
            if slot is None:
                free_slot = engine.find_free_slot()
                if free_slot is not None:
                    if not s.prompt_tokens and s.prompt:
                        try:
                            s.prompt_tokens = loaded_model.backend.tokenize(s.prompt)
                        except Exception:
                            s.prompt_tokens = list(s.prompt.encode("utf-8"))

                    allocated = engine.allocate_slot(
                        session_id=s.session_id,
                        prompt=s.prompt,
                        prompt_tokens=s.prompt_tokens or [],
                        max_tokens=s.max_tokens,
                        temperature=s.temperature,
                        top_p=s.top_p,
                        stop=s.stop,
                    )
                    if allocated:
                        s.slot_id = allocated.slot_id
                        s.metrics.start_time = s.metrics.start_time or time.monotonic()
                        s.admitted = True
                        if hasattr(loaded_model, "inc_active_sessions"):
                            loaded_model.inc_active_sessions()
                        else:
                            loaded_model.active_sessions += 1
                        loaded_model.last_active_time = time.monotonic()

        # 3. Execute batch step on background worker thread
        t_start = time.perf_counter()
        try:
            async with loaded_model.inference_lock:
                batch_res = await asyncio.to_thread(
                    engine.step_batch,
                    loaded_model.backend,
                    primary_session.quantum_tokens or self.default_quantum,
                    primary_session.chunk_size or self.default_chunk_size,
                )
        except Exception as e:
            logger.error("Inference batch step failed for model '%s': %s", loaded_model.tag, e, exc_info=True)
            for s in model_sessions:
                s.phase = SessionPhase.FAILED
                s.error = str(e)
                s.metrics.finish_time = time.monotonic()
                await s.token_queue.put({"done": True, "error": str(e)})
                await self._cleanup_session_resources(s)
            return True

        now = time.monotonic()
        cpu_duration_ms = (time.perf_counter() - t_start) * 1000.0
        loaded_model.last_active_time = now

        # Update initial phase transition for newly admitted prefilling slots
        for s in model_sessions:
            slot = engine.get_slot_by_session(s.session_id)
            if slot and slot.state == BatchSlotState.PREFILLING and s.phase == SessionPhase.WAITING:
                s.phase = SessionPhase.PREFILLING

        # 4. Dispatch prefill completions
        for sid in batch_res.prefill_completed_sessions:
            s = next((x for x in model_sessions if x.session_id == sid), None)
            if s:
                s.phase = SessionPhase.DECODING
                s.metrics.prefill_tokens = batch_res.prefilled_tokens_per_session.get(sid, len(s.prompt_tokens or []))
                s.metrics.total_cpu_time_ms += cpu_duration_ms
                try:
                    s.token_queue.put_nowait({
                        "type": "prefill_complete",
                        "prefilled_tokens": s.metrics.prefill_tokens
                    })
                except asyncio.QueueFull:
                    pass

        # 5. Dispatch generated token deltas to active decoding sessions
        for sid, deltas in batch_res.session_deltas.items():
            s = next((x for x in model_sessions if x.session_id == sid), None)
            if s:
                s.phase = SessionPhase.DECODING
                if s.metrics.first_token_time is None and len(deltas) > 0:
                    s.metrics.first_token_time = now
                s.metrics.last_quantum_time = now
                s.metrics.generated_tokens += len(deltas)
                s.metrics.quanta_executed += 1
                s.metrics.total_cpu_time_ms += cpu_duration_ms
                s.deficit -= len(deltas)
                self.total_quanta_executed += 1
                self.total_tokens_generated += len(deltas)

                if self.enable_mlfq:
                    self.mlfq.record_quantum_consumption(s, len(deltas))
                    s.mlfq_level = self.mlfq.session_levels.get(sid, 0)
                if self.enable_fair_share:
                    self.fair_share.record_execution(s.tenant_id, len(deltas))

                for delta in deltas:
                    s.generated_text += delta
                    try:
                        s.token_queue.put_nowait({
                            "delta": delta,
                            "done": False,
                            "session_id": sid,
                        })
                    except asyncio.QueueFull:
                        logger.warning("Session %s token_queue is full; client backpressure triggered", sid)

        # 6. Dispatch completed sessions
        for sid in batch_res.completed_sessions:
            s = next((x for x in model_sessions if x.session_id == sid), None)
            if s:
                comp = batch_res.session_completions.get(sid, {})
                s.phase = SessionPhase.COMPLETED
                s.stop_reason = comp.get("stop_reason", "stop")
                s.metrics.finish_time = now

                completion_pkt = {
                    "done": True,
                    "stop_reason": s.stop_reason,
                    "text": s.generated_text,
                    "usage": {
                        "prompt_tokens": comp.get("prompt_tokens", len(s.prompt_tokens or [])),
                        "prefix_matched_tokens": comp.get("prefix_matched_tokens", 0),
                        "completion_tokens": s.metrics.generated_tokens,
                        "total_tokens": comp.get("total_tokens", len(s.prompt_tokens or []) + s.metrics.generated_tokens),
                        "ttft_ms": round(s.metrics.ttft_ms or 0.0, 2),
                        "decode_tok_s": round(s.metrics.decode_tok_s, 2),
                        "quanta": s.metrics.quanta_executed,
                        "context_switches": s.metrics.context_switches,
                    }
                }
                try:
                    s.token_queue.put_nowait(completion_pkt)
                except asyncio.QueueFull:
                    try:
                        s.token_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        s.token_queue.put_nowait(completion_pkt)
                    except Exception:
                        pass

                await self._cleanup_session_resources(s)

        # 6b. Dispatch isolated slot failures without impacting other active sessions
        for sid, err_msg in batch_res.failed_sessions.items():
            logger.error("Session %s failed during batch step: %s", sid, err_msg)
            s = next((x for x in model_sessions if x.session_id == sid), None)
            if s and not s.is_done:
                s.phase = SessionPhase.FAILED
                s.error = err_msg
                s.metrics.finish_time = now
                try:
                    s.token_queue.put_nowait({"done": True, "error": err_msg})
                except Exception:
                    pass
                await self._cleanup_session_resources(s)

        # 7. Update status for prefilling sessions that did not complete prefill yet
        for s in model_sessions:
            slot = engine.get_slot_by_session(s.session_id)
            if slot and slot.state == BatchSlotState.PREFILLING and s.phase != SessionPhase.DECODING and not s.is_done:
                s.phase = SessionPhase.PREFILLING
                s.metrics.prefill_tokens = slot.prefill_offset

        # 8. Rotate sessions that participated in this turn
        async with self.lock:
            for s in model_sessions:
                if s in self.runnable_queue and not s.is_done:
                    self.runnable_queue.remove(s)
                    self.runnable_queue.append(s)
                    if self.enable_mlfq:
                        self.mlfq.rotate_session_in_level(s)

        return True

    async def submit_session(self, session: SchedSession) -> None:
        """Submit a new session into the runnable queue with backpressure protection and predictive SLO admission."""
        async with self.lock:
            # Backpressure guard
            active_count = len([s for s in self.runnable_queue if not s.is_done])
            if active_count >= self.max_queue_depth:
                raise QueueFullError(
                    f"Scheduler queue is full ({active_count}/{self.max_queue_depth} active sessions). Backpressure applied."
                )

            # Predictive SLO admission check
            if self.enable_slo and (session.strict_slo or session.slo_target_ttft_ms is not None):
                target = SLOTarget(
                    target_ttft_ms=session.slo_target_ttft_ms or self.slo_controller.default_target_ttft_ms,
                    target_tbt_ms=session.slo_target_tbt_ms or self.slo_controller.default_target_tbt_ms,
                    strict_enforcement=session.strict_slo,
                )
                self.slo_controller.check_admission(session, self.runnable_queue, slo_target=target)

            session.quantum_tokens = session.quantum_tokens or self.default_quantum
            session.chunk_size = session.chunk_size or self.default_chunk_size
            session.metrics.enqueue_time = time.monotonic()
            session.phase = SessionPhase.WAITING

            # Normalize alias model_tag to canonical loaded model tag if model is resident
            if hasattr(self.model_manager, "get_model"):
                lm = self.model_manager.get_model(session.model_tag)
                if lm and lm.tag:
                    session.model_tag = lm.tag

            if self.enable_fair_share:
                self.fair_share.register_session_admission(session, session.tenant_id)

            if self.enable_mlfq:
                self.mlfq.enqueue_session(session)
                session.mlfq_level = self.mlfq.session_levels.get(session.session_id, 0)

            self.runnable_queue.append(session)

    async def cancel_session(self, session_id: str) -> bool:
        """Mark a session as cancelled and release associated resources."""
        target: Optional[SchedSession] = None
        async with self.lock:
            for s in self.runnable_queue:
                if s.session_id == session_id:
                    s.cancelled = True
                    s.phase = SessionPhase.CANCELLED
                    s.metrics.finish_time = time.monotonic()
                    target = s
                    break

        if target:
            # Notify any awaiting reader
            self._emit_terminal_packet(target, {"done": True, "stop_reason": "cancelled", "error": "Session cancelled by client."})
            await self._cleanup_session_resources(target)
            return True
        return False

    def get_active_model_tags(self) -> Set[str]:
        """Return the set of models that currently have runnable sessions."""
        return {s.model_tag for s in self.runnable_queue if not s.is_done and not s.cancelled}

    async def step(self) -> bool:
        """Execute one scheduling turn: select next session, step quantum or prefill chunk.
        
        Returns True if a session step was executed, False if idle.
        """
        async with self.lock:
            # 1. Clean up expired / timed out sessions
            now = time.monotonic()
            expired = []
            for s in list(self.runnable_queue):
                if s.is_expired and not s.is_done:
                    s.phase = SessionPhase.TIMED_OUT
                    s.metrics.finish_time = now
                    s.error = "Session deadline exceeded."
                    expired.append(s)

        for s in expired:
            self._emit_terminal_packet(s, {"done": True, "stop_reason": "timeout", "error": s.error})
            await self._cleanup_session_resources(s)

        # 2. Re-balance CPU threading allocation for active models
        async with self.lock:
            active_tags = self.get_active_model_tags()
            thread_allocs = self.threading_strategy.calculate_threads(active_tags)

        for tag, threads in thread_allocs.items():
            loaded = self.model_manager.get_model(tag)
            if loaded and loaded.n_threads != threads:
                async with loaded.inference_lock:
                    loaded.n_threads = threads
                    loaded.backend.set_n_threads(threads)

        async with self.lock:
            # 3. Select next runnable session via MLFQ / Fair-Share / Deficit Round-Robin
            session: Optional[SchedSession] = None
            if self.enable_mlfq:
                session = self.mlfq.select_next_runnable()
            elif self.enable_fair_share:
                session = self.fair_share.select_next_tenant_session(self.runnable_queue)

            if not session:
                session = self.policy.select_next_session(self.runnable_queue)

            if not session:
                return False

            # Track context-switch telemetry
            if self.last_scheduled_model is not None and self.last_scheduled_model != session.model_tag:
                self.total_context_switches += 1
                session.metrics.context_switches += 1
            self.last_scheduled_model = session.model_tag

        # Retrieve loaded model
        loaded_model = self.model_manager.get_model(session.model_tag)
        if not loaded_model:
            # Trigger load if dynamic load required
            try:
                loaded_model = await self.model_manager.get_or_load_model(session.model_tag)
                session.model_tag = loaded_model.tag
            except Exception as e:
                session.phase = SessionPhase.FAILED
                session.error = str(e)
                session.metrics.finish_time = time.monotonic()
                await session.token_queue.put({"done": True, "error": str(e)})
                await self._cleanup_session_resources(session)
                return True
        elif session.model_tag != loaded_model.tag:
            session.model_tag = loaded_model.tag

        # Check if Continuous Batching is enabled and supported for this model (skip for speculative requests)
        is_speculative = session.use_speculative and bool(session.draft_model_tag)
        if (
            self.enable_continuous_batching
            and not is_speculative
            and hasattr(loaded_model, "batch_engine")
            and loaded_model.batch_engine is not None
        ):
            return await self._step_continuous_batch(loaded_model, session)

        # Phase 1: Initialize session on backend if in WAITING
        if session.phase == SessionPhase.WAITING:
            session.metrics.start_time = time.monotonic()
            session.phase = SessionPhase.PREFILLING
            session.admitted = True
            if hasattr(loaded_model, "inc_active_sessions"):
                loaded_model.inc_active_sessions()
            else:
                loaded_model.active_sessions += 1
            loaded_model.last_active_time = time.monotonic()

            try:
                async with loaded_model.inference_lock:
                    await asyncio.to_thread(
                        loaded_model.backend.new_session,
                        session_id=session.session_id,
                        prompt=session.prompt,
                        max_tokens=session.max_tokens,
                        temperature=session.temperature,
                        top_p=session.top_p,
                        stop=session.stop,
                    )
            except Exception as e:
                session.phase = SessionPhase.FAILED
                session.error = str(e)
                session.metrics.finish_time = time.monotonic()
                self._emit_terminal_packet(session, {"done": True, "error": str(e)})
                await self._cleanup_session_resources(session)
                return True

            # Rotate queue to preserve round-robin fairness across concurrent sessions
            async with self.lock:
                if session in self.runnable_queue:
                    self.runnable_queue.remove(session)
                    self.runnable_queue.append(session)
                if self.enable_mlfq:
                    self.mlfq.rotate_session_in_level(session)
            return True

        # Phase 2: Chunked Prefill (evaluate P tokens at a time)
        if session.phase == SessionPhase.PREFILLING:
            t_start = time.perf_counter()
            try:
                async with loaded_model.inference_lock:
                    tokens_prefilled, is_complete = await asyncio.to_thread(
                        loaded_model.backend.prefill_chunk,
                        session.session_id,
                        session.chunk_size
                    )
            except Exception as e:
                session.phase = SessionPhase.FAILED
                session.error = str(e)
                session.metrics.finish_time = time.monotonic()
                self._emit_terminal_packet(session, {"done": True, "error": str(e)})
                await self._cleanup_session_resources(session)
                return True

            session.metrics.total_cpu_time_ms += (time.perf_counter() - t_start) * 1000.0
            session.metrics.prefill_tokens += tokens_prefilled

            if is_complete:
                session.phase = SessionPhase.DECODING
                # Immediately emit initial acknowledgment so consumers know prefill finished
                try:
                    session.token_queue.put_nowait({
                        "type": "prefill_complete",
                        "prefilled_tokens": session.metrics.prefill_tokens
                    })
                except asyncio.QueueFull:
                    pass

            # Preemption point: yield turn and rotate so other sessions interleave during prefill!
            async with self.lock:
                if session in self.runnable_queue:
                    self.runnable_queue.remove(session)
                    self.runnable_queue.append(session)
                if self.enable_mlfq:
                    self.mlfq.rotate_session_in_level(session)
            return True

        # Phase 3: Token Quantum Decode (generate Q tokens at a time)
        if session.phase == SessionPhase.DECODING:
            remaining = session.max_tokens - session.metrics.generated_tokens
            tokens_to_gen = min(session.quantum_tokens, remaining)

            t_start = time.perf_counter()
            cpu_duration_ms = 0.0
            tokens_generated = 0
            text_deltas: List[str] = []
            is_done = False
            stop_reason: Optional[str] = None

            # Check if speculative decoding is requested and draft model is specified
            draft_model: Optional[LoadedModel] = None
            if session.use_speculative and session.draft_model_tag:
                try:
                    draft_model = await self.model_manager.get_or_load_model(session.draft_model_tag)
                except Exception as e:
                    logger.warning("Draft model %s load failed (%s); falling back to target decode", session.draft_model_tag, e)
                    draft_model = None

            if draft_model is not None:
                pair_key = f"{session.model_tag}:{session.draft_model_tag}"
                spec_engine = self.speculative_engines.setdefault(
                    pair_key,
                    SpeculativeEngine(session.model_tag, session.draft_model_tag)
                )

                # Ensure draft session is initialized
                if not draft_model.backend.get_session(session.session_id):
                    if hasattr(draft_model, "inc_active_sessions"):
                        draft_model.inc_active_sessions()
                    else:
                        draft_model.active_sessions += 1
                    draft_model.last_active_time = time.monotonic()
                    try:
                        async with draft_model.inference_lock:
                            await asyncio.to_thread(
                                draft_model.backend.new_session,
                                session_id=session.session_id,
                                prompt=session.prompt,
                                max_tokens=session.max_tokens,
                                temperature=session.temperature,
                                top_p=session.top_p,
                                stop=session.stop,
                            )
                            # Prefill draft model with cooperative yield
                            while True:
                                _, p_done = await asyncio.to_thread(
                                    draft_model.backend.prefill_chunk,
                                    session.session_id,
                                    session.chunk_size
                                )
                                if p_done:
                                    break
                                await asyncio.sleep(0)
                    except Exception as e:
                        logger.warning("Draft model initialization failed (%s); falling back to target decode", e)
                        draft_model.active_sessions = max(0, draft_model.active_sessions - 1)
                        draft_model = None

            if draft_model is not None:
                try:
                    if loaded_model.tag == draft_model.tag:
                        async with loaded_model.inference_lock:
                            spec_res = await asyncio.to_thread(
                                spec_engine.speculative_step,
                                session.session_id,
                                loaded_model.backend,
                                draft_model.backend,
                                min(tokens_to_gen, 4)
                            )
                    else:
                        first_m, second_m = sorted([loaded_model, draft_model], key=lambda m: m.tag)
                        async with first_m.inference_lock:
                            async with second_m.inference_lock:
                                spec_res = await asyncio.to_thread(
                                    spec_engine.speculative_step,
                                    session.session_id,
                                    loaded_model.backend,
                                    draft_model.backend,
                                    min(tokens_to_gen, 4)
                                )
                    cpu_duration_ms = (time.perf_counter() - t_start) * 1000.0
                    tokens_generated = len(spec_res.emitted_tokens)
                    text_deltas = spec_res.emitted_text_deltas
                    is_done = spec_res.done
                    stop_reason = spec_res.stop_reason
                    draft_model.last_active_time = time.monotonic()
                except Exception as e:
                    logger.warning("Speculative step failed (%s); falling back to regular decode", e)
                    draft_model = None

            if draft_model is None:
                try:
                    async with loaded_model.inference_lock:
                        res = await asyncio.to_thread(
                            loaded_model.backend.decode_step,
                            session.session_id,
                            tokens_to_gen
                        )
                    cpu_duration_ms = (time.perf_counter() - t_start) * 1000.0
                    tokens_generated = len(res.tokens)
                    text_deltas = res.text_deltas
                    is_done = res.done
                    stop_reason = res.stop_reason
                except Exception as e:
                    session.phase = SessionPhase.FAILED
                    session.error = str(e)
                    session.metrics.finish_time = time.monotonic()
                    self._emit_terminal_packet(session, {"done": True, "error": str(e)})
                    await self._cleanup_session_resources(session)
                    return True

            # Record First-Token Time (TTFT)
            now = time.monotonic()
            if session.metrics.first_token_time is None and tokens_generated > 0:
                session.metrics.first_token_time = now

            session.metrics.total_cpu_time_ms += cpu_duration_ms
            session.metrics.last_quantum_time = now
            session.metrics.generated_tokens += tokens_generated
            session.metrics.quanta_executed += 1
            self.total_quanta_executed += 1
            self.total_tokens_generated += tokens_generated
            session.deficit -= tokens_generated
            loaded_model.last_active_time = now

            # MLFQ & Fair-Share tracking
            if self.enable_mlfq:
                self.mlfq.record_quantum_consumption(session, tokens_generated)
                session.mlfq_level = self.mlfq.session_levels.get(session.session_id, 0)
            if self.enable_fair_share:
                self.fair_share.record_execution(session.tenant_id, tokens_generated)

            # Stream generated tokens to consumer queue (non-blocking)
            for delta in text_deltas:
                session.generated_text += delta
                try:
                    session.token_queue.put_nowait({
                        "delta": delta,
                        "done": False,
                        "session_id": session.session_id,
                    })
                except asyncio.QueueFull:
                    logger.warning("Session %s token_queue is full; client backpressure triggered", session.session_id)

            # Check termination condition
            hit_max_tokens = session.metrics.generated_tokens >= session.max_tokens
            if is_done or hit_max_tokens:
                session.phase = SessionPhase.COMPLETED
                session.stop_reason = stop_reason if not hit_max_tokens else "length"
                session.metrics.finish_time = now

                # Emit final completion packet with usage statistics
                completion_pkt = {
                    "done": True,
                    "stop_reason": session.stop_reason,
                    "text": session.generated_text,
                    "usage": {
                        "prompt_tokens": session.metrics.prefill_tokens,
                        "completion_tokens": session.metrics.generated_tokens,
                        "total_tokens": session.metrics.prefill_tokens + session.metrics.generated_tokens,
                        "ttft_ms": round(session.metrics.ttft_ms or 0.0, 2),
                        "decode_tok_s": round(session.metrics.decode_tok_s, 2),
                        "quanta": session.metrics.quanta_executed,
                        "context_switches": session.metrics.context_switches,
                    }
                }
                self._emit_terminal_packet(session, completion_pkt)

                # Safely release backend session, decrement active_sessions, and move to completed history
                await self._cleanup_session_resources(session)
            else:
                # Rotate still-active session to the back of the queue
                async with self.lock:
                    if session in self.runnable_queue:
                        self.runnable_queue.remove(session)
                        self.runnable_queue.append(session)
                    if self.enable_mlfq:
                        self.mlfq.rotate_session_in_level(session)

            # Preemption boundary: step finishes here and yields to runnable queue
            return True

        return False

    async def _schedule_loop(self) -> None:
        """Background coroutine continuously driving token round-robin scheduling."""
        while self.running:
            try:
                stepped = await self.step()
                if not stepped:
                    await asyncio.sleep(0.002)  # 2ms idle backoff
                else:
                    await asyncio.sleep(0)      # Cooperative yield
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Unexpected error in TokenRoundRobinScheduler loop: %s", e)
                await asyncio.sleep(0.01)

    def start(self) -> None:
        """Start the background scheduler task."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        task_active = (
            self._loop_task is not None
            and not self._loop_task.done()
            and (current_loop is None or self._loop_task.get_loop() == current_loop)
        )

        if not self.running or not task_active:
            self.running = True
            self._loop_task = asyncio.create_task(self._schedule_loop())

    def stop(self) -> None:
        """Stop the background scheduler task."""
        self.running = False
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            self._loop_task = None

    def set_thermal_thread_limit(self, max_threads: Optional[int]) -> None:
        """Dynamically cap maximum CPU threads across all models due to thermal constraints."""
        self.threading_strategy.set_max_threads_cap(max_threads)

    def set_threading_mode(self, mode: ThreadingMode) -> None:
        """Switch CPU threading mode dynamically."""
        self.threading_strategy.mode = mode

    def get_telemetry(self) -> Dict[str, Any]:
        """Aggregate real-time metrics across all active and completed sessions."""
        all_completed = list(self.completed_sessions)
        throughputs = [s.metrics.decode_tok_s for s in all_completed if s.metrics.decode_tok_s > 0]
        weights = [s.weight for s in all_completed if s.metrics.decode_tok_s > 0]

        jains_index = self.policy.compute_jains_fairness_index(throughputs, weights)
        ttft_values = [s.metrics.ttft_ms for s in all_completed if s.metrics.ttft_ms is not None]

        p50_ttft = 0.0
        p95_ttft = 0.0
        if ttft_values:
            sorted_ttft = sorted(ttft_values)
            p50_ttft = sorted_ttft[len(sorted_ttft) // 2]
            p95_ttft = sorted_ttft[min(len(sorted_ttft) - 1, int(len(sorted_ttft) * 0.95))]

        return {
            "queue_depth": len([s for s in self.runnable_queue if not s.is_done]),
            "active_models": list(self.get_active_model_tags()),
            "threading_mode": self.threading_strategy.mode.value,
            "total_quanta_executed": self.total_quanta_executed,
            "total_tokens_generated": self.total_tokens_generated,
            "total_context_switches": self.total_context_switches,
            "jains_fairness_index": round(jains_index, 4),
            "ttft_p50_ms": round(p50_ttft, 2),
            "ttft_p95_ms": round(p95_ttft, 2),
            "completed_sessions_count": len(all_completed),
            "mlfq": self.mlfq.get_stats() if self.enable_mlfq else None,
            "slo": self.slo_controller.get_stats() if self.enable_slo else None,
            "fair_share": self.fair_share.get_stats() if self.enable_fair_share else None,
            "speculative": {k: eng.get_stats() for k, eng in self.speculative_engines.items()},
        }
