"""Token-level Round-Robin Scheduler with Chunked Prefill and Preemption Points."""

import asyncio
from collections import deque
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Set

from llmhub.scheduler.policy import (
    DeficitRoundRobinPolicy,
    ThreadingMode,
    ThreadingStrategy,
)
from llmhub.scheduler.session import SchedSession, SessionPhase
from llmhub.server.model_manager import LoadedModel, ModelManager

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
    ):
        self.model_manager = model_manager
        self.default_quantum = default_quantum
        self.default_chunk_size = default_chunk_size
        self.max_queue_depth = max_queue_depth
        self.threading_strategy = ThreadingStrategy(mode=threading_mode)
        self.policy = DeficitRoundRobinPolicy(default_quantum=default_quantum)

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

    async def _cleanup_session_resources(self, session: SchedSession) -> None:
        """Safely release backend session KV cache, decrement active session counters, and prune queue."""
        loaded = self.model_manager.get_model(session.model_tag)
        if loaded:
            try:
                async with loaded.inference_lock:
                    await asyncio.to_thread(loaded.backend.close_session, session.session_id)
            except Exception:
                pass
            loaded.active_sessions = max(0, loaded.active_sessions - 1)

        async with self.lock:
            if session in self.runnable_queue:
                self.runnable_queue.remove(session)
            if session not in self.completed_sessions:
                self.completed_sessions.append(session)

    async def submit_session(self, session: SchedSession) -> None:
        """Submit a new session into the runnable queue with backpressure protection."""
        async with self.lock:
            # Backpressure guard
            active_count = len([s for s in self.runnable_queue if not s.is_done])
            if active_count >= self.max_queue_depth:
                raise QueueFullError(
                    f"Scheduler queue is full ({active_count}/{self.max_queue_depth} active sessions). Backpressure applied."
                )

            session.quantum_tokens = session.quantum_tokens or self.default_quantum
            session.chunk_size = session.chunk_size or self.default_chunk_size
            session.metrics.enqueue_time = time.monotonic()
            session.phase = SessionPhase.WAITING

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
            await target.token_queue.put({"done": True, "stop_reason": "cancelled", "error": "Session cancelled by client."})
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
            await s.token_queue.put({"done": True, "stop_reason": "timeout", "error": s.error})
            await self._cleanup_session_resources(s)

        async with self.lock:
            # 2. Re-balance CPU threading allocation for active models
            active_tags = self.get_active_model_tags()
            thread_allocs = self.threading_strategy.calculate_threads(active_tags)
            for tag, threads in thread_allocs.items():
                loaded = self.model_manager.get_model(tag)
                if loaded and loaded.n_threads != threads:
                    loaded.n_threads = threads
                    loaded.backend.set_n_threads(threads)

            # 3. Select next runnable session via Deficit Round-Robin
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
            except Exception as e:
                session.phase = SessionPhase.FAILED
                session.error = str(e)
                session.metrics.finish_time = time.monotonic()
                await session.token_queue.put({"done": True, "error": str(e)})
                await self._cleanup_session_resources(session)
                return True

        # Phase 1: Initialize session on backend if in WAITING
        if session.phase == SessionPhase.WAITING:
            session.metrics.start_time = time.monotonic()
            session.phase = SessionPhase.PREFILLING
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
                await session.token_queue.put({"done": True, "error": str(e)})
                await self._cleanup_session_resources(session)
                return True

            # Rotate queue to preserve round-robin fairness across concurrent sessions
            async with self.lock:
                if session in self.runnable_queue:
                    self.runnable_queue.remove(session)
                    self.runnable_queue.append(session)
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
                await session.token_queue.put({"done": True, "error": str(e)})
                await self._cleanup_session_resources(session)
                return True

            session.metrics.total_cpu_time_ms += (time.perf_counter() - t_start) * 1000.0
            session.metrics.prefill_tokens += tokens_prefilled

            if is_complete:
                session.phase = SessionPhase.DECODING
                # Immediately emit initial acknowledgment so consumers know prefill finished
                await session.token_queue.put({
                    "type": "prefill_complete",
                    "prefilled_tokens": session.metrics.prefill_tokens
                })

            # Preemption point: yield turn and rotate so other sessions interleave during prefill!
            async with self.lock:
                if session in self.runnable_queue:
                    self.runnable_queue.remove(session)
                    self.runnable_queue.append(session)
            return True

        # Phase 3: Token Quantum Decode (generate Q tokens at a time)
        if session.phase == SessionPhase.DECODING:
            remaining = session.max_tokens - session.metrics.generated_tokens
            tokens_to_gen = min(session.quantum_tokens, remaining)

            t_start = time.perf_counter()
            try:
                async with loaded_model.inference_lock:
                    res = await asyncio.to_thread(
                        loaded_model.backend.decode_step,
                        session.session_id,
                        tokens_to_gen
                    )
            except Exception as e:
                session.phase = SessionPhase.FAILED
                session.error = str(e)
                session.metrics.finish_time = time.monotonic()
                await session.token_queue.put({"done": True, "error": str(e)})
                await self._cleanup_session_resources(session)
                return True

            cpu_duration_ms = (time.perf_counter() - t_start) * 1000.0

            # Record First-Token Time (TTFT)
            now = time.monotonic()
            if session.metrics.first_token_time is None and len(res.tokens) > 0:
                session.metrics.first_token_time = now

            session.metrics.total_cpu_time_ms += cpu_duration_ms
            session.metrics.last_quantum_time = now
            session.metrics.generated_tokens += len(res.tokens)
            session.metrics.quanta_executed += 1
            self.total_quanta_executed += 1
            self.total_tokens_generated += len(res.tokens)
            session.deficit -= len(res.tokens)
            loaded_model.last_active_time = now

            # Stream generated tokens to consumer queue
            for delta in res.text_deltas:
                session.generated_text += delta
                await session.token_queue.put({
                    "delta": delta,
                    "done": False,
                    "session_id": session.session_id,
                })

            # Check termination condition
            hit_max_tokens = session.metrics.generated_tokens >= session.max_tokens
            if res.done or hit_max_tokens:
                session.phase = SessionPhase.COMPLETED
                session.stop_reason = res.stop_reason if not hit_max_tokens else "length"
                session.metrics.finish_time = now

                # Emit final completion packet with usage statistics
                await session.token_queue.put({
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
                })

                # Safely release backend session, decrement active_sessions, and move to completed history
                await self._cleanup_session_resources(session)
            else:
                # Rotate still-active session to the back of the queue
                async with self.lock:
                    if session in self.runnable_queue:
                        self.runnable_queue.remove(session)
                        self.runnable_queue.append(session)

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
        }
