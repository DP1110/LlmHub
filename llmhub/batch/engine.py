"""Continuous Batching Engine with Parallel Sequence Slots and Shared Weights Context."""

from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from llmhub.batch.prefix_cache import PrefixCache
from llmhub.server.backend import Backend, DecodeResult

class BatchSlotState(str, Enum):
    FREE = "free"
    ALLOCATED = "allocated"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BatchSlot:
    """Represents an execution slot inside a shared-weights model context."""
    slot_id: int
    seq_id: int
    state: BatchSlotState = BatchSlotState.FREE
    session_id: Optional[str] = None
    prompt: str = ""
    prompt_tokens: List[int] = field(default_factory=list)
    prefill_offset: int = 0
    prefix_matched_tokens: int = 0
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    done: bool = False
    stop_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.state in (BatchSlotState.ALLOCATED, BatchSlotState.PREFILLING, BatchSlotState.DECODING)

    def reset(self) -> None:
        """Reset slot back to clean FREE state."""
        self.state = BatchSlotState.FREE
        self.session_id = None
        self.prompt = ""
        self.prompt_tokens.clear()
        self.prefill_offset = 0
        self.prefix_matched_tokens = 0
        self.generated_tokens.clear()
        self.generated_text = ""
        self.max_tokens = 512
        self.stop = None
        self.start_time = None
        self.first_token_time = None
        self.last_token_time = None
        self.finish_time = None
        self.done = False
        self.stop_reason = None
        self.error = None

@dataclass
class BatchStepResult:
    """Summary metrics of a continuous batch step."""
    model_tag: str
    active_slots_count: int
    prefilling_slots_count: int
    decoding_slots_count: int
    total_prefill_tokens: int
    total_generated_tokens: int
    completed_sessions: List[str]
    session_deltas: Dict[str, List[str]]
    session_completions: Dict[str, Dict[str, Any]]
    prefill_completed_sessions: List[str] = field(default_factory=list)
    prefilled_tokens_per_session: Dict[str, int] = field(default_factory=dict)
    failed_sessions: Dict[str, str] = field(default_factory=dict)
    execution_duration_ms: float = 0.0

class ContinuousBatchEngine:
    """Manages parallel sequence slots and continuous iteration-level batch execution."""

    def __init__(
        self,
        model_tag: str,
        max_slots: int = 4,
        prefix_cache: Optional[PrefixCache] = None,
    ):
        self.model_tag = model_tag
        self.max_slots = max(1, max_slots)
        self.prefix_cache = prefix_cache or PrefixCache()
        self.slots: List[BatchSlot] = [
            BatchSlot(slot_id=i, seq_id=i) for i in range(self.max_slots)
        ]
        self._lock = threading.Lock()

        # Telemetry
        self.total_sessions_admitted: int = 0
        self.total_sessions_completed: int = 0
        self.total_batch_steps: int = 0
        self.total_tokens_generated: int = 0
        self.total_prefill_tokens: int = 0

    def find_free_slot(self) -> Optional[BatchSlot]:
        """Return first available FREE slot or None if all slots are saturated."""
        with self._lock:
            for slot in self.slots:
                if slot.state == BatchSlotState.FREE:
                    return slot
            return None

    def get_slot_by_session(self, session_id: str) -> Optional[BatchSlot]:
        """Find slot currently bound to session_id."""
        with self._lock:
            for slot in self.slots:
                if slot.session_id == session_id:
                    return slot
            return None

    def allocate_slot(
        self,
        session_id: str,
        prompt: str,
        prompt_tokens: List[int],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Optional[BatchSlot]:
        """Admit a session into a free slot, querying prefix cache for common prompt sequences."""
        with self._lock:
            # Find free slot inside lock
            slot: Optional[BatchSlot] = None
            for s in self.slots:
                if s.state == BatchSlotState.FREE:
                    slot = s
                    break

            if not slot:
                return None

            slot.reset()
            slot.session_id = session_id
            slot.prompt = prompt
            slot.prompt_tokens = list(prompt_tokens)
            slot.max_tokens = max_tokens
            slot.temperature = temperature
            slot.top_p = top_p
            slot.stop = stop
            slot.start_time = time.monotonic()

            # Query Prefix Cache for matching prompt prefix
            matched_tokens, cache_entry = self.prefix_cache.find_longest_prefix(
                self.model_tag,
                slot.prompt_tokens
            )
            slot.prefix_matched_tokens = matched_tokens
            slot.prefill_offset = matched_tokens  # Fast-forward past matched prefix!
            slot.state = BatchSlotState.ALLOCATED

            self.total_sessions_admitted += 1
            return slot

    def free_slot(self, slot_id: int) -> None:
        """Release slot resources and save reusable prompt prefix to cache."""
        with self._lock:
            if 0 <= slot_id < len(self.slots):
                slot = self.slots[slot_id]
                if slot.prompt_tokens and len(slot.prompt_tokens) >= self.prefix_cache.min_prefix_tokens:
                    # Cache prompt tokens for future sessions
                    self.prefix_cache.insert_prefix(self.model_tag, slot.prompt_tokens)
                slot.reset()

    def release_session(self, session_id: str) -> None:
        """Release slot associated with session_id."""
        with self._lock:
            for slot in self.slots:
                if slot.session_id == session_id:
                    if slot.prompt_tokens and len(slot.prompt_tokens) >= self.prefix_cache.min_prefix_tokens:
                        self.prefix_cache.insert_prefix(self.model_tag, slot.prompt_tokens)
                    slot.reset()
                    break

    def step_batch(
        self,
        backend: Backend,
        default_quantum: int = 4,
        chunk_size: int = 256,
    ) -> BatchStepResult:
        """Execute one continuous batch step across all active slots with per-slot error isolation."""
        t_start = time.perf_counter()

        with self._lock:
            active_slots = [s for s in self.slots if s.is_active]

            # 0. Check for freshly ALLOCATED slots and initialize backend sessions
            allocated_slots = [s for s in active_slots if s.state == BatchSlotState.ALLOCATED]
            prefill_completed_sessions: List[str] = []
            prefilled_tokens_per_session: Dict[str, int] = {}
            failed_sessions: Dict[str, str] = {}

            if allocated_slots:
                for slot in allocated_slots:
                    if not slot.session_id:
                        continue
                    try:
                        if not backend.get_session(slot.session_id):
                            backend.new_session(
                                session_id=slot.session_id,
                                prompt=slot.prompt,
                                max_tokens=slot.max_tokens,
                                temperature=slot.temperature,
                                top_p=slot.top_p,
                                stop=slot.stop,
                                prompt_tokens=slot.prompt_tokens,
                                seq_id=slot.seq_id,
                            )
                            backend_sess = backend.get_session(slot.session_id)
                            if backend_sess and slot.prefix_matched_tokens > 0:
                                backend_sess.prefill_offset = slot.prefix_matched_tokens

                        if slot.prefill_offset >= len(slot.prompt_tokens) and len(slot.prompt_tokens) > 0:
                            slot.state = BatchSlotState.DECODING
                            prefill_completed_sessions.append(slot.session_id)
                            prefilled_tokens_per_session[slot.session_id] = slot.prefill_offset
                        else:
                            slot.state = BatchSlotState.PREFILLING
                    except Exception as e:
                        slot.state = BatchSlotState.FAILED
                        slot.error = str(e)
                        slot.done = True
                        failed_sessions[slot.session_id] = str(e)

                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                return BatchStepResult(
                    model_tag=self.model_tag,
                    active_slots_count=len(active_slots),
                    prefilling_slots_count=len([s for s in self.slots if s.state == BatchSlotState.PREFILLING]),
                    decoding_slots_count=len([s for s in self.slots if s.state == BatchSlotState.DECODING]),
                    total_prefill_tokens=0,
                    total_generated_tokens=0,
                    completed_sessions=[],
                    session_deltas={},
                    session_completions={},
                    prefill_completed_sessions=prefill_completed_sessions,
                    prefilled_tokens_per_session=prefilled_tokens_per_session,
                    failed_sessions=failed_sessions,
                    execution_duration_ms=round(elapsed_ms, 3),
                )

            prefilling_slots = [s for s in active_slots if s.state == BatchSlotState.PREFILLING]
            decoding_slots = [s for s in active_slots if s.state == BatchSlotState.DECODING]

            total_prefilled = 0
            total_generated = 0
            completed_sessions: List[str] = []
            session_deltas: Dict[str, List[str]] = {}
            session_completions: Dict[str, Dict[str, Any]] = {}

            # 1. Step PREFILLING slots with per-slot exception isolation
            for slot in prefilling_slots:
                if not slot.session_id:
                    continue

                try:
                    # Ensure backend has session state registered
                    if not backend.get_session(slot.session_id):
                        backend.new_session(
                            session_id=slot.session_id,
                            prompt=slot.prompt,
                            max_tokens=slot.max_tokens,
                            temperature=slot.temperature,
                            top_p=slot.top_p,
                            stop=slot.stop,
                            prompt_tokens=slot.prompt_tokens,
                            seq_id=slot.seq_id,
                        )
                        backend_sess = backend.get_session(slot.session_id)
                        if backend_sess and slot.prefix_matched_tokens > 0:
                            backend_sess.prefill_offset = slot.prefix_matched_tokens

                    tokens_prefilled, is_done = backend.prefill_chunk(slot.session_id, chunk_size)
                    slot.prefill_offset += tokens_prefilled
                    total_prefilled += tokens_prefilled
                    prefilled_tokens_per_session[slot.session_id] = slot.prefill_offset

                    if is_done or slot.prefill_offset >= len(slot.prompt_tokens):
                        slot.state = BatchSlotState.DECODING
                        if slot.session_id not in prefill_completed_sessions:
                            prefill_completed_sessions.append(slot.session_id)
                except Exception as e:
                    slot.state = BatchSlotState.FAILED
                    slot.error = str(e)
                    slot.done = True
                    failed_sessions[slot.session_id] = str(e)

            # 2. Step DECODING slots in parallel with shared weights streaming
            for slot in decoding_slots:
                if not slot.session_id:
                    continue

                try:
                    remaining = slot.max_tokens - len(slot.generated_tokens)
                    n_tokens = min(default_quantum, remaining)

                    decode_res = backend.decode_step(slot.session_id, n_tokens)
                    now = time.monotonic()
                    if slot.first_token_time is None and len(decode_res.tokens) > 0:
                        slot.first_token_time = now
                    slot.last_token_time = now

                    slot.generated_tokens.extend(decode_res.tokens)
                    for d in decode_res.text_deltas:
                        slot.generated_text += d

                    session_deltas[slot.session_id] = decode_res.text_deltas
                    total_generated += len(decode_res.tokens)

                    # Check termination
                    hit_max = len(slot.generated_tokens) >= slot.max_tokens
                    if decode_res.done or hit_max:
                        slot.state = BatchSlotState.COMPLETED
                        slot.done = True
                        slot.stop_reason = decode_res.stop_reason if not hit_max else "length"
                        slot.finish_time = now
                        completed_sessions.append(slot.session_id)
                        self.total_sessions_completed += 1

                        # Compile completion telemetry
                        total_duration_s = (slot.finish_time - (slot.start_time or slot.finish_time))
                        decode_duration_s = (slot.finish_time - (slot.first_token_time or slot.finish_time))
                        decode_tok_s = (len(slot.generated_tokens) / decode_duration_s) if decode_duration_s > 0.001 else 0.0

                        session_completions[slot.session_id] = {
                            "text": slot.generated_text,
                            "stop_reason": slot.stop_reason,
                            "prompt_tokens": len(slot.prompt_tokens),
                            "prefix_matched_tokens": slot.prefix_matched_tokens,
                            "completion_tokens": len(slot.generated_tokens),
                            "total_tokens": len(slot.prompt_tokens) + len(slot.generated_tokens),
                            "ttft_ms": round(((slot.first_token_time or now) - (slot.start_time or now)) * 1000.0, 2),
                            "decode_tok_s": round(decode_tok_s, 2),
                            "duration_ms": round(total_duration_s * 1000.0, 2),
                        }
                except Exception as e:
                    slot.state = BatchSlotState.FAILED
                    slot.error = str(e)
                    slot.done = True
                    failed_sessions[slot.session_id] = str(e)

            self.total_batch_steps += 1
            self.total_tokens_generated += total_generated
            self.total_prefill_tokens += total_prefilled
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            return BatchStepResult(
                model_tag=self.model_tag,
                active_slots_count=len([s for s in self.slots if s.is_active]),
                prefilling_slots_count=len([s for s in self.slots if s.state == BatchSlotState.PREFILLING]),
                decoding_slots_count=len([s for s in self.slots if s.state == BatchSlotState.DECODING]),
                total_prefill_tokens=total_prefilled,
                total_generated_tokens=total_generated,
                completed_sessions=completed_sessions,
                session_deltas=session_deltas,
                session_completions=session_completions,
                prefill_completed_sessions=prefill_completed_sessions,
                prefilled_tokens_per_session=prefilled_tokens_per_session,
                failed_sessions=failed_sessions,
                execution_duration_ms=round(elapsed_ms, 3),
            )

    def get_status(self) -> Dict[str, Any]:
        """Return real-time status of all slots in the batch engine."""
        with self._lock:
            slot_dicts = []
            for s in self.slots:
                slot_dicts.append({
                    "slot_id": s.slot_id,
                    "seq_id": s.seq_id,
                    "state": s.state.value,
                    "session_id": s.session_id,
                    "prompt_tokens_len": len(s.prompt_tokens),
                    "prefill_offset": s.prefill_offset,
                    "prefix_matched_tokens": s.prefix_matched_tokens,
                    "generated_tokens_count": len(s.generated_tokens),
                    "max_tokens": s.max_tokens,
                    "is_active": s.is_active,
                })

            active_count = len([s for s in self.slots if s.is_active])
            utilization = (active_count / self.max_slots) * 100.0 if self.max_slots > 0 else 0.0

            return {
                "model_tag": self.model_tag,
                "max_slots": self.max_slots,
                "active_slots": active_count,
                "utilization_pct": round(utilization, 1),
                "total_sessions_admitted": self.total_sessions_admitted,
                "total_sessions_completed": self.total_sessions_completed,
                "total_batch_steps": self.total_batch_steps,
                "total_tokens_generated": self.total_tokens_generated,
                "prefix_cache": self.prefix_cache.get_stats(),
                "slots": slot_dicts,
            }
