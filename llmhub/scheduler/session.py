"""SchedSession and metrics for token-level round-robin scheduler."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional

class SessionPhase(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"

class SessionPriority(int, Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10

@dataclass
class SessionMetrics:
    enqueue_time: float = field(default_factory=time.monotonic)
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    last_quantum_time: float = field(default_factory=time.monotonic)
    finish_time: Optional[float] = None
    prefill_tokens: int = 0
    generated_tokens: int = 0
    quanta_executed: int = 0
    context_switches: int = 0
    total_cpu_time_ms: float = 0.0

    @property
    def queue_wait_ms(self) -> float:
        if self.start_time is None:
            return (time.monotonic() - self.enqueue_time) * 1000.0
        return (self.start_time - self.enqueue_time) * 1000.0

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.enqueue_time) * 1000.0

    @property
    def total_latency_ms(self) -> Optional[float]:
        if self.finish_time is None:
            return (time.monotonic() - self.enqueue_time) * 1000.0
        return (self.finish_time - self.enqueue_time) * 1000.0

    @property
    def decode_tok_s(self) -> float:
        if (
            self.first_token_time is None
            or self.finish_time is None
            or self.finish_time <= self.first_token_time
        ):
            return 0.0
        decode_duration = self.finish_time - self.first_token_time
        return self.generated_tokens / decode_duration if decode_duration > 0 else 0.0

@dataclass
class SchedSession:
    session_id: str
    model_tag: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    priority: int = 5
    weight: float = 1.0
    quantum_tokens: int = 4
    chunk_size: int = 256
    timeout_seconds: float = 120.0
    deficit: float = 0.0
    phase: SessionPhase = SessionPhase.WAITING
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    token_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=512))
    cancelled: bool = False
    stop_reason: Optional[str] = None
    error: Optional[str] = None
    generated_text: str = ""
    prompt_tokens: Optional[List[int]] = None
    slot_id: Optional[int] = None
    admitted: bool = False
    tenant_id: str = "default"
    slo_target_ttft_ms: Optional[float] = None
    slo_target_tbt_ms: Optional[float] = None
    strict_slo: bool = False
    mlfq_level: int = 0
    use_speculative: bool = False
    draft_model_tag: Optional[str] = None

    def __post_init__(self) -> None:
        self.weight = max(0.01, float(self.weight))
        self.priority = max(1, min(10, int(self.priority)))
        if self.token_queue is None:
            self.token_queue = asyncio.Queue(maxsize=512)
        if self.prompt_tokens is None:
            self.prompt_tokens = list(self.prompt.encode("utf-8")) if self.prompt else []

    @property
    def is_done(self) -> bool:
        return self.phase in (
            SessionPhase.COMPLETED,
            SessionPhase.CANCELLED,
            SessionPhase.TIMED_OUT,
            SessionPhase.FAILED,
        )

    @property
    def is_expired(self) -> bool:
        if self.is_done:
            return False
        return (time.monotonic() - self.metrics.enqueue_time) > self.timeout_seconds
