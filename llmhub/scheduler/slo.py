"""Service Level Objective (SLO) Target Tracking and Predictive Admission Control.

Targets:
- TTFT (Time To First Token): e.g. TTFT < 2000ms (2 seconds)
- TBT (Time Between Tokens) / Decode Latency: e.g. TBT < 50ms (min 20 tok/s)
- Predictive Admission: Estimates queue delay + prefill duration before admitting.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

from llmhub.scheduler.session import SchedSession

class SLOViolationError(Exception):
    """Raised when an incoming session cannot be served within its SLO constraints."""
    def __init__(self, message: str, estimated_ttft_ms: float, target_ttft_ms: float):
        super().__init__(message)
        self.estimated_ttft_ms = estimated_ttft_ms
        self.target_ttft_ms = target_ttft_ms

@dataclass
class SLOTarget:
    """Configured SLO constraints for a session or tenant."""
    target_ttft_ms: float = 2000.0  # 2 seconds default TTFT
    target_tbt_ms: float = 50.0     # 50 ms default Time Between Tokens
    strict_enforcement: bool = True
    admission_timestamp: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.target_ttft_ms = max(1.0, float(self.target_ttft_ms))
        self.target_tbt_ms = max(1.0, float(self.target_tbt_ms))

    @property
    def deadline_timestamp(self) -> float:
        """Absolute monotonic timestamp by which first token must arrive."""
        return self.admission_timestamp + (self.target_ttft_ms / 1000.0)

class SLOAdmissionController:
    """Evaluates queue latency and predicts TTFT before admitting new sessions."""

    def __init__(
        self,
        default_target_ttft_ms: float = 2000.0,
        default_target_tbt_ms: float = 50.0,
        nominal_prefill_tok_s: float = 120.0,
        nominal_decode_tok_s: float = 35.0,
    ):
        self.default_target_ttft_ms = max(1.0, float(default_target_ttft_ms))
        self.default_target_tbt_ms = max(1.0, float(default_target_tbt_ms))
        self.nominal_prefill_tok_s = max(1.0, float(nominal_prefill_tok_s))
        self.nominal_decode_tok_s = max(1.0, float(nominal_decode_tok_s))

        # Telemetry
        self.total_evaluated: int = 0
        self.total_admitted: int = 0
        self.total_rejected: int = 0
        self.slo_met_count: int = 0
        self.slo_violated_count: int = 0

    def predict_ttft_ms(
        self,
        new_prompt_tokens: int,
        active_sessions: List[SchedSession],
    ) -> float:
        """Predict expected Time To First Token for an incoming request.
        
        Formula:
            Queue Latency = Active Preceding Sessions * Quantum Decode Duration
            Prefill Latency = Total Pending Prefill Tokens / Prefill Throughput
            Estimated TTFT = Queue Latency + Prefill Latency + First Token Latency
        """
        # Sum pending prefill work from all waiting/prefilling sessions
        pending_prefill_tokens = max(0, new_prompt_tokens)
        for s in active_sessions:
            if not s.is_done and s.metrics.first_token_time is None:
                total_pt = len(s.prompt_tokens or [])
                remaining = max(0, total_pt - s.metrics.prefill_tokens)
                pending_prefill_tokens += remaining

        prefill_spd = max(1e-5, self.nominal_prefill_tok_s)
        decode_spd = max(1e-5, self.nominal_decode_tok_s)

        prefill_latency_s = pending_prefill_tokens / prefill_spd

        # Active decoding turns delay: each active decoding session takes 1 quantum turn
        active_decoding = len([s for s in active_sessions if not s.is_done and s.metrics.first_token_time is not None])
        queue_delay_s = (active_decoding * 4) / decode_spd

        first_token_decode_s = 1.0 / decode_spd
        total_estimated_s = prefill_latency_s + queue_delay_s + first_token_decode_s

        return round(total_estimated_s * 1000.0, 2)

    def check_admission(
        self,
        session: SchedSession,
        active_sessions: List[SchedSession],
        slo_target: Optional[SLOTarget] = None,
    ) -> bool:
        """Evaluate admission control based on predictive TTFT against SLO targets."""
        self.total_evaluated += 1
        target = slo_target or SLOTarget(
            target_ttft_ms=self.default_target_ttft_ms,
            target_tbt_ms=self.default_target_tbt_ms,
            strict_enforcement=True,
        )

        prompt_len = len(session.prompt_tokens or [])
        estimated_ttft_ms = self.predict_ttft_ms(prompt_len, active_sessions)

        if estimated_ttft_ms > target.target_ttft_ms and target.strict_enforcement:
            self.total_rejected += 1
            raise SLOViolationError(
                f"SLO admission rejected: predicted TTFT ({estimated_ttft_ms:.1f}ms) exceeds target SLO ({target.target_ttft_ms:.1f}ms).",
                estimated_ttft_ms=estimated_ttft_ms,
                target_ttft_ms=target.target_ttft_ms,
            )

        self.total_admitted += 1
        return True

    def record_session_completion(self, session: SchedSession, target: Optional[SLOTarget] = None) -> None:
        """Record actual TTFT and TBT compliance for a completed session."""
        target_ttft = target.target_ttft_ms if target else self.default_target_ttft_ms
        actual_ttft = session.metrics.ttft_ms

        if actual_ttft is not None:
            if actual_ttft <= target_ttft:
                self.slo_met_count += 1
            else:
                self.slo_violated_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics for SLO compliance and admission control."""
        evaluated = max(1, self.total_evaluated)
        completed = self.slo_met_count + self.slo_violated_count
        compliance_pct = (self.slo_met_count / completed * 100.0) if completed > 0 else 100.0

        return {
            "default_target_ttft_ms": self.default_target_ttft_ms,
            "default_target_tbt_ms": self.default_target_tbt_ms,
            "total_evaluated": self.total_evaluated,
            "total_admitted": self.total_admitted,
            "total_rejected": self.total_rejected,
            "admission_rate_pct": round((self.total_admitted / evaluated) * 100.0, 2),
            "slo_met_count": self.slo_met_count,
            "slo_violated_count": self.slo_violated_count,
            "slo_compliance_pct": round(compliance_pct, 2),
        }
