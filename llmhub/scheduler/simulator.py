"""Deterministic Scheduler Simulator to prove fairness, starvation-freedom, and compare threading modes."""

from dataclasses import dataclass
import random
import time
from typing import Any, Dict, List, Optional

from llmhub.scheduler.policy import DeficitRoundRobinPolicy, ThreadingMode
from llmhub.scheduler.session import SchedSession, SessionPhase

@dataclass
class SimSessionSpec:
    session_id: str
    model_tag: str
    prompt_tokens: int
    completion_tokens: int
    weight: float = 1.0
    priority: int = 5
    quantum_tokens: int = 4
    chunk_size: int = 256

@dataclass
class SimResult:
    total_time_ms: float
    total_quanta: int
    total_tokens: int
    total_context_switches: int
    jains_fairness_index: float
    sessions_metrics: Dict[str, Dict[str, float]]
    threading_mode: str
    num_models: int

class DeterministicSchedulerSimulator:
    """Simulates token-level round-robin scheduler deterministically with synthetic timing physics."""

    def __init__(
        self,
        threading_mode: ThreadingMode = ThreadingMode.TIME_SLICE,
        physical_cores: int = 4,
        base_decode_ms_per_tok: float = 20.0,  # ~50 tok/s on full cores
        context_switch_penalty_ms: float = 0.5, # Cache reload penalty when switching models
    ):
        self.threading_mode = threading_mode
        self.physical_cores = physical_cores
        self.base_decode_ms_per_tok = base_decode_ms_per_tok
        self.context_switch_penalty_ms = context_switch_penalty_ms
        self.policy = DeficitRoundRobinPolicy(default_quantum=4)

    def run_simulation(self, session_specs: List[SimSessionSpec]) -> SimResult:
        """Run a fully deterministic simulation of sessions until all complete."""
        # Setup simulated sessions
        sessions: List[SchedSession] = []
        target_tokens: Dict[str, int] = {}
        tokens_remaining_prefill: Dict[str, int] = {}
        active_models: set = set()

        for spec in session_specs:
            s = SchedSession(
                session_id=spec.session_id,
                model_tag=spec.model_tag,
                prompt="",
                max_tokens=spec.completion_tokens,
                priority=spec.priority,
                weight=spec.weight,
                quantum_tokens=spec.quantum_tokens,
                chunk_size=spec.chunk_size,
            )
            s.metrics.enqueue_time = 0.0
            sessions.append(s)
            target_tokens[s.session_id] = spec.completion_tokens
            tokens_remaining_prefill[s.session_id] = spec.prompt_tokens
            active_models.add(spec.model_tag)

        num_models = len(active_models)
        current_time_ms = 0.0
        last_model_tag: Optional[str] = None
        total_context_switches = 0
        total_quanta = 0
        total_tokens = 0

        # Run discrete scheduling steps
        while any(not s.is_done for s in sessions):
            # Select next runnable session via DRR
            runnable = [s for s in sessions if not s.is_done]
            if not runnable:
                break

            session = self.policy.select_next_session(runnable)
            if not session:
                break

            # Context switch penalty check
            if last_model_tag is not None and last_model_tag != session.model_tag:
                total_context_switches += 1
                session.metrics.context_switches += 1
                if self.threading_mode == ThreadingMode.TIME_SLICE:
                    current_time_ms += self.context_switch_penalty_ms
            last_model_tag = session.model_tag

            # Handle Prefill Chunk
            if tokens_remaining_prefill[session.session_id] > 0:
                chunk = min(session.chunk_size, tokens_remaining_prefill[session.session_id])
                tokens_remaining_prefill[session.session_id] -= chunk
                session.metrics.prefill_tokens += chunk

                # Prefill is compute-bound, ~0.5ms per token on CPU
                prefill_ms = chunk * 0.5
                current_time_ms += prefill_ms
                if tokens_remaining_prefill[session.session_id] == 0:
                    session.phase = SessionPhase.DECODING

                # Rotate session to preserve round-robin interleaving
                if session in sessions:
                    sessions.remove(session)
                    sessions.append(session)
                continue

            # Handle Decode Quantum
            session.phase = SessionPhase.DECODING
            remaining_gen = target_tokens[session.session_id] - session.metrics.generated_tokens
            tokens_to_gen = min(session.quantum_tokens, remaining_gen)

            # Threading mode physics:
            # (a) Time-slice: gets all cores -> baseline ms/tok
            # (b) Partition: cores split among models -> ms/tok scales with allocated cores
            # (c) Hybrid: dynamically scales
            if self.threading_mode == ThreadingMode.TIME_SLICE:
                decode_ms_per_tok = self.base_decode_ms_per_tok
            elif self.threading_mode == ThreadingMode.PARTITION:
                threads_allocated = max(1, self.physical_cores // max(1, num_models))
                scaling_factor = self.physical_cores / threads_allocated
                decode_ms_per_tok = self.base_decode_ms_per_tok * (1.0 + 0.3 * (scaling_factor - 1.0))
            else:  # HYBRID
                if num_models == 1:
                    decode_ms_per_tok = self.base_decode_ms_per_tok
                else:
                    threads_allocated = max(1, self.physical_cores // num_models)
                    decode_ms_per_tok = self.base_decode_ms_per_tok * (1.0 + 0.25 * (self.physical_cores / threads_allocated - 1.0))

            quantum_duration_ms = tokens_to_gen * decode_ms_per_tok
            current_time_ms += quantum_duration_ms

            if session.metrics.first_token_time is None:
                session.metrics.first_token_time = current_time_ms / 1000.0

            session.metrics.generated_tokens += tokens_to_gen
            session.metrics.quanta_executed += 1
            session.deficit -= tokens_to_gen
            total_quanta += 1
            total_tokens += tokens_to_gen

            if session.metrics.generated_tokens >= target_tokens[session.session_id]:
                session.phase = SessionPhase.COMPLETED
                session.metrics.finish_time = current_time_ms / 1000.0
            else:
                # Rotate still-active session so other sessions get turns
                if session in sessions:
                    sessions.remove(session)
                    sessions.append(session)

        # Compute fairness and metrics
        metrics_dict: Dict[str, Dict[str, float]] = {}
        throughputs: List[float] = []
        weights: List[float] = []

        for s in sessions:
            gen = s.metrics.generated_tokens
            dur_s = (s.metrics.finish_time or (current_time_ms / 1000.0)) - (s.metrics.first_token_time or 0.0)
            tok_s = gen / dur_s if dur_s > 0 else 0.0
            throughputs.append(tok_s)
            weights.append(s.weight)

            metrics_dict[s.session_id] = {
                "generated_tokens": gen,
                "ttft_ms": (s.metrics.first_token_time or 0.0) * 1000.0,
                "total_duration_ms": (s.metrics.finish_time or (current_time_ms / 1000.0)) * 1000.0,
                "tok_s": round(tok_s, 2),
                "quanta": s.metrics.quanta_executed,
                "context_switches": s.metrics.context_switches,
            }

        jains_index = self.policy.compute_jains_fairness_index(throughputs, weights)

        return SimResult(
            total_time_ms=round(current_time_ms, 2),
            total_quanta=total_quanta,
            total_tokens=total_tokens,
            total_context_switches=total_context_switches,
            jains_fairness_index=round(jains_index, 4),
            sessions_metrics=metrics_dict,
            threading_mode=self.threading_mode.value,
            num_models=num_models,
        )
