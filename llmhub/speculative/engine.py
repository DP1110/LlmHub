"""Speculative Decoding Engine using Draft Model Generation and Target Parallel Verification.

Physics & Algorithm:
1. Small draft model (e.g. 512M) generates K candidate tokens quickly
   (memory-bandwidth reading weights is ~10x faster than target 7B).
2. Target model (e.g. 7B) verifies all K candidate tokens in a single parallel forward pass.
3. Accept matching tokens until the first divergence at index j < K.
4. Sample 1 correction token from the target model at index j.
5. Expected accepted tokens: E[tokens] = sum_{i=1}^K alpha^i + 1
6. Speedup factor: S = E[tokens] / (1 + K * (T_draft / T_target))
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple

from llmhub.server.backend import Backend, DecodeResult

@dataclass
class SpeculativeStepResult:
    """Result of a single speculative decoding verification turn."""
    draft_tokens: List[int]
    accepted_tokens: List[int]
    correction_token: Optional[int]
    emitted_tokens: List[int]
    emitted_text_deltas: List[str]
    acceptance_rate: float
    done: bool
    stop_reason: Optional[str] = None
    verification_duration_ms: float = 0.0
    estimated_speedup: float = 1.0

class SpeculativeEngine:
    """Orchestrates draft candidate generation and target parallel verification."""

    def __init__(
        self,
        target_model_tag: str,
        draft_model_tag: str,
        k_draft: int = 4,
        draft_to_target_cost_ratio: float = 0.15,  # Ratio of draft decode time to target decode time
    ):
        self.target_model_tag = target_model_tag
        self.draft_model_tag = draft_model_tag
        self.k_draft = max(1, min(8, int(k_draft)))
        self.draft_to_target_cost_ratio = draft_to_target_cost_ratio

        # Telemetry
        self.total_rounds: int = 0
        self.total_draft_tokens: int = 0
        self.total_accepted_tokens: int = 0
        self.total_correction_tokens: int = 0
        self.total_emitted_tokens: int = 0

    @property
    def acceptance_rate_alpha(self) -> float:
        """Global token acceptance rate alpha across all verification turns."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def cumulative_speedup(self) -> float:
        """Cumulative effective speedup factor over standard autoregressive decoding."""
        if self.total_rounds == 0:
            return 1.0
        avg_tokens_per_round = self.total_emitted_tokens / self.total_rounds
        cost_per_round = 1.0 + (self.k_draft * self.draft_to_target_cost_ratio)
        return max(1.0, avg_tokens_per_round / cost_per_round)

    def speculative_step(
        self,
        session_id: str,
        target_backend: Backend,
        draft_backend: Backend,
        k_draft: Optional[int] = None,
    ) -> SpeculativeStepResult:
        """Execute one speculative decoding turn:
        
        1. Generate K draft tokens using lightweight draft_backend.
        2. Verify candidate sequence against target_backend.
        3. Accept matching prefix and sample target correction token.
        """
        k = k_draft or self.k_draft
        t_start = time.perf_counter()

        # Step 1: Generate K candidate tokens with draft model
        draft_res = draft_backend.decode_step(session_id, n_tokens=k)
        candidates = list(draft_res.tokens)
        if not candidates:
            # Fallback: draft model didn't emit tokens, run standard target decode
            target_res = target_backend.decode_step(session_id, n_tokens=1)
            return SpeculativeStepResult(
                draft_tokens=[],
                accepted_tokens=[],
                correction_token=target_res.tokens[0] if target_res.tokens else None,
                emitted_tokens=list(target_res.tokens),
                emitted_text_deltas=list(target_res.text_deltas),
                acceptance_rate=0.0,
                done=target_res.done,
                stop_reason=target_res.stop_reason,
                verification_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                estimated_speedup=1.0,
            )

        # Step 2: Target model parallel verification
        # The target evaluates up to K candidate tokens + 1 potential forward prediction
        target_verify_res = target_backend.decode_step(session_id, n_tokens=len(candidates))

        accepted_tokens: List[int] = []
        accepted_deltas: List[str] = []
        correction_token: Optional[int] = None
        correction_delta: Optional[str] = None
        mismatch_found = False

        target_tokens = target_verify_res.tokens
        target_deltas = target_verify_res.text_deltas

        for i in range(len(candidates)):
            cand = candidates[i]
            target_tok = target_tokens[i] if i < len(target_tokens) else None
            target_del = target_deltas[i] if i < len(target_deltas) else f" {target_tok}"

            if not mismatch_found and cand == target_tok:
                accepted_tokens.append(cand)
                accepted_deltas.append(target_del)
            else:
                mismatch_found = True
                if correction_token is None and target_tok is not None:
                    correction_token = target_tok
                    correction_delta = target_del
                break

        # If all K tokens matched, sample 1 additional token from the target model
        if not mismatch_found and len(accepted_tokens) == len(candidates):
            extra_res = target_backend.decode_step(session_id, n_tokens=1)
            if extra_res.tokens:
                correction_token = extra_res.tokens[0]
                correction_delta = extra_res.text_deltas[0] if extra_res.text_deltas else f" {correction_token}"

        emitted_tokens = list(accepted_tokens)
        emitted_deltas = list(accepted_deltas)
        if correction_token is not None:
            emitted_tokens.append(correction_token)
            emitted_deltas.append(correction_delta or "")

        turn_alpha = len(accepted_tokens) / len(candidates) if candidates else 0.0
        turn_cost = 1.0 + (len(candidates) * self.draft_to_target_cost_ratio)
        turn_speedup = max(1.0, len(emitted_tokens) / turn_cost)

        # Update telemetry
        self.total_rounds += 1
        self.total_draft_tokens += len(candidates)
        self.total_accepted_tokens += len(accepted_tokens)
        if correction_token is not None:
            self.total_correction_tokens += 1
        self.total_emitted_tokens += len(emitted_tokens)

        # Synchronize target backend session state with emitted tokens
        target_sess = target_backend.get_session(session_id)
        if target_sess:
            excess_target = len(target_tokens) - len(emitted_tokens)
            if excess_target > 0:
                target_sess.generated_tokens = target_sess.generated_tokens[:-excess_target]
            elif excess_target < 0 and correction_token is not None:
                if not target_sess.generated_tokens or target_sess.generated_tokens[-1] != correction_token:
                    target_sess.generated_tokens.append(correction_token)

        # Synchronize draft backend session state with emitted tokens
        draft_sess = draft_backend.get_session(session_id)
        if draft_sess:
            n_cand = len(candidates)
            if n_cand > 0:
                draft_sess.generated_tokens = draft_sess.generated_tokens[:-n_cand] + emitted_tokens

        duration_ms = (time.perf_counter() - t_start) * 1000.0

        return SpeculativeStepResult(
            draft_tokens=candidates,
            accepted_tokens=accepted_tokens,
            correction_token=correction_token,
            emitted_tokens=emitted_tokens,
            emitted_text_deltas=emitted_deltas,
            acceptance_rate=round(turn_alpha, 3),
            done=target_verify_res.done,
            stop_reason=target_verify_res.stop_reason,
            verification_duration_ms=round(duration_ms, 2),
            estimated_speedup=round(turn_speedup, 2),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics and acceptance rate for speculative decoding."""
        return {
            "target_model": self.target_model_tag,
            "draft_model": self.draft_model_tag,
            "k_draft": self.k_draft,
            "total_rounds": self.total_rounds,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_emitted_tokens": self.total_emitted_tokens,
            "acceptance_rate_alpha": round(self.acceptance_rate_alpha, 4),
            "cumulative_speedup": round(self.cumulative_speedup, 2),
        }
