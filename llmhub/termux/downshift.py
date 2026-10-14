"""Context and Quantization Downshifting Under Memory Pressure.

When system memory becomes constrained on mobile or low-spec devices, attempting
to allocate large KV caches causes immediate OS process termination (Android LMK
or Linux OOM-killer).

This module provides dynamic downshifting algorithms:
1. Context Downshifting: Automatically reduces requested context lengths
   (e.g., 4096 -> 2048 -> 1024 -> 512) to fit within free physical RAM.
2. Quantization Downshifting: Evaluates and recommends lighter quantization formats
   (e.g., Q4_K_M -> Q3_K_M -> Q2_K) or smaller parameter classes when models exceed RAM.
3. Cache Pruning: Signals prefix caches to shed non-essential LRU entries under pressure.
"""

import logging
from typing import List, Optional, Tuple

from llmhub.termux.memory_pressure import MemoryPressureLevel
from llmhub.utils.math_calc import compute_kv_cache_bytes, compute_memory_breakdown

logger = logging.getLogger("llmhub.termux.downshift")

# Quantization order from highest fidelity to highest compression
QUANT_FIDELITY_LADDER: List[Tuple[str, float]] = [
    ("Q8_0", 8.5),
    ("Q6_K", 6.6),
    ("Q5_K_M", 5.5),
    ("Q4_K_M", 4.5),
    ("Q3_K_M", 3.4),
    ("Q2_K", 2.6),
]

class DownshiftEngine:
    """Calculates downshifted context windows and alternative quantization formats."""

    @staticmethod
    def adjust_context_for_pressure(
        desired_ctx: int,
        pressure_level: MemoryPressureLevel,
        min_ctx: int = 512,
        low_memory_mode: bool = False,
    ) -> int:
        """Clamp context length based on real-time memory pressure level without upshifting."""
        effective_min = min(desired_ctx, min_ctx)
        if pressure_level == MemoryPressureLevel.CRITICAL:
            adjusted = effective_min
            if adjusted < desired_ctx:
                logger.warning(
                    "CRITICAL memory pressure: downshifting context from %d to minimum %d.",
                    desired_ctx,
                    adjusted,
                )
            return adjusted
        elif pressure_level == MemoryPressureLevel.MODERATE or low_memory_mode:
            adjusted = min(desired_ctx, max(effective_min, min(desired_ctx, 1024)))
            if adjusted < desired_ctx:
                logger.info(
                    "MODERATE memory pressure (or low-mem mode): downshifting context from %d to %d.",
                    desired_ctx,
                    adjusted,
                )
            return adjusted
        return desired_ctx

    @staticmethod
    def find_fitting_context(
        weights_bytes: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        desired_ctx: int,
        available_ram_bytes: int,
        safety_margin: float = 0.85,
        min_ctx: int = 512,
        base_overhead_bytes: int = 64 * 1024 * 1024,
    ) -> Tuple[Optional[int], str]:
        """Find the largest standard context length that fits within available RAM.
        
        Returns: (fitting_ctx_or_None, rationale)
        """
        budget_bytes = int(available_ram_bytes * safety_margin)
        available_for_kv = budget_bytes - weights_bytes - base_overhead_bytes

        if available_for_kv <= 0:
            return None, (
                f"Model weights ({weights_bytes / (1024**2):.1f} MB) plus overhead "
                f"exceed usable RAM budget ({budget_bytes / (1024**2):.1f} MB)."
            )

        bytes_per_token = compute_kv_cache_bytes(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            n_ctx=1,
            bytes_per_element=2,
        )
        if bytes_per_token <= 0:
            bytes_per_token = 1

        max_tokens = available_for_kv // bytes_per_token
        effective_min = min(desired_ctx, min_ctx)
        if max_tokens < effective_min:
            return None, (
                f"Available RAM can only fit {max_tokens} tokens KV cache, below min_ctx {effective_min}."
            )

        # Standard power-of-two candidate context lengths, strictly descending
        candidate_pool = {desired_ctx, effective_min, 4096, 2048, 1024, 768, 512}
        candidates = sorted([c for c in candidate_pool if effective_min <= c <= desired_ctx], reverse=True)

        # Filter candidates <= max_tokens
        viable = [c for c in candidates if c <= max_tokens]

        if not viable:
            return effective_min, f"Clamped to minimum context {effective_min}."

        best = viable[0]
        if best < desired_ctx:
            rationale = (
                f"Downshifted context from {desired_ctx} to {best} tokens to fit within "
                f"{available_ram_bytes / (1024**2):.0f} MB available RAM."
            )
        else:
            rationale = f"Requested context {desired_ctx} fits comfortably."

        return best, rationale

    @staticmethod
    def suggest_downshifted_quant(
        param_count_billions: float,
        available_ram_bytes: int,
        safety_margin: float = 0.85,
        context_size: int = 1024,
    ) -> Optional[Tuple[str, float, str]]:
        """Suggest the highest fidelity quantization format that fits within available RAM.
        
        Returns: (quant_name, bits_per_weight, description) or None if no quant fits.
        """
        usable_gb = (available_ram_bytes * safety_margin) / (1024**3)
        # Approximate KV cache + overhead in GB for the given context
        fixed_overhead_gb = 0.20 + (context_size / 2048.0) * 0.15

        for quant, bits in QUANT_FIDELITY_LADDER:
            weights_gb = param_count_billions * (bits / 8.0)
            total_needed_gb = weights_gb + fixed_overhead_gb
            if total_needed_gb <= usable_gb:
                desc = (
                    f"Selected {quant} ({bits:.1f} bpw). Estimated working set: {total_needed_gb:.2f} GB "
                    f"(fits within {usable_gb:.2f} GB usable RAM)."
                )
                return quant, bits, desc

        return None
