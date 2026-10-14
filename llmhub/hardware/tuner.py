"""Heuristic Auto-Tuning Engine for LLMHub.

Calculates optimal thread counts, safe dynamic context sizes based on RAM safety margins,
and evaluates quantization compatibility for models from 512M to 70B+ parameters.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from llmhub.hardware.detector import HardwareDetector, HardwareProfile
from llmhub.utils.math_calc import compute_kv_cache_bytes, compute_memory_breakdown, estimate_decode_throughput

@dataclass
class QuantSuggestion:
    quant: str
    bits_per_weight: float
    weights_size_gb: float
    total_ram_needed_gb: float  # with 2048 context
    fits_free_ram: bool
    fits_total_ram: bool
    status: str
    notes: str

@dataclass
class ModelTuningPlan:
    model_tag: str
    recommended_threads: int
    thread_rationale: str
    recommended_ctx: int
    max_safe_ctx: int
    memory_breakdown_mb: Dict[str, float]
    free_ram_after_load_mb: float
    is_safe: bool
    warnings: List[str]

class AutoTuner:
    """Computes optimal execution parameters for models on target hardware."""

    @classmethod
    def select_threads(cls, profile: Optional[HardwareProfile] = None) -> Tuple[int, str]:
        """Select CPU thread count, strictly avoiding hyperthreads and efficiency cores."""
        prof = profile or HardwareDetector.get_profile()
        cpu = prof.cpu

        if cpu.is_hybrid_big_little:
            raw_threads = cpu.performance_cores
            threads = min(16, max(1, raw_threads))
            rationale = (
                f"Selected {threads} threads matching performance cluster. "
                f"Excluding {cpu.efficiency_cores} efficiency cores to prevent synchronization barrier stalls."
            )
        else:
            raw_threads = cpu.physical_cores
            threads = min(16, max(1, raw_threads))
            if raw_threads > 16:
                rationale = (
                    f"Selected {threads} threads (capped at 16 from {raw_threads} physical cores to avoid "
                    f"cache line bouncing and memory bus synchronization stalls). "
                    f"Logical cores ({cpu.logical_cores}) excluded."
                )
            else:
                rationale = (
                    f"Selected {threads} physical cores. "
                    f"Logical cores ({cpu.logical_cores}) excluded to eliminate SMT cache and pipeline contention."
                )

        return threads, rationale

    @classmethod
    def calculate_safe_context(
        cls,
        weights_bytes: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        ctx_max: int,
        free_ram_bytes: int,
        safety_margin: float = 0.85,
        base_overhead_bytes: int = 64 * 1024 * 1024
    ) -> Tuple[int, int, bool]:
        """Determine safe context window length fitting within available free RAM.
        
        Returns: (recommended_ctx, max_safe_ctx, is_safe)
        """
        budget_bytes = int(free_ram_bytes * safety_margin)
        available_for_kv = budget_bytes - weights_bytes - base_overhead_bytes

        if available_for_kv <= 0:
            # Memory pressure: cannot even fit 512 context safely within safety margin
            return 512, 512, False

        bytes_per_token = compute_kv_cache_bytes(n_layers, n_kv_heads, head_dim, 1, bytes_per_element=2)
        if bytes_per_token <= 0:
            bytes_per_token = 1

        raw_max_tokens = available_for_kv // bytes_per_token
        max_safe_ctx = min(ctx_max, raw_max_tokens)

        # If budget cannot even safely fit standard 512 context, mark unsafe
        is_safe = raw_max_tokens >= 512
        if not is_safe:
            return 512, max(512, max_safe_ctx), False

        # Standard power-of-two candidate context lengths
        candidates = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        safe_candidates = [c for c in candidates if c <= max_safe_ctx]

        if safe_candidates:
            # Recommend up to 4096 or highest safe candidate
            recommended = min(4096, safe_candidates[-1])
        else:
            recommended = 512

        return recommended, max_safe_ctx, True

    @classmethod
    def generate_tuning_plan(
        cls,
        model_tag: str,
        weights_bytes: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        ctx_max: int,
        profile: Optional[HardwareProfile] = None,
        safety_margin: float = 0.85
    ) -> ModelTuningPlan:
        """Create a holistic tuning plan for a specific model on the current hardware."""
        prof = profile or HardwareDetector.get_profile()
        threads, thread_rationale = cls.select_threads(prof)

        rec_ctx, max_ctx, is_safe = cls.calculate_safe_context(
            weights_bytes=weights_bytes,
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            ctx_max=ctx_max,
            free_ram_bytes=prof.free_ram_bytes,
            safety_margin=safety_margin
        )

        mem = compute_memory_breakdown(
            weights_bytes=weights_bytes,
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            n_ctx=rec_ctx
        )

        free_after = (prof.free_ram_bytes - mem.total_bytes) / (1024 * 1024)
        warnings = list(prof.warnings)

        if not is_safe or free_after < 200:
            if free_after < 0:
                warnings.append(
                    f"Severe memory constraint: Model exceeds available free RAM by {abs(free_after):.0f} MB "
                    f"(requires OS paging/swap; extreme risk of OOM or Android LMK termination)."
                )
            else:
                warnings.append(
                    f"Severe memory constraint: Available free RAM ({prof.free_ram_bytes / (1024*1024):.0f} MB) "
                    f"leaves only {free_after:.0f} MB after loading model."
                )

        return ModelTuningPlan(
            model_tag=model_tag,
            recommended_threads=threads,
            thread_rationale=thread_rationale,
            recommended_ctx=rec_ctx,
            max_safe_ctx=max_ctx,
            memory_breakdown_mb={
                "weights_mb": mem.weights_mb,
                "kv_cache_mb": mem.kv_cache_mb,
                "overhead_mb": mem.overhead_mb,
                "total_mb": mem.total_mb,
            },
            free_ram_after_load_mb=free_after,
            is_safe=is_safe and free_after >= 200,
            warnings=warnings
        )

    @classmethod
    def evaluate_quant_suggestions(
        cls,
        param_count_billions: float,
        profile: Optional[HardwareProfile] = None
    ) -> List[QuantSuggestion]:
        """Evaluate quantization formats for a given parameter scale (e.g. 0.5, 1.5, 7, 8, 14, 32, 70)."""
        prof = profile or HardwareDetector.get_profile()
        free_ram_gb = prof.free_ram_bytes / (1024**3)
        total_ram_gb = prof.total_ram_bytes / (1024**3)

        # Standard quantization formats: (name, bits_per_param, notes)
        formats = [
            ("Q8_0", 8.5, "Near unquantized precision (FP16-like), high RAM footprint"),
            ("Q6_K", 6.6, "Extremely high fidelity, negligible perplexity loss"),
            ("Q5_K_M", 5.5, "Superior perplexity-to-size balance"),
            ("Q4_K_M", 4.5, "Standard gold standard balance of speed, RAM, and intelligence"),
            ("Q3_K_M", 3.4, "Low-memory budget option, slight perplexity increase"),
            ("Q2_K", 2.6, "Extreme compression for heavily constrained devices"),
        ]

        suggestions = []
        # Base KV cache at 2048 ctx + runtime overhead ~0.3 GB
        fixed_overhead_gb = 0.35

        for name, bits, notes in formats:
            # Size in GB = (param_count * 10^9 * (bits / 8)) / 10^9
            weights_gb = param_count_billions * (bits / 8.0)
            total_needed_gb = weights_gb + fixed_overhead_gb

            fits_free = total_needed_gb <= (free_ram_gb * 0.85)
            fits_total = total_needed_gb <= (total_ram_gb * 0.90)

            if fits_free:
                status = "Fits Comfortably"
            elif fits_total:
                status = "Fits (Requires Paging/Closing Apps)"
            else:
                status = "Exceeds Physical RAM"

            suggestions.append(QuantSuggestion(
                quant=name,
                bits_per_weight=bits,
                weights_size_gb=weights_gb,
                total_ram_needed_gb=total_needed_gb,
                fits_free_ram=fits_free,
                fits_total_ram=fits_total,
                status=status,
                notes=notes
            ))

        return suggestions
