"""Mathematical physics calculation engine for LLM memory footprints and bandwidth limits.

Derivations:
1. Weights Memory = file_size_bytes (mmap mapped directly into virtual address space)
2. KV Cache Memory = 2 * n_layers * n_kv_heads * head_dim * n_ctx * bytes_per_element
   (2x factor accounts for Key tensor and Value tensor)
3. Compute Scratch Memory = scratch working buffers (~50MB to 500MB depending on context)
4. Decode Token Generation Speed:
   Autoregressive generation (batch=1) requires streaming every single model parameter
   from system RAM through CPU caches per token:
   Theoretical_Max_Tok_Per_Sec <= System_Memory_Bandwidth_GBs / Model_Weights_Size_GB
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MemoryBreakdown:
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    total_bytes: int
    weights_mb: float
    kv_cache_mb: float
    overhead_mb: float
    total_mb: float
    total_gb: float

@dataclass
class BandwidthEstimate:
    memory_bandwidth_gb_s: float
    weights_gb: float
    theoretical_max_tok_s: float
    bottleneck_type: str
    explanation: str

def compute_kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    n_ctx: int,
    bytes_per_element: int = 2  # default FP16 KV cache
) -> int:
    """Compute the exact memory bytes needed for the KV cache.
    
    Formula: 2 (Key + Value) * n_layers * n_kv_heads * head_dim * n_ctx * bytes_per_element
    """
    if n_layers <= 0 or n_kv_heads <= 0 or head_dim <= 0 or n_ctx <= 0:
        return 0
    return 2 * n_layers * n_kv_heads * head_dim * n_ctx * bytes_per_element

def compute_memory_breakdown(
    weights_bytes: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    n_ctx: int,
    bytes_per_element: int = 2,
    base_overhead_bytes: int = 64 * 1024 * 1024  # 64 MB base runtime overhead
) -> MemoryBreakdown:
    """Compute complete memory allocation profile for weights, KV cache, and runtime overhead."""
    kv_bytes = compute_kv_cache_bytes(n_layers, n_kv_heads, head_dim, n_ctx, bytes_per_element)
    
    # Compute scratch buffer scales with context and embedding dimension
    # Approximate scratch tensor space
    context_overhead = n_ctx * head_dim * n_kv_heads * 4  # intermediate activations
    overhead_bytes = base_overhead_bytes + context_overhead
    
    total_bytes = weights_bytes + kv_bytes + overhead_bytes

    return MemoryBreakdown(
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_bytes,
        overhead_bytes=overhead_bytes,
        total_bytes=total_bytes,
        weights_mb=weights_bytes / (1024 * 1024),
        kv_cache_mb=kv_bytes / (1024 * 1024),
        overhead_mb=overhead_bytes / (1024 * 1024),
        total_mb=total_bytes / (1024 * 1024),
        total_gb=total_bytes / (1024 * 1024 * 1024),
    )

def estimate_decode_throughput(
    weights_bytes: int,
    estimated_bandwidth_gb_s: float = 40.0  # Dual-channel DDR4-3200 default
) -> BandwidthEstimate:
    """Calculate theoretical upper limit of single-session token generation speed.
    
    Autoregressive decoding has an operational intensity of ~1 FLOP/byte.
    Every token requires reading the entire model weights through memory channels once.
    """
    weights_gb = weights_bytes / (1024 * 1024 * 1024)
    if weights_gb <= 0:
        return BandwidthEstimate(
            memory_bandwidth_gb_s=estimated_bandwidth_gb_s,
            weights_gb=0.0,
            theoretical_max_tok_s=0.0,
            bottleneck_type="Unknown",
            explanation="Invalid model weight size"
        )
    
    max_tok_s = estimated_bandwidth_gb_s / weights_gb
    
    if weights_gb < 0.8:
        bottleneck = "Compute/Instruction-Latency Bound"
        explanation = (
            f"Weights ({weights_gb:.2f} GB) fit largely in CPU caches or stream very rapidly. "
            f"Actual speed is limited by single-thread instruction dispatch, vector unit latency, "
            f"and thread sync, typically maxing out at 60-120 tok/s on CPU."
        )
    else:
        bottleneck = "Memory-Bandwidth Bound"
        explanation = (
            f"Autoregressive token generation requires reading the entire model ({weights_gb:.2f} GB) "
            f"from RAM per token. At {estimated_bandwidth_gb_s:.1f} GB/s bandwidth, "
            f"the theoretical ceiling is {max_tok_s:.1f} tok/s. Adding threads beyond memory channel "
            f"saturation cannot exceed this physical limit."
        )

    return BandwidthEstimate(
        memory_bandwidth_gb_s=estimated_bandwidth_gb_s,
        weights_gb=weights_gb,
        theoretical_max_tok_s=max_tok_s,
        bottleneck_type=bottleneck,
        explanation=explanation,
    )
