from llmhub.utils.math_calc import (
    compute_kv_cache_bytes,
    compute_memory_breakdown,
    estimate_decode_throughput,
)

def test_compute_kv_cache_bytes_zero():
    assert compute_kv_cache_bytes(0, 8, 128, 2048) == 0
    assert compute_kv_cache_bytes(32, 0, 128, 2048) == 0

def test_compute_kv_cache_bytes_qwen_05b():
    # Qwen2.5-0.5B: 24 layers, 2 KV heads, 64 head_dim, 2048 context, 2 bytes (fp16)
    # Formula: 2 * 24 * 2 * 64 * 2048 * 2 = 25,165,824 bytes = 24.0 MB
    bytes_kv = compute_kv_cache_bytes(
        n_layers=24,
        n_kv_heads=2,
        head_dim=64,
        n_ctx=2048,
        bytes_per_element=2
    )
    assert bytes_kv == 25165824
    assert bytes_kv / (1024 * 1024) == 24.0

def test_compute_kv_cache_bytes_llama_8b():
    # Llama-3.1-8B: 32 layers, 8 KV heads, 128 head_dim, 8192 context, 2 bytes
    # Formula: 2 * 32 * 8 * 128 * 8192 * 2 = 1,073,741,824 bytes = 1.0 GB
    bytes_kv = compute_kv_cache_bytes(
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        n_ctx=8192,
        bytes_per_element=2
    )
    assert bytes_kv == 1073741824

def test_compute_memory_breakdown():
    weights = 400 * 1024 * 1024  # 400 MB
    breakdown = compute_memory_breakdown(
        weights_bytes=weights,
        n_layers=24,
        n_kv_heads=2,
        head_dim=64,
        n_ctx=2048
    )
    assert breakdown.weights_mb == 400.0
    assert breakdown.kv_cache_mb == 24.0
    assert breakdown.total_mb > 424.0

def test_estimate_decode_throughput_small_model():
    weights = 380 * 1024 * 1024  # ~0.37 GB
    est = estimate_decode_throughput(weights, estimated_bandwidth_gb_s=40.0)
    assert est.theoretical_max_tok_s > 100.0
    assert est.bottleneck_type == "Compute/Instruction-Latency Bound"

def test_estimate_decode_throughput_large_model():
    weights = int(4.9 * 1024 * 1024 * 1024)  # 4.9 GB (8B Q4)
    est = estimate_decode_throughput(weights, estimated_bandwidth_gb_s=40.0)
    assert 7.0 < est.theoretical_max_tok_s < 9.0
    assert est.bottleneck_type == "Memory-Bandwidth Bound"
