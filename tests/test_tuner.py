from llmhub.hardware.detector import (
    HardwareProfile,
    CPUTopology,
    GPUDeviceInfo,
)
from llmhub.hardware.tuner import AutoTuner

def test_select_threads_homogeneous():
    prof = HardwareProfile(
        os_name="Linux",
        arch="x86_64",
        cpu=CPUTopology(
            arch="x86_64",
            physical_cores=8,
            logical_cores=16,
            is_hybrid_big_little=False,
            performance_cores=8,
            efficiency_cores=0,
            simd_extensions=["AVX2"]
        ),
        total_ram_bytes=16 * 1024**3,
        free_ram_bytes=8 * 1024**3,
        gpu=GPUDeviceInfo(is_available=False),
        is_termux=False,
        is_non_root=False,
        suggested_threads=8
    )
    threads, rationale = AutoTuner.select_threads(prof)
    assert threads == 8
    assert "Selected 8 physical cores" in rationale

def test_select_threads_hybrid():
    prof = HardwareProfile(
        os_name="Linux",
        arch="aarch64",
        cpu=CPUTopology(
            arch="aarch64",
            physical_cores=8,
            logical_cores=8,
            is_hybrid_big_little=True,
            performance_cores=4,
            efficiency_cores=4,
            simd_extensions=["NEON"]
        ),
        total_ram_bytes=8 * 1024**3,
        free_ram_bytes=4 * 1024**3,
        gpu=GPUDeviceInfo(is_available=False),
        is_termux=True,
        is_non_root=True,
        suggested_threads=4
    )
    threads, rationale = AutoTuner.select_threads(prof)
    assert threads == 4
    assert "performance cluster" in rationale

def test_calculate_safe_context():
    # 0.5B model weights ~400 MB
    weights_bytes = 400 * 1024 * 1024
    free_ram = 2 * 1024 * 1024 * 1024  # 2 GB free

    rec_ctx, max_ctx, is_safe = AutoTuner.calculate_safe_context(
        weights_bytes=weights_bytes,
        n_layers=24,
        n_kv_heads=2,
        head_dim=64,
        ctx_max=32768,
        free_ram_bytes=free_ram
    )
    assert is_safe is True
    assert rec_ctx >= 2048
    assert max_ctx >= rec_ctx

def test_quant_suggestions_scale():
    suggestions = AutoTuner.evaluate_quant_suggestions(param_count_billions=7.0)
    assert len(suggestions) == 6
    quants = [s.quant for s in suggestions]
    assert "Q4_K_M" in quants
    assert "Q8_0" in quants
    assert "Q2_K" in quants

    # Q4_K_M for 7B should be around 3.9 GB
    q4 = next(s for s in suggestions if s.quant == "Q4_K_M")
    assert 3.5 <= q4.weights_size_gb <= 4.2

def test_select_threads_high_core_server_capped():
    # 64-core AMD EPYC server
    prof = HardwareProfile(
        os_name="Linux",
        arch="x86_64",
        cpu=CPUTopology(
            arch="x86_64",
            physical_cores=64,
            logical_cores=128,
            is_hybrid_big_little=False,
            performance_cores=64,
            efficiency_cores=0,
            simd_extensions=["AVX512"]
        ),
        total_ram_bytes=256 * 1024**3,
        free_ram_bytes=128 * 1024**3,
        gpu=GPUDeviceInfo(is_available=False),
        is_termux=False,
        is_non_root=False,
        suggested_threads=16
    )
    threads, rationale = AutoTuner.select_threads(prof)
    assert threads == 16
    assert "capped at 16" in rationale

def test_calculate_safe_context_insufficient_ram():
    # Model weights 4 GB, but only 2 GB free RAM
    weights_bytes = 4 * 1024 * 1024 * 1024
    free_ram = 2 * 1024 * 1024 * 1024

    rec_ctx, max_ctx, is_safe = AutoTuner.calculate_safe_context(
        weights_bytes=weights_bytes,
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        ctx_max=4096,
        free_ram_bytes=free_ram
    )
    assert is_safe is False
    assert rec_ctx == 512
