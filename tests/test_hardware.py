from llmhub.hardware.detector import (
    HardwareDetector,
    CPUTopology,
    GPUDeviceInfo,
    HardwareProfile,
)

def test_hardware_detector_profile():
    profile = HardwareDetector.get_profile()
    assert profile.os_name is not None
    assert profile.cpu.physical_cores >= 1
    assert profile.cpu.logical_cores >= profile.cpu.physical_cores
    assert profile.total_ram_bytes > 0
    assert profile.free_ram_bytes > 0
    assert profile.suggested_threads >= 1
    assert profile.suggested_threads <= profile.cpu.physical_cores

def test_detect_simd():
    simd = HardwareDetector.detect_simd()
    assert isinstance(simd, list)
    # On x86_64, AVX2 or AVX should be detected
    if profile_cpu := HardwareDetector.detect_cpu_topology():
        if "arm" not in profile_cpu.arch.lower() and "aarch64" not in profile_cpu.arch.lower():
            assert "AVX" in simd or "AVX2" in simd

def test_detect_gpu():
    gpu = HardwareDetector.detect_gpu()
    assert isinstance(gpu, GPUDeviceInfo)
    assert gpu.backend in ("cuda", "metal", "vulkan", "cpu")

def test_hybrid_big_little_simulation():
    # Simulate an octa-core ARM64 CPU (4 P-cores + 4 E-cores)
    topo = CPUTopology(
        arch="aarch64",
        physical_cores=8,
        logical_cores=8,
        is_hybrid_big_little=True,
        performance_cores=4,
        efficiency_cores=4,
        simd_extensions=["NEON", "ARM_FP16"]
    )
    assert topo.is_hybrid_big_little is True
    assert topo.performance_cores == 4
    assert topo.efficiency_cores == 4
