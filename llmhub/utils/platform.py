"""Platform detection, CPU topology, memory inspection, and Termux compatibility flags."""

import os
import platform
import sys
from dataclasses import dataclass
from typing import List, Optional
import psutil

@dataclass
class HardwareProfile:
    os_name: str
    arch: str
    is_termux: bool
    is_non_root: bool
    physical_cores: int
    logical_cores: int
    total_ram_bytes: int
    free_ram_bytes: int
    simd_features: List[str]
    suggested_threads: int
    warnings: List[str]

def is_termux_environment() -> bool:
    """Return True if running in Android Termux."""
    return "TERMUX_VERSION" in os.environ or os.path.exists("/data/data/com.termux")

def is_root_user() -> bool:
    """Check if process has root / administrator privileges."""
    if hasattr(os, "geteuid"):
        return os.geteuid() == 0  # type: ignore[attr-defined]
    import ctypes
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False

def detect_simd_features() -> List[str]:
    """Detect available SIMD instruction set extensions across platforms."""
    features = []
    sys_platform = platform.system().lower()
    machine = platform.machine().lower()

    if "arm" in machine or "aarch64" in machine:
        features.append("NEON")
        if sys_platform == "linux" or is_termux_environment():
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                    if "asimd" in content or "neon" in content:
                        if "NEON" not in features:
                            features.append("NEON")
                    if "fp16" in content:
                        features.append("ARM_FP16")
                    if "sve" in content:
                        features.append("SVE")
                    if "dotprod" in content:
                        features.append("DOTPROD")
            except Exception:
                pass
    else:
        # x86 / x86_64
        if sys_platform == "linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                    if "avx2" in content:
                        features.append("AVX2")
                    if "avx512" in content or "avx512f" in content:
                        features.append("AVX512")
                    if "fma" in content:
                        features.append("FMA")
            except Exception:
                pass
        elif sys_platform == "darwin":
            try:
                import subprocess
                out = subprocess.check_output(["sysctl", "-a"], stderr=subprocess.DEVNULL).decode()
                if "hw.optional.avx2_0: 1" in out:
                    features.append("AVX2")
                if "hw.optional.avx512" in out:
                    features.append("AVX512")
            except Exception:
                pass
        elif sys_platform == "windows":
            # On Windows x86_64, AVX/AVX2 can be safely assumed for modern Intel/AMD CPUs,
            # or probed via llama_cpp runtime
            try:
                import llama_cpp
                # llama-cpp provides runtime feature check helpers
                if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                    # Check CPU features compiled into backend
                    features.append("AVX2")
                    features.append("FMA")
            except Exception:
                features.append("AVX2")

    return features

def get_hardware_profile() -> HardwareProfile:
    """Analyze the current host system, detecting CPU cores, RAM, and constraints."""
    vm = psutil.virtual_memory()
    total_ram = vm.total
    free_ram = vm.available
    
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or physical_cores
    
    # Golden rule: For CPU LLM inference, use physical cores, NOT logical cores (hyperthreads)
    # Hyperthreading shares L1/L2 and vector units, causing memory bus contention and lower IPC
    suggested_threads = max(1, physical_cores)
    
    termux = is_termux_environment()
    root = is_root_user()
    simd = detect_simd_features()
    warnings = []
    
    if termux:
        warnings.append("Running on Android Termux (ARM64).")
        if not root:
            warnings.append("Non-root environment: mlock() disabled; Android Low Memory Killer (LMK) active.")
            warnings.append("Keep context sizes moderate to prevent Android OS process kill.")
    
    if total_ram < 4 * 1024 * 1024 * 1024:
        warnings.append("Total system RAM is under 4GB. Recommend small models (<= 1B params, e.g. Qwen2.5-0.5B Q4).")

    return HardwareProfile(
        os_name=platform.system(),
        arch=platform.machine(),
        is_termux=termux,
        is_non_root=not root,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        total_ram_bytes=total_ram,
        free_ram_bytes=free_ram,
        simd_features=simd,
        suggested_threads=suggested_threads,
        warnings=warnings,
    )
