"""Advanced hardware detection, CPU topology, SIMD probing, and GPU acceleration discovery.

Supports Windows, Linux, macOS, and Android non-root Termux (ARM64).
"""

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil

@dataclass
class GPUDeviceInfo:
    is_available: bool
    name: str = "None"
    backend: str = "cpu"  # cuda | metal | vulkan | cpu
    vram_bytes: int = 0
    device_count: int = 0

@dataclass
class CPUTopology:
    arch: str
    physical_cores: int
    logical_cores: int
    is_hybrid_big_little: bool
    performance_cores: int
    efficiency_cores: int
    simd_extensions: List[str]

@dataclass
class HardwareProfile:
    os_name: str
    arch: str
    cpu: CPUTopology
    total_ram_bytes: int
    free_ram_bytes: int
    gpu: GPUDeviceInfo
    is_termux: bool
    is_non_root: bool
    suggested_threads: int
    warnings: List[str] = field(default_factory=list)

class HardwareDetector:
    """Probes CPU topology, SIMD capabilities, memory, and accelerator devices."""

    @staticmethod
    def is_termux() -> bool:
        return "TERMUX_VERSION" in os.environ or os.path.exists("/data/data/com.termux")

    @staticmethod
    def is_root() -> bool:
        if hasattr(os, "geteuid"):
            return os.geteuid() == 0  # type: ignore[attr-defined]
        import ctypes
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False

    @classmethod
    def detect_simd(cls) -> List[str]:
        """Detect CPU vector extensions."""
        simd = []
        machine = platform.machine().lower()
        sys_os = platform.system().lower()

        if "arm" in machine or "aarch64" in machine:
            simd.append("NEON")
            if sys_os == "linux" or cls.is_termux():
                try:
                    with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().lower()
                        if "asimd" in content and "NEON" not in simd:
                            simd.append("NEON")
                        if "fp16" in content:
                            simd.append("ARM_FP16")
                        if "dotprod" in content:
                            simd.append("DOTPROD")
                        if "sve2" in content:
                            simd.append("SVE2")
                        elif "sve" in content:
                            simd.append("SVE")
                except Exception:
                    pass
        else:
            # x86 / x86_64
            if sys_os == "linux":
                try:
                    with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().lower()
                        for flag, name in [
                            ("avx512f", "AVX512"),
                            ("avx2", "AVX2"),
                            ("avx", "AVX"),
                            ("fma", "FMA"),
                            ("f16c", "F16C")
                        ]:
                            if flag in content and name not in simd:
                                simd.append(name)
                except Exception:
                    pass
            elif sys_os == "darwin":
                for key, name in [
                    ("hw.optional.avx512f", "AVX512"),
                    ("hw.optional.avx2_0", "AVX2"),
                    ("hw.optional.fma", "FMA")
                ]:
                    try:
                        val = subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL).decode().strip()
                        if val == "1":
                            simd.append(name)
                    except Exception:
                        pass
            elif sys_os == "windows":
                # Modern Windows x86_64 CPUs support AVX2 + FMA
                simd.extend(["AVX", "AVX2", "FMA"])

        return simd

    @classmethod
    def detect_cpu_topology(cls) -> CPUTopology:
        """Detect physical vs logical cores and Big.LITTLE clusters."""
        arch = platform.machine()
        logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        
        # In non-root Android Termux and sandboxes, psutil logical=False returns None
        physical = psutil.cpu_count(logical=False)
        if physical is None:
            if "arm" in arch.lower() or "aarch64" in arch.lower():
                physical = logical  # Mobile ARM chips generally lack SMT/hyperthreading
            else:
                physical = max(1, logical // 2)

        is_hybrid = False
        p_cores = physical
        e_cores = 0

        # Detect Big.LITTLE on Linux / Android Termux
        if cls.is_termux() or platform.system().lower() == "linux":
            cpu_caps_dir = Path("/sys/devices/system/cpu")
            if cpu_caps_dir.exists():
                capacities = []
                for p in cpu_caps_dir.glob("cpu[0-9]*/cpu_capacity"):
                    try:
                        capacities.append(int(p.read_text().strip()))
                    except Exception:
                        pass
                if capacities:
                    # Reconcile core counts with discovered capacities
                    physical = max(physical, len(capacities))
                    logical = max(logical, len(capacities))
                    if len(set(capacities)) > 1:
                        is_hybrid = True
                        max_cap = max(capacities)
                        p_cores = sum(1 for c in capacities if c >= max_cap * 0.8)
                        e_cores = len(capacities) - p_cores
        
        # Fallback heuristic for ARM64 octa-core chips
        if not is_hybrid and ("arm" in arch.lower() or "aarch64" in arch.lower()) and physical >= 8:
            is_hybrid = True
            p_cores = 4
            e_cores = physical - 4

        # Guarantee consistency: p_cores cannot exceed physical cores
        p_cores = min(p_cores, physical)

        simd = cls.detect_simd()
        return CPUTopology(
            arch=arch,
            physical_cores=physical,
            logical_cores=logical,
            is_hybrid_big_little=is_hybrid,
            performance_cores=p_cores,
            efficiency_cores=e_cores,
            simd_extensions=simd
        )

    @classmethod
    def detect_gpu(cls) -> GPUDeviceInfo:
        """Probe for NVIDIA CUDA, Apple Metal, or Vulkan acceleration."""
        # 1. Check Apple Metal on macOS (native on Apple Silicon, zero PyTorch dependency)
        if platform.system().lower() == "darwin":
            machine = platform.machine().lower()
            if "arm" in machine or "aarch64" in machine:
                vm = psutil.virtual_memory()
                return GPUDeviceInfo(
                    is_available=True,
                    name="Apple Silicon Unified Memory (Metal)",
                    backend="metal",
                    vram_bytes=int(vm.total * 0.75),  # Unified memory pool
                    device_count=1
                )

        # 2. Check NVIDIA via nvidia-smi
        try:
            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=3
            ).decode().strip()
            if smi_out:
                lines = smi_out.splitlines()
                first = lines[0].split(",")
                gpu_name = first[0].strip()
                vram_mb = int(first[1].strip())
                return GPUDeviceInfo(
                    is_available=True,
                    name=gpu_name,
                    backend="cuda",
                    vram_bytes=vram_mb * 1024 * 1024,
                    device_count=len(lines)
                )
        except Exception:
            pass

        # 3. Check llama_cpp compiled GPU offload support
        try:
            import llama_cpp
            if hasattr(llama_cpp, "llama_supports_gpu_offload") and llama_cpp.llama_supports_gpu_offload():
                return GPUDeviceInfo(
                    is_available=True,
                    name="Generic GPU Accelerator",
                    backend="cuda",
                    vram_bytes=0,
                    device_count=1
                )
        except Exception:
            pass

        return GPUDeviceInfo(is_available=False, name="None", backend="cpu", vram_bytes=0, device_count=0)

    @classmethod
    def get_profile(cls) -> HardwareProfile:
        """Build complete hardware profile with auto-tuned thread suggestions and warnings."""
        vm = psutil.virtual_memory()
        cpu = cls.detect_cpu_topology()
        gpu = cls.detect_gpu()
        termux = cls.is_termux()
        root = cls.is_root()
        warnings = []

        # Thread selection heuristic:
        # Strictly physical cores. On Big.LITTLE, target the performance cluster size.
        if cpu.is_hybrid_big_little:
            suggested_threads = max(1, cpu.performance_cores)
            warnings.append(
                f"Heterogeneous CPU architecture (Big.LITTLE) detected: "
                f"Auto-selected {suggested_threads} performance cores (excluding {cpu.efficiency_cores} efficiency cores)."
            )
        else:
            suggested_threads = max(1, cpu.physical_cores)

        if termux:
            warnings.append("Environment: Android Termux (ARM64).")
            if not root:
                warnings.append("Non-root permissions: mlock() disabled, Android Low Memory Killer (LMK) active.")
                warnings.append("Keep context lengths moderate to avoid Android OS killing the process.")

        if vm.total < 4 * 1024 * 1024 * 1024:
            warnings.append(f"Low system RAM ({vm.total / (1024**3):.1f} GB). Recommend models <= 1B parameters.")

        return HardwareProfile(
            os_name=platform.system(),
            arch=cpu.arch,
            cpu=cpu,
            total_ram_bytes=vm.total,
            free_ram_bytes=vm.available,
            gpu=gpu,
            is_termux=termux,
            is_non_root=not root,
            suggested_threads=suggested_threads,
            warnings=warnings
        )
