"""Android Low Memory Killer (LMK) Awareness and Memory Pressure Monitor.

On Android, when available physical RAM drops below kernel watermarks, the Android
Low Memory Killer daemon (`lmkd`) selects and kills processes with high `oom_score_adj`.
Without root, an app cannot change its own oom_score_adj to negative values, but
LLMHub can monitor:
1. Current process `/proc/self/oom_score_adj` (0 = foreground, 900+ = kill imminent).
2. Available RAM via `/proc/meminfo` (MemAvailable) and fallback to psutil.
3. Linux Pressure Stall Information (PSI) via `/proc/pressure/memory`.

This telemetry allows LLMHub to proactively prune KV cache slots, reject oversized
allocations, downshift context windows, and avoid abrupt OS termination.
"""

import enum
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("llmhub.termux.memory_pressure")

class MemoryPressureLevel(enum.Enum):
    NOMINAL = "nominal"      # Plenty of headroom (> 25% or > 1.2 GB free)
    MODERATE = "moderate"    # Constrained (12% - 25% or 500 MB - 1.2 GB free)
    CRITICAL = "critical"    # Danger of Android LMK kill (< 12% or < 500 MB free)

@dataclass
class MemoryPressureStats:
    level: MemoryPressureLevel
    total_ram_bytes: int
    available_ram_bytes: int
    free_ram_bytes: int
    available_percent: float
    oom_score_adj: Optional[int] = None
    psi_some_avg10: Optional[float] = None
    psi_full_avg10: Optional[float] = None
    source: str = "none"

class MemoryPressureMonitor:
    """Monitors OS memory pressure, LMK kill priority, and PSI stall metrics."""

    def __init__(
        self,
        moderate_threshold_ratio: float = 0.22,
        critical_threshold_ratio: float = 0.12,
        moderate_min_bytes: int = 1200 * 1024 * 1024,  # 1.2 GB
        critical_min_bytes: int = 450 * 1024 * 1024,   # 450 MB
    ) -> None:
        self.moderate_threshold_ratio = moderate_threshold_ratio
        self.critical_threshold_ratio = critical_threshold_ratio
        self.moderate_min_bytes = moderate_min_bytes
        self.critical_min_bytes = critical_min_bytes

        self._injected_stats: Optional[MemoryPressureStats] = None

    def inject_stats(self, stats: Optional[MemoryPressureStats]) -> None:
        """Inject synthetic stats for deterministic testing."""
        self._injected_stats = stats

    @staticmethod
    def read_oom_score_adj() -> Optional[int]:
        """Read Android / Linux /proc/self/oom_score_adj."""
        path = Path("/proc/self/oom_score_adj")
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    val = f.read().strip()
                return int(val)
            except Exception:
                pass
        return None

    @staticmethod
    def read_proc_meminfo() -> Optional[Dict[str, int]]:
        """Parse Linux / Android /proc/meminfo into byte values."""
        path = Path("/proc/meminfo")
        if not path.exists():
            return None
        try:
            info: Dict[str, int] = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val_str = parts[1].strip()
                        val_parts = val_str.split()
                        if val_parts and val_parts[0].isdigit():
                            val = int(val_parts[0])
                            # If unit is kB, convert to bytes
                            if len(val_parts) > 1 and val_parts[1].lower() == "kb":
                                val *= 1024
                            info[key] = val
            return info
        except Exception:
            return None

    @staticmethod
    def read_proc_pressure_memory() -> Tuple[Optional[float], Optional[float]]:
        """Parse Linux PSI /proc/pressure/memory for some_avg10 and full_avg10."""
        path = Path("/proc/pressure/memory")
        if not path.exists():
            return None, None
        try:
            some_avg10 = None
            full_avg10 = None
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    prefix = parts[0]
                    for field in parts[1:]:
                        if field.startswith("avg10="):
                            val = float(field.split("=")[1])
                            if prefix == "some":
                                some_avg10 = val
                            elif prefix == "full":
                                full_avg10 = val
            return some_avg10, full_avg10
        except Exception:
            return None, None

    def probe(self) -> MemoryPressureStats:
        """Probe memory pressure, LMK score, and PSI metrics."""
        if self._injected_stats is not None:
            return self._injected_stats

        oom_score = self.read_oom_score_adj()
        psi_some, psi_full = self.read_proc_pressure_memory()

        # Try /proc/meminfo
        meminfo = self.read_proc_meminfo()
        if meminfo and "MemTotal" in meminfo:
            total_b = meminfo["MemTotal"]
            avail_b = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            free_b = meminfo.get("MemFree", 0)
            avail_pct = max(0.0, min(100.0, (avail_b / total_b) * 100.0 if total_b > 0 else 0.0))
            level = self._classify_pressure(avail_b, total_b, oom_score, psi_full)
            return MemoryPressureStats(
                level=level,
                total_ram_bytes=total_b,
                available_ram_bytes=avail_b,
                free_ram_bytes=free_b,
                available_percent=avail_pct,
                oom_score_adj=oom_score,
                psi_some_avg10=psi_some,
                psi_full_avg10=psi_full,
                source="procfs",
            )

        # Fallback to psutil
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_b = vm.total
            avail_b = vm.available
            free_b = getattr(vm, "free", avail_b)
            avail_pct = max(0.0, min(100.0, (avail_b / total_b) * 100.0 if total_b > 0 else 0.0))
            level = self._classify_pressure(avail_b, total_b, oom_score, psi_full)
            return MemoryPressureStats(
                level=level,
                total_ram_bytes=total_b,
                available_ram_bytes=avail_b,
                free_ram_bytes=free_b,
                available_percent=avail_pct,
                oom_score_adj=oom_score,
                psi_some_avg10=psi_some,
                psi_full_avg10=psi_full,
                source="psutil",
            )
        except Exception:
            # Safe default fallback
            return MemoryPressureStats(
                level=MemoryPressureLevel.NOMINAL,
                total_ram_bytes=4 * 1024 * 1024 * 1024,
                available_ram_bytes=2 * 1024 * 1024 * 1024,
                free_ram_bytes=1 * 1024 * 1024 * 1024,
                available_percent=50.0,
                oom_score_adj=oom_score,
                source="fallback",
            )

    def _classify_pressure(
        self,
        avail_bytes: int,
        total_bytes: int,
        oom_score: Optional[int],
        psi_full: Optional[float]
    ) -> MemoryPressureLevel:
        ratio = (avail_bytes / total_bytes) if total_bytes > 0 else 0.0

        # High PSI stall (> 15% of time spent in direct reclaim) signals critical thrashing
        if psi_full is not None and psi_full > 15.0:
            return MemoryPressureLevel.CRITICAL

        # High oom_score_adj (> 800) means process is about to be terminated by LMK
        if oom_score is not None and oom_score > 800 and ratio < 0.20:
            return MemoryPressureLevel.CRITICAL

        if avail_bytes < self.critical_min_bytes or ratio < self.critical_threshold_ratio:
            return MemoryPressureLevel.CRITICAL

        if avail_bytes < self.moderate_min_bytes or ratio < self.moderate_threshold_ratio:
            return MemoryPressureLevel.MODERATE

        return MemoryPressureLevel.NOMINAL
