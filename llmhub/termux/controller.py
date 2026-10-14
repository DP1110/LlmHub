"""Termux and Low-Memory Controller.

Coordinates Android wake-lock lifecycle, background thermal monitoring with
adaptive thread throttling, and memory pressure surveillance. Integrates with the
LLMHub scheduler, model manager, and continuous batch engine.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from llmhub.config import LLMHubConfig, is_termux
from llmhub.termux.downshift import DownshiftEngine
from llmhub.termux.memory_pressure import MemoryPressureLevel, MemoryPressureMonitor, MemoryPressureStats
from llmhub.termux.thermal import ThermalMonitor, ThermalReading, ThermalState
from llmhub.termux.wakelock import WakelockManager

logger = logging.getLogger("llmhub.termux.controller")

class TermuxController:
    """Central supervisor for mobile / low-memory environment operations."""

    def __init__(
        self,
        config: Optional[LLMHubConfig] = None,
        wakelock_manager: Optional[WakelockManager] = None,
        thermal_monitor: Optional[ThermalMonitor] = None,
        memory_monitor: Optional[MemoryPressureMonitor] = None,
        base_threads: int = 4,
    ) -> None:
        self.config = config or LLMHubConfig()
        termux_cfg = self.config.settings.get("termux", {})

        self.enable_wake_lock = termux_cfg.get("enable_wake_lock", True)
        self.enable_thermal = termux_cfg.get("enable_thermal_throttle", True)
        self.poll_interval_s = float(termux_cfg.get("thermal_poll_interval_s", 5.0))
        self.base_threads = base_threads
        self.effective_threads = base_threads

        self.wakelock = wakelock_manager or WakelockManager()
        self.thermal = thermal_monitor or ThermalMonitor(
            warning_temp_c=float(termux_cfg.get("thermal_warning_temp_c", 75.0)),
            critical_temp_c=float(termux_cfg.get("thermal_critical_temp_c", 85.0)),
        )
        self.memory = memory_monitor or MemoryPressureMonitor()
        self.downshift = DownshiftEngine()

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._thread_change_callback: Optional[Callable[[int], None]] = None
        self._memory_pressure_callback: Optional[Callable[[MemoryPressureLevel], None]] = None

    def set_thread_callback(self, cb: Callable[[int], None]) -> None:
        """Register callback invoked when thermal conditions change thread allocation."""
        self._thread_change_callback = cb

    def set_memory_pressure_callback(self, cb: Callable[[MemoryPressureLevel], None]) -> None:
        """Register callback invoked when memory pressure level changes."""
        self._memory_pressure_callback = cb

    async def start(self) -> None:
        """Start asynchronous monitoring loop and acquire wake-lock."""
        if self._running:
            return
        self._running = True
        if self.enable_wake_lock and self.wakelock.is_supported and not self.wakelock.is_held:
            await asyncio.to_thread(self.wakelock.acquire)
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "Started Termux/Low-Memory controller loop (interval: %.1fs, base threads: %d).",
            self.poll_interval_s,
            self.base_threads,
        )

    async def stop(self) -> None:
        """Stop monitoring loop and release wake lock."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        if self.wakelock.is_held:
            await asyncio.to_thread(self.wakelock.release)
        logger.info("Stopped Termux/Low-Memory controller.")

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await asyncio.to_thread(self.tick)
            except Exception as e:
                logger.error("Error in Termux controller monitor loop: %s", e)
            await asyncio.sleep(self.poll_interval_s)

    def tick(self) -> Dict[str, Any]:
        """Perform one synchronous evaluation step of thermal and memory state."""
        # 1. Thermal evaluation
        reading, target_threads = self.thermal.update_and_get_target_threads(self.base_threads)
        if target_threads != self.effective_threads:
            logger.warning(
                "Adaptive thread governor adjusting threads: %d -> %d (thermal state: %s, max temp: %.1f°C).",
                self.effective_threads,
                target_threads,
                reading.state.value.upper(),
                reading.max_temp_celsius,
            )
            self.effective_threads = target_threads
            if self._thread_change_callback:
                try:
                    self._thread_change_callback(target_threads)
                except Exception as e:
                    logger.error("Thread change callback failed: %s", e)

        # 2. Memory pressure evaluation
        mem_stats = self.memory.probe()
        if mem_stats.level == MemoryPressureLevel.CRITICAL:
            logger.warning(
                "CRITICAL memory pressure detected! Avail: %.1f MB (%.1f%%). OOM score adj: %s. "
                "Active downshifting engaged to prevent Android LMK kill.",
                mem_stats.available_ram_bytes / (1024**2),
                mem_stats.available_percent,
                mem_stats.oom_score_adj,
            )
        elif mem_stats.level == MemoryPressureLevel.MODERATE:
            logger.info(
                "Moderate memory pressure: %.1f MB available (%.1f%%).",
                mem_stats.available_ram_bytes / (1024**2),
                mem_stats.available_percent,
            )

        if self._memory_pressure_callback:
            try:
                self._memory_pressure_callback(mem_stats.level)
            except Exception as e:
                logger.error("Memory pressure callback failed: %s", e)

        return self.get_status(reading=reading, mem_stats=mem_stats)

    def get_status(
        self,
        reading: Optional[ThermalReading] = None,
        mem_stats: Optional[MemoryPressureStats] = None,
    ) -> Dict[str, Any]:
        """Return comprehensive mobile and low-memory telemetry."""
        r = reading or self.thermal.probe()
        m = mem_stats or self.memory.probe()

        from llmhub.hardware.detector import HardwareDetector
        prof = HardwareDetector.get_profile()

        return {
            "is_termux": prof.is_termux,
            "is_non_root": prof.is_non_root,
            "low_memory_mode": self.config.is_low_memory_mode,
            "wakelock": {
                "is_supported": self.wakelock.is_supported,
                "is_held": self.wakelock.is_held,
            },
            "thermal": {
                "state": r.state.value,
                "max_temp_c": round(r.max_temp_celsius, 1),
                "avg_temp_c": round(r.avg_temp_celsius, 1),
                "source": r.source,
                "base_threads": self.base_threads,
                "effective_threads": self.effective_threads,
                "zones": [
                    {
                        "id": z.zone_id,
                        "type": z.zone_type,
                        "temp_c": round(z.temp_celsius, 1),
                    }
                    for z in r.zones
                ],
            },
            "memory_pressure": {
                "level": m.level.value,
                "total_ram_mb": round(m.total_ram_bytes / (1024**2), 1),
                "available_ram_mb": round(m.available_ram_bytes / (1024**2), 1),
                "available_percent": round(m.available_percent, 1),
                "oom_score_adj": m.oom_score_adj,
                "psi_some_avg10": m.psi_some_avg10,
                "psi_full_avg10": m.psi_full_avg10,
                "source": m.source,
            },
        }
