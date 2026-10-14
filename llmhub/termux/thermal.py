"""Thermal Throttling Monitor and Adaptive Thread Governor.

On mobile ARM SoCs (Snapdragon, MediaTek, Exynos, Google Tensor), continuous LLM
inference generates significant heat, quickly reaching 75°C-85°C. Unmanaged, this
causes the Linux/Android kernel thermal governor to throttle frequencies drastically
or initiate emergency thermal shutdown.

This module probes SoC thermal sensors and battery temperature, categorizes thermal
severity (NORMAL, WARM, HOT, CRITICAL), and calculates throttled thread allocations
using a hysteresis filter to prevent thread oscillation.
"""

import enum
import glob
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("llmhub.termux.thermal")

class ThermalState(enum.Enum):
    NORMAL = "normal"      # < 65°C: full performance cores
    WARM = "warm"          # 65°C - 75°C: warning threshold
    HOT = "hot"            # 75°C - 85°C: throttle to 50% threads
    CRITICAL = "critical"  # > 85°C: throttle to 1 thread to avoid thermal shutdown

@dataclass
class ThermalZoneInfo:
    zone_id: str
    zone_type: str
    temp_celsius: float

@dataclass
class ThermalReading:
    max_temp_celsius: float
    avg_temp_celsius: float
    state: ThermalState
    zones: List[ThermalZoneInfo] = field(default_factory=list)
    source: str = "none"

class ThermalMonitor:
    """Probes system thermal sensors and computes adaptive thread limits."""

    def __init__(
        self,
        warning_temp_c: float = 75.0,
        critical_temp_c: float = 85.0,
        warm_temp_c: float = 65.0,
        cooldown_margin_c: float = 3.0,
        consecutive_cool_required: int = 2,
    ) -> None:
        self.warning_temp_c = warning_temp_c
        self.critical_temp_c = critical_temp_c
        self.warm_temp_c = warm_temp_c
        self.cooldown_margin_c = cooldown_margin_c
        self.consecutive_cool_required = consecutive_cool_required

        self._current_state = ThermalState.NORMAL
        self._cool_reading_count = 0
        self._injected_temp: Optional[float] = None
        self._callbacks: List[Callable[[ThermalState, ThermalState, int], None]] = []

    @property
    def current_state(self) -> ThermalState:
        return self._current_state

    def inject_temperature(self, temp_c: Optional[float]) -> None:
        """Inject synthetic temperature reading for deterministic unit testing."""
        self._injected_temp = temp_c

    def register_callback(
        self, callback: Callable[[ThermalState, ThermalState, int], None]
    ) -> None:
        """Register a callback for thermal state transitions: (old_state, new_state, suggested_threads)."""
        self._callbacks.append(callback)

    def read_sysfs_thermal(self) -> List[ThermalZoneInfo]:
        """Read Linux / Android /sys/class/thermal/thermal_zone* sensors."""
        zones: List[ThermalZoneInfo] = []
        zone_paths = glob.glob("/sys/class/thermal/thermal_zone*")
        
        for path_str in zone_paths:
            p = Path(path_str)
            zone_id = p.name
            type_file = p / "type"
            temp_file = p / "temp"

            zone_type = "unknown"
            if type_file.exists():
                try:
                    with open(type_file, "r", encoding="utf-8", errors="ignore") as f:
                        zone_type = f.read().strip()
                except Exception:
                    pass

            if temp_file.exists():
                try:
                    with open(temp_file, "r", encoding="utf-8", errors="ignore") as f:
                        raw_temp = f.read().strip()
                    val = float(raw_temp)
                    # sysfs temperatures > 1000 are in millidegrees Celsius
                    temp_c = val / 1000.0 if val > 1000 else val
                    # Sanity check: ignore erroneous readings like negative or > 150C
                    if 0.0 < temp_c < 150.0:
                        zones.append(ThermalZoneInfo(zone_id=zone_id, zone_type=zone_type, temp_celsius=temp_c))
                except Exception:
                    pass

        return zones

    def read_termux_battery_temp(self) -> Optional[float]:
        """Read battery temperature from termux-battery-status command if available."""
        cmd = shutil.which("termux-battery-status")
        if not cmd:
            prefix_cmd = Path("/data/data/com.termux/files/usr/bin/termux-battery-status")
            if prefix_cmd.exists() and os.access(prefix_cmd, os.X_OK):
                cmd = str(prefix_cmd)
        if not cmd:
            return None

        try:
            res = subprocess.run(
                [cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3.0,
                check=False
            )
            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout)
                temp = data.get("temperature")
                if isinstance(temp, (int, float)):
                    temp_c = temp / 10.0 if temp >= 100.0 else float(temp)
                    if 0.0 < temp_c < 100.0:
                        return temp_c
        except Exception:
            pass
        return None

    def read_psutil_temperatures(self) -> List[ThermalZoneInfo]:
        """Fallback to psutil.sensors_temperatures() if available on Linux/macOS."""
        zones: List[ThermalZoneInfo] = []
        try:
            import psutil
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for idx, entry in enumerate(entries):
                            if entry.current and 0.0 < entry.current < 150.0:
                                zones.append(
                                    ThermalZoneInfo(
                                        zone_id=f"{name}_{idx}",
                                        zone_type=entry.label or name,
                                        temp_celsius=float(entry.current),
                                    )
                                )
        except Exception:
            pass
        return zones

    def probe(self) -> ThermalReading:
        """Probe all available thermal sources and return current ThermalReading."""
        # Check test injection first
        if self._injected_temp is not None:
            temp = self._injected_temp
            state = self._classify_temperature(temp)
            return ThermalReading(
                max_temp_celsius=temp,
                avg_temp_celsius=temp,
                state=state,
                zones=[ThermalZoneInfo(zone_id="injected", zone_type="mock", temp_celsius=temp)],
                source="injected",
            )

        # 1. Sysfs thermal zones
        zones = self.read_sysfs_thermal()
        if zones:
            # Prioritize CPU/SoC zones if available
            cpu_zones = [
                z for z in zones
                if any(k in z.zone_type.lower() for k in ["cpu", "soc", "tsens", "cluster", "ap"])
            ]
            eval_zones = cpu_zones if cpu_zones else zones
            max_t = max(z.temp_celsius for z in eval_zones)
            avg_t = sum(z.temp_celsius for z in eval_zones) / len(eval_zones)
            state = self._classify_temperature(max_t)
            return ThermalReading(
                max_temp_celsius=max_t,
                avg_temp_celsius=avg_t,
                state=state,
                zones=zones,
                source="sysfs",
            )

        # 2. Battery status via Termux-API
        batt_temp = self.read_termux_battery_temp()
        if batt_temp is not None:
            # Battery temperature runs cooler than CPU die (~10-15C lower).
            # We scale or use as proxy
            state = self._classify_temperature(batt_temp + 10.0)
            return ThermalReading(
                max_temp_celsius=batt_temp,
                avg_temp_celsius=batt_temp,
                state=state,
                zones=[ThermalZoneInfo(zone_id="battery", zone_type="battery", temp_celsius=batt_temp)],
                source="termux-battery",
            )

        # 3. psutil sensors
        psutil_zones = self.read_psutil_temperatures()
        if psutil_zones:
            max_t = max(z.temp_celsius for z in psutil_zones)
            avg_t = sum(z.temp_celsius for z in psutil_zones) / len(psutil_zones)
            state = self._classify_temperature(max_t)
            return ThermalReading(
                max_temp_celsius=max_t,
                avg_temp_celsius=avg_t,
                state=state,
                zones=psutil_zones,
                source="psutil",
            )

        # 4. Unknown / fallback (assume NORMAL 45°C nominal)
        return ThermalReading(
            max_temp_celsius=45.0,
            avg_temp_celsius=45.0,
            state=ThermalState.NORMAL,
            zones=[],
            source="default",
        )

    def _classify_temperature(self, temp_c: float) -> ThermalState:
        if temp_c >= self.critical_temp_c:
            return ThermalState.CRITICAL
        elif temp_c >= self.warning_temp_c:
            return ThermalState.HOT
        elif temp_c >= self.warm_temp_c:
            return ThermalState.WARM
        return ThermalState.NORMAL

    def update_and_get_target_threads(self, base_threads: int) -> Tuple[ThermalReading, int]:
        """Probe temperature, update thermal state with hysteresis, and calculate target threads.
        
        Returns: (reading, target_threads)
        """
        reading = self.probe()
        raw_state = reading.state

        # State transition with hysteresis to prevent oscillation
        new_state = self._apply_hysteresis(raw_state, reading.max_temp_celsius)
        
        target_threads = self.calculate_target_threads(base_threads, new_state)

        if new_state != self._current_state:
            old_state = self._current_state
            self._current_state = new_state
            logger.warning(
                "Thermal state transition: %s -> %s (max temp: %.1f°C). Adjusted threads: %d -> %d.",
                old_state.value.upper(),
                new_state.value.upper(),
                reading.max_temp_celsius,
                base_threads,
                target_threads,
            )
            for cb in self._callbacks:
                try:
                    cb(old_state, new_state, target_threads)
                except Exception as e:
                    logger.error("Thermal callback error: %s", e)

        return reading, target_threads

    def _apply_hysteresis(self, raw_state: ThermalState, temp_c: float) -> ThermalState:
        """Apply cooldown hysteresis so we don't bounce between states."""
        # If escalating (hotter), transition immediately
        severity_order = {
            ThermalState.NORMAL: 0,
            ThermalState.WARM: 1,
            ThermalState.HOT: 2,
            ThermalState.CRITICAL: 3,
        }

        if severity_order[raw_state] > severity_order[self._current_state]:
            self._cool_reading_count = 0
            return raw_state

        if severity_order[raw_state] < severity_order[self._current_state]:
            # Cooldown requires temperature to drop below the previous threshold minus cooldown_margin_c
            # Check threshold of current state
            needed_temp = 999.0
            if self._current_state == ThermalState.CRITICAL:
                needed_temp = self.critical_temp_c - self.cooldown_margin_c
            elif self._current_state == ThermalState.HOT:
                needed_temp = self.warning_temp_c - self.cooldown_margin_c
            elif self._current_state == ThermalState.WARM:
                needed_temp = self.warm_temp_c - self.cooldown_margin_c

            if temp_c <= needed_temp:
                self._cool_reading_count += 1
                if self._cool_reading_count >= self.consecutive_cool_required:
                    self._cool_reading_count = 0
                    return raw_state
            else:
                self._cool_reading_count = 0
            return self._current_state

        self._cool_reading_count = 0
        return self._current_state

    @staticmethod
    def calculate_target_threads(base_threads: int, state: ThermalState) -> int:
        """Determine CPU worker thread count given base threads and thermal state."""
        b = max(1, base_threads)
        if state == ThermalState.CRITICAL:
            # Emergency thermal limit: 1 thread to avoid power surge & kernel shutdown
            return 1
        elif state == ThermalState.HOT:
            # 50% thread reduction
            return max(1, b // 2)
        elif state == ThermalState.WARM:
            # If high thread count, shed 1 thread to slow temperature rise
            return max(1, b - 1 if b >= 4 else b)
        else:
            return b
