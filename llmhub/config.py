"""Configuration and filesystem layout management for LLMHub.

Storage layout:
~/.llmhub/
    config.toml
    registry.db
    models/<org>/<name>/<quant>/{model.gguf, manifest.json}
    logs/
    run/
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import toml

# Environment variable to override base directory
ENV_LLMHUB_HOME = "LLMHUB_HOME"

def is_termux() -> bool:
    """Detect whether running inside Android Termux environment."""
    return "TERMUX_VERSION" in os.environ or os.path.exists("/data/data/com.termux")

def get_base_dir() -> Path:
    """Return the root storage directory for LLMHub (~/.llmhub or $LLMHUB_HOME)."""
    env_home = os.environ.get(ENV_LLMHUB_HOME)
    if env_home:
        base = Path(env_home).expanduser().resolve()
    else:
        # Cross-platform home resolution:
        # Windows: C:\Users\<user>\.llmhub
        # Linux/macOS: ~/.llmhub
        # Termux: /data/data/com.termux/files/home/.llmhub
        base = Path.home() / ".llmhub"
    return base

class LLMHubConfig:
    """Manages directory hierarchy and config.toml persistence."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or get_base_dir()
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.run_dir = self.base_dir / "run"
        self.config_path = self.base_dir / "config.toml"
        self.db_path = self.base_dir / "registry.db"
        
        self.settings: Dict[str, Any] = self._default_settings()
        self.ensure_dirs()
        self.load()

    def _default_settings(self) -> Dict[str, Any]:
        return {
            "server": {
                "host": "127.0.0.1",
                "port": 11434,
                "timeout_keep_alive": 300,
            },
            "runtime": {
                "default_threads": max(1, (os.cpu_count() or 2) // 2),
                "default_context_size": 2048,
                "use_mmap": True,
                "use_mlock": False,  # Disabled by default for Termux / unprivileged OS safety
                "memory_safety_margin": 0.85,
            },
            "scheduler": {
                "quantum_tokens": 4,
                "prefill_chunk_size": 256,
                "threading_mode": "time-slice",  # time-slice | partition | hybrid
            },
            "batch": {
                "default_max_slots": 4,
                "enable_continuous_batching": True,
                "max_cached_tokens": 32768,
            },
            "storage": {
                "models_dir": str(self.models_dir),
                "db_path": str(self.db_path),
            },
            "termux": {
                "enable_wake_lock": True,
                "enable_thermal_throttle": True,
                "thermal_poll_interval_s": 5.0,
                "thermal_warning_temp_c": 75.0,
                "thermal_critical_temp_c": 85.0,
                "enable_memory_pressure_monitor": True,
                "low_memory_mode": False,
                "allow_ctx_downshift": True,
                "min_ctx_size": 512,
            },
            "observability": {
                "metrics_enabled": True,
                "log_level": "INFO",
                "log_json": True,
            },
            "security": {
                "auth_enabled": False,
                "api_keys": [],
                "key_configs": {},
                "rate_limit_enabled": True,
                "default_rpm": 60,
                "default_tpm": 30000,
                "max_request_body_bytes": 4 * 1024 * 1024,  # 4 MB
                "exempt_localhost": True,
            },
            "lifecycle": {
                "drain_timeout_s": 30.0,
            }
        }

    def ensure_dirs(self) -> None:
        """Create necessary directory hierarchy with appropriate permissions."""
        for directory in [self.base_dir, self.models_dir, self.logs_dir, self.run_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Load configuration from config.toml if it exists, otherwise write defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = toml.load(f)
                    for section, values in loaded.items():
                        if section in self.settings and isinstance(values, dict):
                            self.settings[section].update(values)
                        else:
                            self.settings[section] = values
            except Exception as e:
                # If corrupt, keep defaults and warn
                sys.stderr.write(f"[WARN] Failed to read {self.config_path}: {e}. Using defaults.\n")
        else:
            self.save()

    def save(self) -> None:
        """Serialize configuration back to config.toml."""
        self.ensure_dirs()
        with open(self.config_path, "w", encoding="utf-8") as f:
            toml.dump(self.settings, f)

    def get_model_dir(self, org: str, name: str, quant: str) -> Path:
        """Return the directory path for a specific model quant: models/<org>/<name>/<quant>."""
        path = self.models_dir / org / name / quant.lower()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_low_memory_mode(self) -> bool:
        """Check if low-memory mode is explicitly requested or running on tight RAM / Termux."""
        return bool(self.settings.get("termux", {}).get("low_memory_mode", False))

    def set_low_memory_mode(self, enabled: bool) -> None:
        """Toggle low-memory mode, adjusting context and safety settings."""
        if "termux" not in self.settings:
            self.settings["termux"] = {}
        self.settings["termux"]["low_memory_mode"] = enabled
        if enabled:
            # Conservative tuning for low-memory environments
            self.settings["runtime"]["default_context_size"] = min(
                1024, self.settings.get("runtime", {}).get("default_context_size", 2048)
            )
            self.settings["runtime"]["memory_safety_margin"] = 0.90
            self.settings["runtime"]["use_mlock"] = False
            self.settings["batch"]["default_max_slots"] = min(
                2, self.settings.get("batch", {}).get("default_max_slots", 4)
            )
            self.settings["batch"]["max_cached_tokens"] = min(
                8192, self.settings.get("batch", {}).get("max_cached_tokens", 32768)
            )

    def validate(self, strict: bool = False) -> List[str]:
        """Validate currently loaded configuration settings."""
        errors = validate_config_schema(self.settings)
        if errors:
            msg = f"Configuration validation warnings in {self.config_path}: {'; '.join(errors)}"
            if strict:
                raise ConfigValidationError(msg)
            sys.stderr.write(f"[WARN] {msg}\n")
        return errors


class ConfigValidationError(ValueError):
    """Raised when configuration values fail strict schema constraints."""
    pass


def validate_config_schema(settings: Dict[str, Any]) -> List[str]:
    """Validate configuration dictionary against strict schema rules.
    
    Returns a list of error diagnostic messages.
    """
    errors: List[str] = []
    known_sections = {
        "server", "runtime", "scheduler", "batch", "storage",
        "termux", "observability", "security", "lifecycle"
    }

    for section in settings.keys():
        if section not in known_sections:
            errors.append(f"Unknown configuration section '[{section}]'")

    # 1. Server section
    server = settings.get("server", {})
    if "port" in server:
        port = server["port"]
        if not isinstance(port, int) or port < 1 or port > 65535:
            errors.append(f"server.port must be an integer between 1 and 65535 (got {port!r})")
    if "host" in server and not isinstance(server["host"], str):
        errors.append(f"server.host must be a string (got {server['host']!r})")

    # 2. Runtime section
    runtime = settings.get("runtime", {})
    if "memory_safety_margin" in runtime:
        margin = runtime["memory_safety_margin"]
        if not isinstance(margin, (int, float)) or margin <= 0.0 or margin > 1.0:
            errors.append(f"runtime.memory_safety_margin must be a float in (0.0, 1.0] (got {margin!r})")
    if "default_threads" in runtime:
        threads = runtime["default_threads"]
        if not isinstance(threads, int) or threads < 1:
            errors.append(f"runtime.default_threads must be an integer >= 1 (got {threads!r})")
    if "default_context_size" in runtime:
        ctx = runtime["default_context_size"]
        if not isinstance(ctx, int) or ctx < 64:
            errors.append(f"runtime.default_context_size must be an integer >= 64 (got {ctx!r})")

    # 3. Scheduler section
    sched = settings.get("scheduler", {})
    if "quantum_tokens" in sched:
        q = sched["quantum_tokens"]
        if not isinstance(q, int) or q < 1:
            errors.append(f"scheduler.quantum_tokens must be an integer >= 1 (got {q!r})")
    if "prefill_chunk_size" in sched:
        p = sched["prefill_chunk_size"]
        if not isinstance(p, int) or p < 1:
            errors.append(f"scheduler.prefill_chunk_size must be an integer >= 1 (got {p!r})")
    if "threading_mode" in sched:
        mode = sched["threading_mode"]
        if mode not in ("time-slice", "partition", "hybrid"):
            errors.append(f"scheduler.threading_mode must be one of 'time-slice', 'partition', 'hybrid' (got {mode!r})")

    # 4. Security section
    sec = settings.get("security", {})
    if "default_rpm" in sec:
        rpm = sec["default_rpm"]
        if not isinstance(rpm, int) or rpm < 1:
            errors.append(f"security.default_rpm must be an integer >= 1 (got {rpm!r})")
    if "default_tpm" in sec:
        tpm = sec["default_tpm"]
        if not isinstance(tpm, int) or tpm < 1:
            errors.append(f"security.default_tpm must be an integer >= 1 (got {tpm!r})")
    if "max_request_body_bytes" in sec:
        max_b = sec["max_request_body_bytes"]
        if not isinstance(max_b, int) or max_b < 1024:
            errors.append(f"security.max_request_body_bytes must be an integer >= 1024 (got {max_b!r})")

    # 5. Termux section
    termux = settings.get("termux", {})
    warn_t = termux.get("thermal_warning_temp_c")
    crit_t = termux.get("thermal_critical_temp_c")
    if warn_t is not None and (not isinstance(warn_t, (int, float)) or warn_t <= 0):
        errors.append(f"termux.thermal_warning_temp_c must be a positive number (got {warn_t!r})")
    if crit_t is not None and (not isinstance(crit_t, (int, float)) or crit_t <= 0):
        errors.append(f"termux.thermal_critical_temp_c must be a positive number (got {crit_t!r})")
    if warn_t is not None and crit_t is not None and isinstance(warn_t, (int, float)) and isinstance(crit_t, (int, float)):
        if crit_t <= warn_t:
            errors.append(f"termux.thermal_critical_temp_c ({crit_t}) must be greater than thermal_warning_temp_c ({warn_t})")

    return errors


