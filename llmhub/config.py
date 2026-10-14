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
from typing import Any, Dict
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
