import os
from pathlib import Path
import pytest
from llmhub.config import LLMHubConfig, get_base_dir, is_termux

def test_config_initialization(tmp_path: Path):
    config = LLMHubConfig(base_dir=tmp_path)
    assert config.base_dir == tmp_path
    assert config.models_dir.exists()
    assert config.logs_dir.exists()
    assert config.run_dir.exists()
    assert config.config_path.exists()
    assert config.settings["server"]["port"] == 11434
    assert config.settings["scheduler"]["quantum_tokens"] == 4

def test_config_save_and_reload(tmp_path: Path):
    config1 = LLMHubConfig(base_dir=tmp_path)
    config1.settings["runtime"]["default_context_size"] = 4096
    config1.save()

    config2 = LLMHubConfig(base_dir=tmp_path)
    assert config2.settings["runtime"]["default_context_size"] == 4096

def test_get_model_dir(tmp_path: Path):
    config = LLMHubConfig(base_dir=tmp_path)
    model_dir = config.get_model_dir("Qwen", "Qwen2.5-0.5B", "Q4_K_M")
    assert model_dir.exists()
    assert model_dir == tmp_path / "models" / "Qwen" / "Qwen2.5-0.5B" / "q4_k_m"

def test_is_termux_detection():
    # Verify function executes without throwing
    res = is_termux()
    assert isinstance(res, bool)
