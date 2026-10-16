from pathlib import Path
from llmhub.core.registry import ModelRegistry

def test_registry_lifecycle(tmp_path: Path):
    db_file = tmp_path / "registry.db"
    reg = ModelRegistry(db_file)

    # Register model
    rec = reg.register_model(
        org="Qwen",
        name="Qwen2.5-0.5B",
        quant="Q4_K_M",
        file_path=tmp_path / "model.gguf",
        manifest_path=tmp_path / "manifest.json",
        size_bytes=390000000,
        sha256="abcdef1234567890",
        architecture="qwen2",
        n_layers=24,
        n_kv_heads=2,
        head_dim=64,
        ctx_max=32768
    )

    assert rec.id is not None
    assert rec.tag == "Qwen/Qwen2.5-0.5B:q4_k_m"

    # List models
    models = reg.list_models()
    assert len(models) == 1
    assert models[0].name == "Qwen2.5-0.5B"

    # Find model by full tag
    found1 = reg.find_model("Qwen/Qwen2.5-0.5B:q4_k_m")
    assert found1 is not None
    assert found1.id == rec.id

    # Find model by short name:quant
    found2 = reg.find_model("Qwen2.5-0.5B:q4_k_m")
    assert found2 is not None
    assert found2.id == rec.id

    # Find model by name only
    found3 = reg.find_model("Qwen2.5-0.5B")
    assert found3 is not None
    assert found3.id == rec.id

    # Remove model
    removed = reg.remove_model("Qwen2.5-0.5B:q4_k_m")
    assert removed is not None
    assert len(reg.list_models()) == 0


def test_registry_case_insensitive_matching(tmp_path: Path):
    db_file = tmp_path / "registry.db"
    reg = ModelRegistry(db_file)
    rec = reg.register_model(
        org="Qwen",
        name="Qwen2.5-0.5B-Instruct",
        quant="q4_k_m",
        file_path=tmp_path / "model.gguf",
        manifest_path=tmp_path / "manifest.json",
        size_bytes=390000000,
        sha256="",
        architecture="qwen2",
        n_layers=24,
        n_kv_heads=2,
        head_dim=64,
        ctx_max=32768
    )

    # Test lowercase and hyphen variations
    assert reg.find_model("qwen/qwen2.5-0.5b-instruct:q4_k_m") is not None
    assert reg.find_model("qwen2.5-0.5b-instruct:q4_k_m") is not None
    assert reg.find_model("qwen2.5-0.5b-instruct-q4_k_m") is not None


def test_scan_and_register_local_models(tmp_path: Path):
    from tests.test_gguf_parser import create_synthetic_gguf

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    gguf_path = models_dir / "test-model-q4_k_m.gguf"
    create_synthetic_gguf(gguf_path)

    db_file = tmp_path / "registry.db"
    reg = ModelRegistry(db_file)

    assert len(reg.list_models()) == 0
    scanned = reg.scan_and_register_local_models(models_dir)
    assert len(scanned) == 1
    assert scanned[0].file_path == str(gguf_path.resolve())

    # Idempotent re-scan
    scanned_again = reg.scan_and_register_local_models(models_dir)
    assert len(scanned_again) == 0
    assert len(reg.list_models()) == 1


def test_registry_alias_lookup(tmp_path: Path):
    db_file = tmp_path / "registry.db"
    reg = ModelRegistry(db_file)
    rec = reg.register_model(
        org="bartowski",
        name="SmolLM2-135M-Instruct-GGUF",
        quant="q4_k_m",
        file_path=tmp_path / "model.gguf",
        manifest_path=tmp_path / "manifest.json",
        size_bytes=100000000,
        sha256="",
        architecture="llama",
        n_layers=16,
        n_kv_heads=4,
        head_dim=64,
        ctx_max=8192
    )

    # Shorthand alias lookups
    found = reg.find_model("smollm2:135m")
    assert found is not None
    assert found.id == rec.id

    found_base = reg.find_model("smollm2")
    assert found_base is not None
    assert found_base.id == rec.id

    found_hyphen = reg.find_model("smollm2-135m")
    assert found_hyphen is not None
    assert found_hyphen.id == rec.id


def test_registry_alias_lookup_with_non_default_quant(tmp_path: Path):
    db_file = tmp_path / "registry.db"
    reg = ModelRegistry(db_file)
    rec = reg.register_model(
        org="bartowski",
        name="SmolLM2-135M-Instruct-GGUF",
        quant="q8_0",
        file_path=tmp_path / "model.gguf",
        manifest_path=tmp_path / "manifest.json",
        size_bytes=150000000,
        sha256="",
        architecture="llama",
        n_layers=16,
        n_kv_heads=4,
        head_dim=64,
        ctx_max=8192
    )

    # Shorthand alias lookup with default quant must find the installed q8_0 model
    found = reg.find_model("smollm2:135m")
    assert found is not None
    assert found.id == rec.id
    assert found.quant == "q8_0"

    # Base alias lookup must also find it
    found_base = reg.find_model("smollm2")
    assert found_base is not None
    assert found_base.id == rec.id


