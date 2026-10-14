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
