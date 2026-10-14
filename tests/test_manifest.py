from pathlib import Path
from llmhub.core.gguf_parser import GGUFParser
from llmhub.hardware.manifest import ModelManifest
from tests.test_gguf_parser import create_synthetic_gguf

def test_manifest_lifecycle(tmp_path: Path):
    gguf_path = tmp_path / "model.gguf"
    manifest_path = tmp_path / "manifest.json"
    create_synthetic_gguf(gguf_path)

    metadata = GGUFParser.parse_file(gguf_path)
    manifest = ModelManifest.create_from_gguf(
        org="Qwen",
        name="Qwen2.5-0.5B-Instruct",
        quant="Q4_K_M",
        file_path=gguf_path,
        sha256="fake_sha256_hash",
        metadata=metadata,
        recommended_threads=4,
        recommended_ctx=2048
    )

    assert manifest.org == "Qwen"
    assert manifest.name == "Qwen2.5-0.5B-Instruct"
    assert manifest.quant == "q4_k_m"
    assert manifest.architecture == "llama"
    assert manifest.hardware.recommended_threads == 4
    assert manifest.hardware.recommended_ctx == 2048
    assert manifest.hardware.min_ram_mb > 0

    # Save to disk
    manifest.save(manifest_path)
    assert manifest_path.exists()

    # Load back and verify roundtrip
    loaded = ModelManifest.load(manifest_path)
    assert loaded.org == "Qwen"
    assert loaded.quant == "q4_k_m"
    assert loaded.hardware.recommended_threads == 4
    assert loaded.hardware.recommended_ctx == 2048
