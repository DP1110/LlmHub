import struct
from pathlib import Path
import pytest
from llmhub.core.gguf_parser import (
    GGUFParser,
    GGUF_MAGIC,
    GGUF_TYPE_UINT32,
    GGUF_TYPE_STRING,
    GGUF_TYPE_ARRAY,
    GGUFParseError,
)

def create_synthetic_gguf(path: Path) -> None:
    """Generate a valid binary GGUF v3 file containing metadata for testing."""
    with open(path, "wb") as f:
        # Magic (4 bytes uint32)
        f.write(struct.pack("<I", GGUF_MAGIC))
        # Version (4 bytes uint32)
        f.write(struct.pack("<I", 3))
        # Tensor count (8 bytes uint64)
        f.write(struct.pack("<Q", 145))
        # Metadata KV count (8 bytes uint64)
        f.write(struct.pack("<Q", 6))

        def write_str(s: str):
            data = s.encode("utf-8")
            f.write(struct.pack("<Q", len(data)))
            f.write(data)

        # KV 1: general.architecture (STRING)
        write_str("general.architecture")
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        write_str("llama")

        # KV 2: llama.context_length (UINT32)
        write_str("llama.context_length")
        f.write(struct.pack("<I", GGUF_TYPE_UINT32))
        f.write(struct.pack("<I", 4096))

        # KV 3: llama.block_count (UINT32)
        write_str("llama.block_count")
        f.write(struct.pack("<I", GGUF_TYPE_UINT32))
        f.write(struct.pack("<I", 28))

        # KV 4: llama.attention.head_count (UINT32)
        write_str("llama.attention.head_count")
        f.write(struct.pack("<I", GGUF_TYPE_UINT32))
        f.write(struct.pack("<I", 16))

        # KV 5: llama.attention.head_count_kv (UINT32)
        write_str("llama.attention.head_count_kv")
        f.write(struct.pack("<I", GGUF_TYPE_UINT32))
        f.write(struct.pack("<I", 4))

        # KV 6: tokenizer.chat_template (STRING)
        write_str("tokenizer.chat_template")
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        write_str("{{ messages[0]['content'] }}")

def test_parse_synthetic_gguf(tmp_path: Path):
    gguf_file = tmp_path / "test_model.gguf"
    create_synthetic_gguf(gguf_file)

    meta = GGUFParser.parse_file(gguf_file)
    assert meta.version == 3
    assert meta.tensor_count == 145
    assert meta.architecture == "llama"
    assert meta.context_length == 4096
    assert meta.block_count == 28
    assert meta.head_count == 16
    assert meta.head_count_kv == 4
    assert meta.chat_template == "{{ messages[0]['content'] }}"

def test_parse_invalid_magic(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt.gguf"
    corrupt_file.write_bytes(b"BADMAGICDATA")
    with pytest.raises(GGUFParseError, match="Invalid GGUF magic"):
        GGUFParser.parse_file(corrupt_file)

def test_parse_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        GGUFParser.parse_file(tmp_path / "does_not_exist.gguf")
