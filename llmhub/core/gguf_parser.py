"""High-performance standalone binary GGUF v2/v3 header and metadata parser.

Enables instant model inspection, chat template extraction, and parameter math
without loading gigabytes of weights into memory or initializing llama.cpp.
"""

import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# GGUF Magic: "GGUF" in ASCII (0x46554747 little-endian)
GGUF_MAGIC = 0x46554747

# GGUF Metadata Value Types
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

class GGUFParseError(Exception):
    """Raised when parsing GGUF binary format fails."""
    pass

class GGUFMetadata:
    """Holds parsed GGUF metadata with convenience accessors."""

    def __init__(self, raw_metadata: Dict[str, Any], tensor_count: int, version: int):
        self.raw = raw_metadata
        self.tensor_count = tensor_count
        self.version = version

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def architecture(self) -> str:
        return str(self.raw.get("general.architecture", "llama"))

    @property
    def model_name(self) -> str:
        return str(self.raw.get("general.name", self.raw.get("general.basename", "unknown")))

    @property
    def context_length(self) -> int:
        arch = self.architecture
        return int(self.raw.get(f"{arch}.context_length", self.raw.get("llama.context_length", 2048)))

    @property
    def embedding_length(self) -> int:
        arch = self.architecture
        return int(self.raw.get(f"{arch}.embedding_length", self.raw.get("llama.embedding_length", 4096)))

    @property
    def block_count(self) -> int:
        """Number of transformer layers."""
        arch = self.architecture
        return int(self.raw.get(f"{arch}.block_count", self.raw.get("llama.block_count", 32)))

    @property
    def head_count(self) -> int:
        """Number of query attention heads."""
        arch = self.architecture
        return int(self.raw.get(f"{arch}.attention.head_count", self.raw.get("llama.attention.head_count", 32)))

    @property
    def head_count_kv(self) -> int:
        """Number of key-value attention heads (for GQA/MQA)."""
        arch = self.architecture
        return int(self.raw.get(f"{arch}.attention.head_count_kv", self.raw.get("llama.attention.head_count_kv", self.head_count)))

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        arch = self.architecture
        key_len = self.raw.get(f"{arch}.attention.key_length", self.raw.get("llama.attention.key_length"))
        if key_len is not None:
            return int(key_len)
        if self.head_count > 0:
            return self.embedding_length // self.head_count
        return 128

    @property
    def chat_template(self) -> Optional[str]:
        """Jinja2 chat template embedded in model."""
        template = self.raw.get("tokenizer.chat_template")
        if isinstance(template, str) and template.strip():
            return template
        return None

    @property
    def bos_token_id(self) -> Optional[int]:
        val = self.raw.get("tokenizer.ggml.bos_token_id")
        return int(val) if val is not None else None

    @property
    def eos_token_id(self) -> Optional[int]:
        val = self.raw.get("tokenizer.ggml.eos_token_id")
        return int(val) if val is not None else None


class GGUFParser:
    """Reads GGUF binary format stream."""

    @classmethod
    def parse_file(cls, file_path: Union[str, Path], max_metadata_bytes: int = 128 * 1024 * 1024) -> GGUFMetadata:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {path}")

        with open(path, "rb") as f:
            return cls.parse_stream(f, max_metadata_bytes=max_metadata_bytes)

    @classmethod
    def parse_stream(cls, stream, max_metadata_bytes: int = 128 * 1024 * 1024) -> GGUFMetadata:
        magic_bytes = stream.read(4)
        if len(magic_bytes) < 4:
            raise GGUFParseError("Unexpected end of file reading magic header.")
        
        magic = struct.unpack("<I", magic_bytes)[0]
        if magic != GGUF_MAGIC:
            raise GGUFParseError(f"Invalid GGUF magic number: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})")

        version_bytes = stream.read(4)
        version = struct.unpack("<I", version_bytes)[0]
        if version not in (2, 3):
            raise GGUFParseError(f"Unsupported GGUF version: {version}. Expected v2 or v3.")

        tensor_count_bytes = stream.read(8)
        metadata_kv_count_bytes = stream.read(8)
        tensor_count = struct.unpack("<Q", tensor_count_bytes)[0]
        metadata_kv_count = struct.unpack("<Q", metadata_kv_count_bytes)[0]

        metadata: Dict[str, Any] = {}
        for _ in range(metadata_kv_count):
            key = cls._read_string(stream)
            val_type = struct.unpack("<I", stream.read(4))[0]
            val = cls._read_value(stream, val_type)
            metadata[key] = val

        return GGUFMetadata(metadata, tensor_count=tensor_count, version=version)

    @classmethod
    def _read_string(cls, stream, max_str_len: int = 16 * 1024 * 1024) -> str:
        len_bytes = stream.read(8)
        if len(len_bytes) < 8:
            raise GGUFParseError("Unexpected EOF reading string length.")
        str_len = struct.unpack("<Q", len_bytes)[0]
        if str_len > max_str_len:
            raise GGUFParseError(f"Corrupt or excessive string length encountered: {str_len} bytes.")
        raw_str = stream.read(str_len)
        return raw_str.decode("utf-8", errors="replace")

    @classmethod
    def _read_value(cls, stream, val_type: int) -> Any:
        if val_type == GGUF_TYPE_UINT8:
            return struct.unpack("<B", stream.read(1))[0]
        elif val_type == GGUF_TYPE_INT8:
            return struct.unpack("<b", stream.read(1))[0]
        elif val_type == GGUF_TYPE_UINT16:
            return struct.unpack("<H", stream.read(2))[0]
        elif val_type == GGUF_TYPE_INT16:
            return struct.unpack("<h", stream.read(2))[0]
        elif val_type == GGUF_TYPE_UINT32:
            return struct.unpack("<I", stream.read(4))[0]
        elif val_type == GGUF_TYPE_INT32:
            return struct.unpack("<i", stream.read(4))[0]
        elif val_type == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", stream.read(4))[0]
        elif val_type == GGUF_TYPE_BOOL:
            return bool(struct.unpack("<B", stream.read(1))[0])
        elif val_type == GGUF_TYPE_STRING:
            return cls._read_string(stream)
        elif val_type == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", stream.read(8))[0]
        elif val_type == GGUF_TYPE_INT64:
            return struct.unpack("<q", stream.read(8))[0]
        elif val_type == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", stream.read(8))[0]
        elif val_type == GGUF_TYPE_ARRAY:
            elem_type = struct.unpack("<I", stream.read(4))[0]
            arr_len = struct.unpack("<Q", stream.read(8))[0]
            
            # If array of tokens or big strings, avoid loading millions of elements if not needed
            if arr_len > 100000 and elem_type == GGUF_TYPE_STRING:
                # Token list can have 150k items; skip reading to preserve memory if tokenizer is huge
                # Read first 10 tokens and seek past remainder
                items = []
                for _ in range(min(10, arr_len)):
                    items.append(cls._read_string(stream))
                for _ in range(arr_len - len(items)):
                    s_len = struct.unpack("<Q", stream.read(8))[0]
                    stream.seek(s_len, 1)
                return items

            arr = []
            for _ in range(arr_len):
                arr.append(cls._read_value(stream, elem_type))
            return arr
        else:
            raise GGUFParseError(f"Unknown GGUF value type: {val_type}")
