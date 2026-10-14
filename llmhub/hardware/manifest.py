"""Standardized Model Manifest Schema, Validator, and Serialization.

Manifest path: ~/.llmhub/models/<org>/<name>/<quant>/manifest.json
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from llmhub.core.gguf_parser import GGUFMetadata
from llmhub.utils.math_calc import compute_memory_breakdown, estimate_decode_throughput

@dataclass
class HardwareRequirements:
    min_ram_mb: float
    recommended_ram_mb: float
    recommended_threads: int
    recommended_ctx: int
    estimated_decode_tok_s: float
    bottleneck: str

@dataclass
class ModelManifest:
    repo: str
    org: str
    name: str
    quant: str
    size_bytes: int
    sha256: str
    architecture: str
    n_layers: int
    kv_heads: int
    head_dim: int
    ctx_max: int
    has_chat_template: bool
    template: Optional[str]
    created_at: str
    hardware: HardwareRequirements
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> "ModelManifest":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        hw_data = data.get("hardware", {})
        hw = HardwareRequirements(
            min_ram_mb=hw_data.get("min_ram_mb", 0.0),
            recommended_ram_mb=hw_data.get("recommended_ram_mb", 0.0),
            recommended_threads=hw_data.get("recommended_threads", 4),
            recommended_ctx=hw_data.get("recommended_ctx", 2048),
            estimated_decode_tok_s=hw_data.get("estimated_decode_tok_s", 0.0),
            bottleneck=hw_data.get("bottleneck", "Memory-Bandwidth Bound")
        )

        return cls(
            repo=data.get("repo", "unknown/unknown"),
            org=data.get("org", "unknown"),
            name=data.get("name", "unknown"),
            quant=data.get("quant", "q4_k_m"),
            size_bytes=int(data.get("size_bytes", 0)),
            sha256=data.get("sha256", ""),
            architecture=data.get("architecture", "llama"),
            n_layers=int(data.get("n_layers", 32)),
            kv_heads=int(data.get("kv_heads", 8)),
            head_dim=int(data.get("head_dim", 128)),
            ctx_max=int(data.get("ctx_max", 2048)),
            has_chat_template=bool(data.get("has_chat_template", False)),
            template=data.get("template"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            hardware=hw,
            extra_metadata=data.get("extra_metadata", {})
        )

    @classmethod
    def create_from_gguf(
        cls,
        org: str,
        name: str,
        quant: str,
        file_path: Path,
        sha256: str,
        metadata: GGUFMetadata,
        recommended_threads: int = 4,
        recommended_ctx: int = 2048
    ) -> "ModelManifest":
        file_size = file_path.stat().st_size
        
        # Calculate min RAM (weights + 512 ctx)
        min_mem = compute_memory_breakdown(
            weights_bytes=file_size,
            n_layers=metadata.block_count,
            n_kv_heads=metadata.head_count_kv,
            head_dim=metadata.head_dim,
            n_ctx=512
        )
        # Recommended RAM (weights + recommended_ctx)
        rec_mem = compute_memory_breakdown(
            weights_bytes=file_size,
            n_layers=metadata.block_count,
            n_kv_heads=metadata.head_count_kv,
            head_dim=metadata.head_dim,
            n_ctx=recommended_ctx
        )
        bandwidth = estimate_decode_throughput(file_size)

        hw = HardwareRequirements(
            min_ram_mb=min_mem.total_mb,
            recommended_ram_mb=rec_mem.total_mb,
            recommended_threads=recommended_threads,
            recommended_ctx=recommended_ctx,
            estimated_decode_tok_s=bandwidth.theoretical_max_tok_s,
            bottleneck=bandwidth.bottleneck_type
        )

        return cls(
            repo=f"{org}/{name}",
            org=org,
            name=name,
            quant=quant.lower(),
            size_bytes=file_size,
            sha256=sha256,
            architecture=metadata.architecture,
            n_layers=metadata.block_count,
            kv_heads=metadata.head_count_kv,
            head_dim=metadata.head_dim,
            ctx_max=metadata.context_length,
            has_chat_template=metadata.chat_template is not None,
            template=metadata.chat_template,
            created_at=datetime.now(timezone.utc).isoformat(),
            hardware=hw
        )
