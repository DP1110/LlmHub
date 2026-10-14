"""Edge Model Pipeline Partitioning and Distributed Bandwidth Coordinator.

Calculates optimal multi-device transformer layer partitioning across edge nodes
(e.g., PC + Android Phone) and computes intermediate activation transfer physics.
"""

from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Optional

from llmhub.cluster.node import NodeInfo

logger = logging.getLogger("llmhub.cluster.pipeline")


@dataclass
class PipelineStage:
    """A partition of sequential transformer layers assigned to a specific cluster node."""
    stage_id: int
    node_id: str
    node_name: str
    layer_start: int
    layer_end: int  # exclusive
    layer_count: int
    estimated_memory_mb: float
    has_embeddings: bool = False
    has_lm_head: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelinePlan:
    """Full execution layout and physics analysis for distributed pipeline parallel inference."""
    model_tag: str
    total_layers: int
    hidden_dim: int
    stages: List[PipelineStage]
    activation_bytes_per_token: int
    estimated_transfer_ms_per_step: float
    is_feasible: bool
    feasibility_notes: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stages"] = [s.to_dict() for s in self.stages]
        return d


class PipelineCoordinator:
    """Plans multi-node layer sharding and evaluates network-latency bounds."""

    @staticmethod
    def calculate_activation_bytes(hidden_dim: int, element_size: int = 2) -> int:
        """Compute the byte size of an intermediate hidden state vector transferred between stages."""
        # 1 token vector = hidden_dim * element_size (FP16/BF16 is 2 bytes)
        return hidden_dim * element_size

    @classmethod
    def plan_pipeline(
        cls,
        model_tag: str,
        total_layers: int,
        hidden_dim: int,
        total_weights_mb: float,
        nodes: List[NodeInfo],
        network_latency_ms: float = 3.0,
        network_bandwidth_mbps: float = 100.0,
    ) -> PipelinePlan:
        """Compute optimal layer allocation across available cluster nodes based on free RAM."""
        if not nodes:
            return PipelinePlan(
                model_tag=model_tag,
                total_layers=total_layers,
                hidden_dim=hidden_dim,
                stages=[],
                activation_bytes_per_token=cls.calculate_activation_bytes(hidden_dim),
                estimated_transfer_ms_per_step=0.0,
                is_feasible=False,
                feasibility_notes="No available cluster nodes provided.",
            )

        if total_layers < 1:
            return PipelinePlan(
                model_tag=model_tag,
                total_layers=total_layers,
                hidden_dim=hidden_dim,
                stages=[],
                activation_bytes_per_token=cls.calculate_activation_bytes(hidden_dim),
                estimated_transfer_ms_per_step=0.0,
                is_feasible=False,
                feasibility_notes=f"Total layers must be at least 1 (got {total_layers}).",
            )

        # Filter nodes with at least 500 MB free RAM
        valid_nodes = [n for n in nodes if n.is_available and n.free_ram_mb >= 500.0]
        if not valid_nodes:
            return PipelinePlan(
                model_tag=model_tag,
                total_layers=total_layers,
                hidden_dim=hidden_dim,
                stages=[],
                activation_bytes_per_token=cls.calculate_activation_bytes(hidden_dim),
                estimated_transfer_ms_per_step=0.0,
                is_feasible=False,
                feasibility_notes="No nodes with sufficient free RAM (minimum 500 MB required).",
            )

        # A pipeline partition requires at least 1 layer per participating node
        if len(valid_nodes) > total_layers:
            valid_nodes = valid_nodes[:total_layers]

        total_cluster_free_ram = sum(n.free_ram_mb for n in valid_nodes)
        if total_cluster_free_ram < total_weights_mb:
            return PipelinePlan(
                model_tag=model_tag,
                total_layers=total_layers,
                hidden_dim=hidden_dim,
                stages=[],
                activation_bytes_per_token=cls.calculate_activation_bytes(hidden_dim),
                estimated_transfer_ms_per_step=0.0,
                is_feasible=False,
                feasibility_notes=(
                    f"Aggregate cluster free RAM ({total_cluster_free_ram:.1f} MB) is smaller than "
                    f"model weights ({total_weights_mb:.1f} MB)."
                ),
            )

        # Compute proportional layers per node
        stages: List[PipelineStage] = []
        current_layer = 0
        weight_mb_per_layer = total_weights_mb / max(1, total_layers)

        for i, node in enumerate(valid_nodes):
            is_last = (i == len(valid_nodes) - 1)
            remaining_nodes = len(valid_nodes) - i - 1
            remaining_layers = total_layers - current_layer
            if is_last:
                n_layers = remaining_layers
            else:
                ratio = node.free_ram_mb / total_cluster_free_ram
                n_layers = max(1, round(total_layers * ratio))
                # Ensure each remaining node gets at least 1 layer
                n_layers = max(1, min(n_layers, remaining_layers - remaining_nodes))

            stage_mem = n_layers * weight_mb_per_layer
            # Add embedding overhead to first stage, LM head overhead to last stage
            if i == 0:
                stage_mem += (hidden_dim * 32000 * 2) / (1024 * 1024)  # ~100MB approx
            if is_last:
                stage_mem += (hidden_dim * 32000 * 2) / (1024 * 1024)

            stages.append(
                PipelineStage(
                    stage_id=i,
                    node_id=node.node_id,
                    node_name=node.name,
                    layer_start=current_layer,
                    layer_end=current_layer + n_layers,
                    layer_count=n_layers,
                    estimated_memory_mb=round(stage_mem, 1),
                    has_embeddings=(i == 0),
                    has_lm_head=is_last,
                )
            )
            current_layer += n_layers

        # Network transfer calculations
        act_bytes = cls.calculate_activation_bytes(hidden_dim)
        # Transfer time across stages: (N_stages - 1) * (latency + bytes / bandwidth)
        num_transitions = max(0, len(stages) - 1)
        wire_bytes_per_sec = (network_bandwidth_mbps * 1e6) / 8.0
        wire_transfer_s = act_bytes / wire_bytes_per_sec if wire_bytes_per_sec > 0 else 0.0
        total_transfer_ms = num_transitions * (network_latency_ms + (wire_transfer_s * 1000.0))

        return PipelinePlan(
            model_tag=model_tag,
            total_layers=total_layers,
            hidden_dim=hidden_dim,
            stages=stages,
            activation_bytes_per_token=act_bytes,
            estimated_transfer_ms_per_step=round(total_transfer_ms, 2),
            is_feasible=True,
            feasibility_notes=(
                f"Pipeline partitioned across {len(stages)} nodes. "
                f"Transfer payload per step: {act_bytes / 1024.0:.1f} KB. "
                f"Network overhead per token: {total_transfer_ms:.2f} ms."
            ),
        )
