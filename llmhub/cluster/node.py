"""Cluster Node representations and state machine for distributed LLMHub."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional


class NodeStatus(str, Enum):
    """Lifecycle state of a cluster member node."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    DRAINING = "draining"
    OFFLINE = "offline"


class NodeRole(str, Enum):
    """Functional role of the node in the cluster."""
    WORKER = "worker"
    COORDINATOR = "coordinator"
    HYBRID = "hybrid"


@dataclass
class NodeInfo:
    """Detailed topology, health, and resident model metadata for a cluster node."""
    node_id: str
    name: str = ""
    host: str = "127.0.0.1"
    port: int = 11434
    api_url: str = ""
    role: NodeRole = NodeRole.HYBRID
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    total_ram_mb: float = 0.0
    free_ram_mb: float = 0.0
    active_sessions: int = 0
    resident_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    is_termux: bool = False
    cluster_secret: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.api_url:
            self.api_url = f"http://{self.host}:{self.port}"
        if not self.name:
            self.name = self.node_id

    @property
    def is_available(self) -> bool:
        """Check if node can accept new inference requests."""
        return self.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED)

    def load_score(self, model_tag: Optional[str] = None) -> float:
        """Compute relative load score for routing decisions (lower is better).
        
        Takes into account active sessions, free RAM, latency, and model residency.
        """
        # Base score starts with active sessions
        score = float(self.active_sessions * 100.0)

        # Penalize low free RAM (less RAM available = higher penalty)
        if self.free_ram_mb > 0:
            score += max(0.0, 5000.0 / self.free_ram_mb)
        else:
            score += 10000.0

        # Add network latency penalty (1 ms latency ~ 2 score points)
        score += self.latency_ms * 2.0

        # High discount if model is already resident in RAM on this node
        if model_tag and model_tag in self.resident_models:
            score -= 200.0

        # Slight discount if node is healthy vs degraded
        if self.status == NodeStatus.DEGRADED:
            score += 150.0

        return max(0.0, score)

    def update_heartbeat(
        self,
        free_ram_mb: Optional[float] = None,
        active_sessions: Optional[int] = None,
        resident_models: Optional[Dict[str, Dict[str, Any]]] = None,
        status: Optional[NodeStatus] = None,
    ) -> None:
        """Update node dynamic metrics from a heartbeat."""
        self.last_heartbeat = time.time()
        if free_ram_mb is not None:
            self.free_ram_mb = free_ram_mb
        if active_sessions is not None:
            self.active_sessions = active_sessions
        if resident_models is not None:
            self.resident_models = resident_models
        if status is not None:
            self.status = status
        elif self.status == NodeStatus.UNREACHABLE:
            self.status = NodeStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """Convert NodeInfo to JSON-serializable dictionary."""
        d = asdict(self)
        d["role"] = self.role.value if isinstance(self.role, NodeRole) else str(self.role)
        d["status"] = self.status.value if isinstance(self.status, NodeStatus) else str(self.status)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        """Reconstruct NodeInfo from dictionary."""
        d = dict(data)
        if "role" in d and isinstance(d["role"], str):
            try:
                d["role"] = NodeRole(d["role"])
            except ValueError:
                d["role"] = NodeRole.HYBRID
        if "status" in d and isinstance(d["status"], str):
            try:
                d["status"] = NodeStatus(d["status"])
            except ValueError:
                d["status"] = NodeStatus.HEALTHY
        # Safely filter only valid dataclass fields
        valid_fields = getattr(cls, "__dataclass_fields__", {})
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
