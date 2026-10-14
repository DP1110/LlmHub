"""Multi-Tenant Fair-Share Scheduling across Users and API Keys using Hierarchical DRR."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set

from llmhub.scheduler.session import SchedSession

@dataclass
class TenantRecord:
    """Tracks state, deficit credits, and resource consumption for a single tenant."""
    tenant_id: str
    weight: float = 1.0
    deficit: float = 0.0
    quantum: float = 16.0  # Base token credit allocation per round
    active_sessions: Set[str] = field(default_factory=set)
    total_tokens_consumed: int = 0
    total_sessions_completed: int = 0
    rate_limit_tokens_per_min: Optional[int] = None
    minute_window_start: float = field(default_factory=time.monotonic)
    tokens_in_current_minute: int = 0

    def replenish(self) -> None:
        """Replenish deficit credits based on tenant weight."""
        self.deficit += self.quantum * self.weight

    def record_usage(self, tokens: int) -> None:
        """Record token consumption and check minute rate limit window."""
        self.deficit = max(0.0, self.deficit - tokens)
        self.total_tokens_consumed += tokens

        now = time.monotonic()
        if now - self.minute_window_start >= 60.0:
            self.minute_window_start = now
            self.tokens_in_current_minute = 0
        self.tokens_in_current_minute += tokens

    @property
    def is_rate_limited(self) -> bool:
        if self.rate_limit_tokens_per_min is None:
            return False
        now = time.monotonic()
        if now - self.minute_window_start >= 60.0:
            return False
        return self.tokens_in_current_minute >= self.rate_limit_tokens_per_min

class TenantFairShareManager:
    """Manages tenant accounts and enforces fair-share turns across multiple tenants."""

    def __init__(self, default_tenant_quantum: float = 16.0):
        self.default_tenant_quantum = default_tenant_quantum
        self.tenants: Dict[str, TenantRecord] = {}
        self.tenant_rr_pointer: int = 0

    def get_or_create_tenant(self, tenant_id: str, weight: float = 1.0) -> TenantRecord:
        """Retrieve existing tenant record or initialize new one with default weights."""
        tid = tenant_id or "default"
        if tid not in self.tenants:
            self.tenants[tid] = TenantRecord(
                tenant_id=tid,
                weight=max(0.01, float(weight)),
                quantum=self.default_tenant_quantum,
            )
        return self.tenants[tid]

    def register_session_admission(self, session: SchedSession, tenant_id: Optional[str] = None) -> None:
        """Bind session to a tenant and update active session tracking and weight."""
        tid = tenant_id or getattr(session, "tenant_id", "default") or "default"
        session.tenant_id = tid
        weight = max(0.01, float(getattr(session, "weight", 1.0) or 1.0))
        tenant = self.get_or_create_tenant(tid, weight=weight)
        tenant.weight = weight
        tenant.active_sessions.add(session.session_id)

    def register_session_completion(self, session: SchedSession) -> None:
        """Unbind completed session from tenant record."""
        tid = getattr(session, "tenant_id", "default") or "default"
        if tid in self.tenants:
            tenant = self.tenants[tid]
            tenant.active_sessions.discard(session.session_id)
            tenant.total_sessions_completed += 1

    def select_next_tenant_session(self, runnable_sessions: List[SchedSession]) -> Optional[SchedSession]:
        """Select next eligible session according to Hierarchical Deficit Round-Robin across tenants."""
        if not runnable_sessions:
            return None

        # Group sessions by tenant
        tenant_map: Dict[str, List[SchedSession]] = {}
        for s in runnable_sessions:
            if not s.is_done and not s.cancelled:
                tid = getattr(s, "tenant_id", "default") or "default"
                tenant_map.setdefault(tid, []).append(s)

        if not tenant_map:
            return None

        # Filter out rate-limited tenants to prevent rate-limit bypass
        eligible_tids = [
            tid for tid in tenant_map.keys()
            if not self.get_or_create_tenant(tid).is_rate_limited
        ]
        if not eligible_tids:
            return None

        tids_count = len(eligible_tids)
        rotated_tids = [
            eligible_tids[(self.tenant_rr_pointer + i) % tids_count]
            for i in range(tids_count)
        ]

        # Check if any tenant has sufficient deficit to run
        for tid in rotated_tids:
            tenant = self.get_or_create_tenant(tid)
            if tenant.deficit >= tenant.quantum:
                self.tenant_rr_pointer = (self.tenant_rr_pointer + 1) % tids_count
                return tenant_map[tid][0]

        # Replenish deficit credits across active eligible tenants
        for _ in range(10):
            for tid in eligible_tids:
                tenant = self.get_or_create_tenant(tid)
                tenant.replenish()
            for tid in rotated_tids:
                tenant = self.get_or_create_tenant(tid)
                if tenant.deficit >= tenant.quantum:
                    self.tenant_rr_pointer = (self.tenant_rr_pointer + 1) % tids_count
                    return tenant_map[tid][0]

        # Fallback to first non-rate-limited runnable session
        self.tenant_rr_pointer = (self.tenant_rr_pointer + 1) % tids_count
        return tenant_map[eligible_tids[0]][0]

    def record_execution(self, tenant_id: str, tokens: int) -> None:
        """Deduct deficit and record usage for the executed tenant."""
        tid = tenant_id or "default"
        if tid in self.tenants:
            self.tenants[tid].record_usage(tokens)

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics across all tracked tenants."""
        total_tokens = sum(t.total_tokens_consumed for t in self.tenants.values())
        tenants_data = {}
        for tid, t in self.tenants.items():
            share_pct = (t.total_tokens_consumed / total_tokens * 100.0) if total_tokens > 0 else 0.0
            tenants_data[tid] = {
                "weight": t.weight,
                "deficit": round(t.deficit, 1),
                "active_sessions": len(t.active_sessions),
                "tokens_consumed": t.total_tokens_consumed,
                "share_pct": round(share_pct, 1),
                "is_rate_limited": t.is_rate_limited,
            }
        return {
            "total_tenants": len(self.tenants),
            "total_tokens_served": total_tokens,
            "tenants": tenants_data,
        }
