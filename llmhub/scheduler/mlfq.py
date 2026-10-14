"""Multilevel Feedback Queue (MLFQ) Priority Scheduler for LLMHub.

Rules:
1. Priority Levels:
   - Q0 (Highest): Interactive/short completions, quantum allotment T0 (e.g. 4 tokens).
   - Q1 (Medium): Medium completions, quantum allotment T1 (e.g. 8 tokens).
   - Q2 (Lowest): Batch / long-running completions, quantum allotment T2 (e.g. 16 tokens).
2. Dynamic Demotion:
   - Sessions arrive at Q0.
   - When a session exhausts its queue budget (e.g. 16 tokens in Q0, 64 tokens in Q1),
     it demotes to level k+1.
3. Starvation-Free Global Periodic Boost:
   - Every boost_interval_seconds (default 10s), all runnable sessions are reset to Q0,
     preventing starvation of low-priority batch requests.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Deque, Dict, List, Optional, Set

from llmhub.scheduler.session import SchedSession, SessionPhase

@dataclass
class MLFQLevelConfig:
    """Configuration for an individual priority queue level."""
    level: int
    name: str
    quantum_tokens: int
    token_budget: int  # Tokens a session can consume at this level before demotion (float('inf') for lowest)

class MultilevelFeedbackQueue:
    """Manages MLFQ priority queues, demotion on token consumption, and periodic boosting."""

    def __init__(
        self,
        boost_interval_seconds: float = 10.0,
        enable_mlfq: bool = True,
    ):
        self.boost_interval_seconds = max(0.01, float(boost_interval_seconds))
        self.enable_mlfq = enable_mlfq

        # Define 3 MLFQ levels: Q0 (interactive), Q1 (medium), Q2 (batch)
        self.level_configs: List[MLFQLevelConfig] = [
            MLFQLevelConfig(level=0, name="Q0-Interactive", quantum_tokens=4, token_budget=16),
            MLFQLevelConfig(level=1, name="Q1-Medium", quantum_tokens=8, token_budget=64),
            MLFQLevelConfig(level=2, name="Q2-Batch", quantum_tokens=16, token_budget=1_000_000_000),
        ]

        # Priority queues: level -> Deque[session_id]
        self.queues: Dict[int, Deque[SchedSession]] = {
            lvl.level: deque() for lvl in self.level_configs
        }

        # Tracking per session: session_id -> {level, tokens_consumed_at_level}
        self.session_levels: Dict[str, int] = {}
        self.session_tokens_at_level: Dict[str, int] = {}

        self.last_boost_time: float = time.monotonic()
        self.total_boosts: int = 0
        self.total_demotions: int = 0

    def enqueue_session(self, session: SchedSession) -> None:
        """Enqueue a new or re-admitted session at highest priority (Q0)."""
        sid = session.session_id
        current_lvl = self.session_levels.get(sid, 0)
        if not self.enable_mlfq:
            current_lvl = 0

        self.session_levels[sid] = current_lvl
        if sid not in self.session_tokens_at_level:
            self.session_tokens_at_level[sid] = 0

        # Remove from any queue first to prevent duplicates
        for q in self.queues.values():
            try:
                while session in q:
                    q.remove(session)
            except ValueError:
                pass

        # Enqueue at target level
        self.queues[current_lvl].append(session)
        # Update session quantum to match level quantum
        session.quantum_tokens = self.level_configs[current_lvl].quantum_tokens
        session.mlfq_level = current_lvl

    def remove_session(self, session: SchedSession) -> None:
        """Remove a completed, cancelled, or timed-out session from all queues."""
        for q in self.queues.values():
            try:
                while session in q:
                    q.remove(session)
            except ValueError:
                pass
        self.session_levels.pop(session.session_id, None)
        self.session_tokens_at_level.pop(session.session_id, None)

    def record_quantum_consumption(self, session: SchedSession, tokens_generated: int) -> bool:
        """Record generated tokens and check for demotion to lower priority level.
        
        Returns True if session was demoted, False otherwise.
        """
        if not self.enable_mlfq:
            return False

        sid = session.session_id
        current_lvl = self.session_levels.get(sid, 0)
        self.session_tokens_at_level[sid] = self.session_tokens_at_level.get(sid, 0) + tokens_generated

        cfg = self.level_configs[min(current_lvl, len(self.level_configs) - 1)]
        if current_lvl < len(self.level_configs) - 1 and self.session_tokens_at_level[sid] >= cfg.token_budget:
            # Demote session to next lower level
            new_lvl = current_lvl + 1
            self.session_levels[sid] = new_lvl
            self.session_tokens_at_level[sid] = 0
            self.total_demotions += 1

            # Move in queue
            for q in self.queues.values():
                try:
                    while session in q:
                        q.remove(session)
                except ValueError:
                    pass
            self.queues[new_lvl].append(session)
            session.quantum_tokens = self.level_configs[new_lvl].quantum_tokens
            session.mlfq_level = new_lvl
            return True

        return False

    def check_and_apply_periodic_boost(self, now: Optional[float] = None) -> bool:
        """Perform starvation-free priority boost: move all sessions to Q0 if timer expired."""
        if not self.enable_mlfq:
            return False

        t = now or time.monotonic()
        if (t - self.last_boost_time) >= self.boost_interval_seconds:
            # Gather all non-done sessions from all queues
            all_sessions: List[SchedSession] = []
            for lvl in range(len(self.level_configs)):
                q = self.queues[lvl]
                while q:
                    s = q.popleft()
                    if not s.is_done and not s.cancelled:
                        all_sessions.append(s)

            # Boost all to Q0
            for s in all_sessions:
                self.session_levels[s.session_id] = 0
                self.session_tokens_at_level[s.session_id] = 0
                s.quantum_tokens = self.level_configs[0].quantum_tokens
                s.mlfq_level = 0
                self.queues[0].append(s)

            self.last_boost_time = t
            self.total_boosts += 1
            return True

        return False

    def select_next_runnable(self, filter_model_tag: Optional[str] = None) -> Optional[SchedSession]:
        """Select next eligible session: strictly prioritizes Q0 over Q1 over Q2.
        
        Within each queue level, sessions rotate round-robin.
        """
        self.check_and_apply_periodic_boost()

        for lvl in range(len(self.level_configs)):
            q = self.queues[lvl]
            if not q:
                continue

            # Look for first eligible runnable session
            for s in list(q):
                if s.is_done or s.cancelled:
                    q.remove(s)
                    self.session_levels.pop(s.session_id, None)
                    self.session_tokens_at_level.pop(s.session_id, None)
                    continue

                if filter_model_tag is not None and s.model_tag != filter_model_tag:
                    continue

                return s

        return None

    def rotate_session_in_level(self, session: SchedSession) -> None:
        """Rotate a still-active session to the back of its current priority queue."""
        sid = session.session_id
        lvl = self.session_levels.get(sid, 0)
        q = self.queues[lvl]
        if session in q:
            q.remove(session)
            q.append(session)

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics and queue distributions for MLFQ."""
        queue_counts = {
            f"q{lvl}_{cfg.name}": len([s for s in self.queues[lvl] if not s.is_done])
            for lvl, cfg in enumerate(self.level_configs)
        }
        return {
            "enabled": self.enable_mlfq,
            "boost_interval_seconds": self.boost_interval_seconds,
            "total_boosts": self.total_boosts,
            "total_demotions": self.total_demotions,
            "queue_depths": queue_counts,
            "total_active_sessions": sum(queue_counts.values()),
        }
