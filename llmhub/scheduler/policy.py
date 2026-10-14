"""CPU Threading Strategies and Deficit Round-Robin Policy for LLMHub Scheduler."""

from enum import Enum
import math
from typing import Dict, List, Optional, Set

from llmhub.hardware.detector import HardwareDetector, HardwareProfile
from llmhub.scheduler.session import SchedSession, SessionPhase

class ThreadingMode(str, Enum):
    TIME_SLICE = "time-slice"
    PARTITION = "partition"
    HYBRID = "hybrid"

class ThreadingStrategy:
    """Calculates and manages CPU core allocations per model context."""

    def __init__(self, mode: ThreadingMode = ThreadingMode.HYBRID, profile: Optional[HardwareProfile] = None):
        self.mode = mode
        self.profile = profile or HardwareDetector.get_profile()
        self.total_physical_cores = max(1, self.profile.cpu.physical_cores)

    def calculate_threads(self, active_model_tags: Set[str]) -> Dict[str, int]:
        """Calculate thread allocation for each active model according to active mode."""
        active_count = len(active_model_tags)
        if active_count == 0:
            return {}

        allocations: Dict[str, int] = {}

        if self.mode == ThreadingMode.TIME_SLICE:
            # Mode (a): Every model gets all physical cores during its time slice
            for tag in active_model_tags:
                allocations[tag] = self.total_physical_cores

        elif self.mode == ThreadingMode.PARTITION:
            # Mode (b): Statically partition physical cores evenly among active models
            threads_per_model = max(1, self.total_physical_cores // active_count)
            for tag in active_model_tags:
                allocations[tag] = threads_per_model

        elif self.mode == ThreadingMode.HYBRID:
            # Mode (c): Dynamic hybrid partitioning
            if active_count == 1:
                for tag in active_model_tags:
                    allocations[tag] = self.total_physical_cores
            else:
                if self.profile.cpu.is_hybrid_big_little and self.profile.cpu.performance_cores > 0:
                    # Allocate performance cores + proportional efficiency cores
                    total_p = self.profile.cpu.performance_cores
                    total_e = self.profile.cpu.efficiency_cores
                    p_per_model = max(1, total_p // active_count)
                    e_per_model = max(0, total_e // active_count)
                    for tag in active_model_tags:
                        allocations[tag] = max(1, p_per_model + e_per_model)
                else:
                    threads_per_model = max(1, self.total_physical_cores // active_count)
                    for tag in active_model_tags:
                        allocations[tag] = threads_per_model

        return allocations

class DeficitRoundRobinPolicy:
    """Starvation-free Deficit Round-Robin (DRR) scheduler with priority weighting."""

    def __init__(self, default_quantum: int = 4):
        self.default_quantum = default_quantum

    def replenish_credits(self, session: SchedSession) -> None:
        """Add credits to session deficit counter proportional to weight and priority."""
        # Priority scaling: base normal priority is 5.
        priority_multiplier = max(0.2, session.priority / 5.0)
        credits = self.default_quantum * max(0.01, session.weight) * priority_multiplier
        session.deficit += credits

    def select_next_session(self, queue: List[SchedSession]) -> Optional[SchedSession]:
        """Find the next runnable session eligible for quantum execution using DRR."""
        if not queue:
            return None

        # Filter runnable sessions
        runnable = [s for s in queue if not s.is_done and not s.cancelled]
        if not runnable:
            return None

        # Check if any session already has enough deficit to run
        for sess in runnable:
            if sess.deficit >= sess.quantum_tokens:
                return sess

        # Replenish in synchronized rounds across all runnable sessions with bounded iteration guard
        max_rounds = 100
        for _ in range(max_rounds):
            for sess in runnable:
                self.replenish_credits(sess)
            for sess in runnable:
                if sess.deficit >= sess.quantum_tokens:
                    return sess

        # Fallback to prevent hang
        return runnable[0]


    def compute_jains_fairness_index(self, throughputs: List[float], weights: Optional[List[float]] = None) -> float:
        """Compute Jain's Fairness Index across session throughputs.
        
        Formula:
            y_i = throughput_i / weight_i
            J = (sum(y_i))^2 / (n * sum(y_i^2))
        """
        n = len(throughputs)
        if n == 0:
            return 1.0

        if weights is None:
            weights = [1.0] * n

        normalized = []
        for t, w in zip(throughputs, weights):
            w_safe = max(1e-6, w)
            normalized.append(t / w_safe)

        sum_y = sum(normalized)
        sum_sq_y = sum(y * y for y in normalized)

        if sum_sq_y == 0.0:
            return 1.0

        return (sum_y * sum_y) / (n * sum_sq_y)
