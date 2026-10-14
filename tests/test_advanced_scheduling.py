"""Tests for Phase 7 Advanced Scheduling: MLFQ, Predictive SLO, Fair-Share, and Speculative Decoding."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
import pytest
from fastapi.testclient import TestClient

from llmhub.scheduler.fair_share import TenantFairShareManager, TenantRecord
from llmhub.scheduler.mlfq import MultilevelFeedbackQueue
from llmhub.scheduler.policy import ThreadingMode
from llmhub.scheduler.scheduler import TokenRoundRobinScheduler
from llmhub.scheduler.session import SchedSession, SessionPhase
from llmhub.scheduler.slo import SLOAdmissionController, SLOTarget, SLOViolationError
from llmhub.server.app import create_app, state
from llmhub.server.backend import Backend, DecodeResult, SessionState
from llmhub.server.model_manager import LoadedModel
from llmhub.speculative.engine import SpeculativeEngine, SpeculativeStepResult

class MockBackend(Backend):
    """Deterministic mock backend for testing advanced scheduling and speculative decoding."""

    def __init__(self, token_sequence: Optional[List[int]] = None, name: str = "mock"):
        super().__init__()
        self.name = name
        self.token_sequence = token_sequence or [100, 101, 102, 103, 104, 105, 106, 107]
        self.sessions = {}
        self.decode_call_count = 0
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_path: str, **kwargs) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False
        self.sessions.clear()

    def get_info(self) -> Dict[str, Any]:
        return {"status": "loaded", "name": self.name}

    def new_session(self, session_id: str, prompt: str, **kwargs) -> SessionState:
        tokens = [ord(c) for c in prompt[:16]] or [1]
        sess = SessionState(
            session_id=session_id,
            model_tag=self.name,
            prompt=prompt,
            prompt_tokens=tokens,
            max_tokens=kwargs.get("max_tokens", 128),
        )
        self.sessions[session_id] = sess
        return sess

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def prefill_chunk(self, session_id: str, chunk_size: int) -> tuple[int, bool]:
        sess = self.sessions[session_id]
        sess.prefill_offset = len(sess.prompt_tokens)
        return len(sess.prompt_tokens), True

    def decode_step(self, session_id: str, n_tokens: int) -> DecodeResult:
        self.decode_call_count += 1
        sess = self.sessions[session_id]
        generated = []
        deltas = []
        for _ in range(n_tokens):
            idx = len(sess.generated_tokens) % len(self.token_sequence)
            tok = self.token_sequence[idx]
            generated.append(tok)
            deltas.append(f" {tok}")
            sess.generated_tokens.append(tok)
            if len(sess.generated_tokens) >= sess.max_tokens:
                break

        done = len(sess.generated_tokens) >= sess.max_tokens
        return DecodeResult(
            session_id=session_id,
            tokens=generated,
            text_deltas=deltas,
            done=done,
            stop_reason="length" if done else None,
        )

# ============================================================================
# 1. MLFQ (Multilevel Feedback Queue) Unit Tests
# ============================================================================

def test_mlfq_priority_levels_and_initial_enqueue():
    mlfq = MultilevelFeedbackQueue(boost_interval_seconds=10.0, enable_mlfq=True)
    assert len(mlfq.level_configs) == 3
    assert mlfq.level_configs[0].quantum_tokens == 4
    assert mlfq.level_configs[1].quantum_tokens == 8
    assert mlfq.level_configs[2].quantum_tokens == 16

    s1 = SchedSession(session_id="s1", model_tag="test", prompt="hello", max_tokens=64)
    mlfq.enqueue_session(s1)

    assert mlfq.session_levels["s1"] == 0
    assert s1.quantum_tokens == 4
    assert len(mlfq.queues[0]) == 1
    assert len(mlfq.queues[1]) == 0
    assert len(mlfq.queues[2]) == 0

def test_mlfq_dynamic_demotion_on_token_accumulation():
    mlfq = MultilevelFeedbackQueue(boost_interval_seconds=10.0, enable_mlfq=True)
    s1 = SchedSession(session_id="s1", model_tag="test", prompt="hello", max_tokens=128)
    mlfq.enqueue_session(s1)

    # In Q0, budget is 16 tokens. Consume 12 tokens: no demotion
    demoted = mlfq.record_quantum_consumption(s1, 12)
    assert not demoted
    assert mlfq.session_levels["s1"] == 0
    assert s1.quantum_tokens == 4

    # Consume 4 more tokens (total 16): demotes to Q1
    demoted = mlfq.record_quantum_consumption(s1, 4)
    assert demoted
    assert mlfq.session_levels["s1"] == 1
    assert s1.quantum_tokens == 8
    assert len(mlfq.queues[0]) == 0
    assert len(mlfq.queues[1]) == 1

    # In Q1, budget is 64 tokens. Consume 64 tokens: demotes to Q2
    demoted = mlfq.record_quantum_consumption(s1, 64)
    assert demoted
    assert mlfq.session_levels["s1"] == 2
    assert s1.quantum_tokens == 16
    assert len(mlfq.queues[1]) == 0
    assert len(mlfq.queues[2]) == 1

def test_mlfq_starvation_free_periodic_boost():
    mlfq = MultilevelFeedbackQueue(boost_interval_seconds=0.1, enable_mlfq=True)
    s1 = SchedSession(session_id="s1", model_tag="test", prompt="hello", max_tokens=128)
    mlfq.enqueue_session(s1)

    # Force demotion to Q2
    mlfq.record_quantum_consumption(s1, 16)
    mlfq.record_quantum_consumption(s1, 64)
    assert mlfq.session_levels["s1"] == 2
    assert len(mlfq.queues[2]) == 1

    # Wait for boost interval to expire
    time.sleep(0.12)
    boosted = mlfq.check_and_apply_periodic_boost()
    assert boosted
    assert mlfq.session_levels["s1"] == 0
    assert s1.quantum_tokens == 4
    assert len(mlfq.queues[0]) == 1
    assert len(mlfq.queues[2]) == 0
    assert mlfq.total_boosts == 1

def test_mlfq_selection_prioritizes_higher_queues():
    mlfq = MultilevelFeedbackQueue(enable_mlfq=True)
    s_interactive = SchedSession(session_id="s_int", model_tag="test", prompt="short", max_tokens=16)
    s_batch = SchedSession(session_id="s_batch", model_tag="test", prompt="long", max_tokens=256)

    mlfq.enqueue_session(s_interactive)
    mlfq.enqueue_session(s_batch)

    # Demote s_batch to Q1
    mlfq.record_quantum_consumption(s_batch, 16)
    assert mlfq.session_levels["s_batch"] == 1

    # select_next_runnable must pick s_interactive in Q0 before s_batch in Q1
    selected = mlfq.select_next_runnable()
    assert selected is not None
    assert selected.session_id == "s_int"

# ============================================================================
# 2. SLO Admission Control Unit Tests
# ============================================================================

def test_slo_predictive_ttft_calculation():
    slo = SLOAdmissionController(nominal_prefill_tok_s=100.0, nominal_decode_tok_s=25.0)

    # Session with 200 prompt tokens and no active sessions
    s1 = SchedSession(session_id="s1", model_tag="test", prompt="", prompt_tokens=[0] * 200)
    ttft_ms = slo.predict_ttft_ms(200, [])
    # Prefill: 200 / 100 = 2.0s, Queue: 0, First token decode: 1/25 = 0.04s -> Total 2.04s = 2040ms
    assert ttft_ms == pytest.approx(2040.0, abs=10.0)

def test_slo_admission_enforcement_and_rejection():
    slo = SLOAdmissionController(default_target_ttft_ms=1000.0, nominal_prefill_tok_s=100.0)
    target = SLOTarget(target_ttft_ms=1000.0, strict_enforcement=True)

    # Prompt with 50 tokens: estimated ~540ms <= 1000ms -> admitted
    s_small = SchedSession(session_id="s_small", model_tag="test", prompt="", prompt_tokens=[0] * 50)
    assert slo.check_admission(s_small, [], slo_target=target)
    assert slo.total_admitted == 1

    # Prompt with 500 tokens: estimated ~5040ms > 1000ms -> raises SLOViolationError
    s_large = SchedSession(session_id="s_large", model_tag="test", prompt="", prompt_tokens=[0] * 500)
    with pytest.raises(SLOViolationError) as exc_info:
        slo.check_admission(s_large, [], slo_target=target)

    assert exc_info.value.target_ttft_ms == 1000.0
    assert exc_info.value.estimated_ttft_ms > 1000.0
    assert slo.total_rejected == 1

def test_slo_compliance_stats_tracking():
    slo = SLOAdmissionController(default_target_ttft_ms=1500.0)
    target = SLOTarget(target_ttft_ms=1500.0)

    s1 = SchedSession(session_id="s1", model_tag="test", prompt="")
    s1.metrics.enqueue_time = 100.0
    s1.metrics.first_token_time = 101.0  # TTFT = 1000ms <= 1500ms

    s2 = SchedSession(session_id="s2", model_tag="test", prompt="")
    s2.metrics.enqueue_time = 100.0
    s2.metrics.first_token_time = 102.5  # TTFT = 2500ms > 1500ms

    slo.record_session_completion(s1, target)
    slo.record_session_completion(s2, target)

    stats = slo.get_stats()
    assert stats["slo_met_count"] == 1
    assert stats["slo_violated_count"] == 1
    assert stats["slo_compliance_pct"] == 50.0

# ============================================================================
# 3. Multi-Tenant Fair-Share Unit Tests
# ============================================================================

def test_tenant_fair_share_deficit_and_round_robin():
    mgr = TenantFairShareManager(default_tenant_quantum=16.0)

    # Tenant A (weight 1.0) and Tenant B (weight 2.0)
    mgr.get_or_create_tenant("tenant_a", weight=1.0)
    mgr.get_or_create_tenant("tenant_b", weight=2.0)

    s_a = SchedSession(session_id="sa1", model_tag="m", prompt="", tenant_id="tenant_a")
    s_b = SchedSession(session_id="sb1", model_tag="m", prompt="", tenant_id="tenant_b")

    mgr.register_session_admission(s_a, "tenant_a")
    mgr.register_session_admission(s_b, "tenant_b")

    # Selection should choose based on available deficit
    selected = mgr.select_next_tenant_session([s_a, s_b])
    assert selected in (s_a, s_b)

    # Record execution for Tenant A
    mgr.record_execution("tenant_a", 16)
    assert mgr.tenants["tenant_a"].total_tokens_consumed == 16

    stats = mgr.get_stats()
    assert stats["total_tenants"] == 2
    assert stats["total_tokens_served"] == 16
    assert stats["tenants"]["tenant_a"]["tokens_consumed"] == 16

def test_tenant_rate_limiting():
    mgr = TenantFairShareManager()
    t = mgr.get_or_create_tenant("tenant_limited", weight=1.0)
    t.rate_limit_tokens_per_min = 100

    assert not t.is_rate_limited
    t.record_usage(90)
    assert not t.is_rate_limited
    t.record_usage(20)
    assert t.is_rate_limited

# ============================================================================
# 4. Speculative Decoding Unit Tests
# ============================================================================

def test_speculative_decoding_full_acceptance():
    # Target and Draft models emit the exact same tokens
    shared_tokens = [201, 202, 203, 204, 205]
    draft_backend = MockBackend(token_sequence=shared_tokens, name="draft")
    target_backend = MockBackend(token_sequence=shared_tokens, name="target")

    draft_backend.new_session("sess_spec", "test prompt")
    target_backend.new_session("sess_spec", "test prompt")

    spec_engine = SpeculativeEngine(
        target_model_tag="target-7b",
        draft_model_tag="draft-512m",
        k_draft=4,
    )

    res = spec_engine.speculative_step("sess_spec", target_backend, draft_backend, k_draft=4)

    assert len(res.draft_tokens) == 4
    assert len(res.accepted_tokens) == 4
    # All 4 accepted + 1 sampled extra target token = 5 emitted tokens
    assert len(res.emitted_tokens) == 5
    assert res.acceptance_rate == 1.0
    assert res.estimated_speedup > 1.0
    assert spec_engine.acceptance_rate_alpha == 1.0

def test_speculative_decoding_partial_acceptance_and_correction():
    # Draft emits [10, 20, 30, 40], Target emits [10, 20, 999, 40]
    draft_backend = MockBackend(token_sequence=[10, 20, 30, 40], name="draft")
    target_backend = MockBackend(token_sequence=[10, 20, 999, 40], name="target")

    draft_backend.new_session("sess_partial", "test prompt")
    target_backend.new_session("sess_partial", "test prompt")

    spec_engine = SpeculativeEngine(
        target_model_tag="target-7b",
        draft_model_tag="draft-512m",
        k_draft=4,
    )

    res = spec_engine.speculative_step("sess_partial", target_backend, draft_backend, k_draft=4)

    # First 2 tokens match ([10, 20])
    assert res.accepted_tokens == [10, 20]
    # Mismatch at index 2: correction token is 999
    assert res.correction_token == 999
    assert res.emitted_tokens == [10, 20, 999]
    assert res.acceptance_rate == 0.5

# ============================================================================
# 5. Full Scheduler Integration Test (MLFQ + Fair-Share + Speculative)
# ============================================================================

def test_scheduler_mlfq_end_to_end(tmp_path):
    async def _run():
        from unittest.mock import MagicMock
        from pathlib import Path
        from llmhub.config import LLMHubConfig
        from llmhub.core.registry import ModelRegistry
        from llmhub.core.template import ChatTemplateEngine
        from llmhub.server.model_manager import ModelManager

        config = LLMHubConfig(base_dir=tmp_path)
        reg = ModelRegistry(config.db_path)
        mm = ModelManager(config=config, registry=reg)

        meta = MagicMock()
        meta.chat_template = None
        meta.architecture = "llama"
        backend = MockBackend(token_sequence=[1, 2, 3, 4, 5])

        loaded = LoadedModel(
            tag="test-model",
            file_path=Path("m.gguf"),
            backend=backend,
            metadata=meta,
            template_engine=ChatTemplateEngine(None),
            n_ctx=2048,
            n_threads=4,
            weights_bytes=100 * 1024 * 1024,
            kv_cache_bytes=20 * 1024 * 1024,
            overhead_bytes=10 * 1024 * 1024,
            total_memory_bytes=130 * 1024 * 1024,
            pinned=False,
        )
        mm.loaded_models["test-model"] = loaded

        sched = TokenRoundRobinScheduler(
            model_manager=mm,
            default_quantum=4,
            enable_continuous_batching=False,
            enable_mlfq=True,
            enable_fair_share=True,
            enable_slo=True,
        )

        # Submit an interactive session (max_tokens=8) and a batch session (max_tokens=64)
        s_interactive = SchedSession(
            session_id="int_1",
            model_tag="test-model",
            prompt="hello",
            max_tokens=8,
            tenant_id="user_alice",
        )
        s_batch = SchedSession(
            session_id="batch_1",
            model_tag="test-model",
            prompt="write an essay",
            max_tokens=64,
            tenant_id="user_bob",
        )

        await sched.submit_session(s_interactive)
        await sched.submit_session(s_batch)

        # Both start at MLFQ Q0
        assert sched.mlfq.session_levels["int_1"] == 0
        assert sched.mlfq.session_levels["batch_1"] == 0

        # Step through prefill and decode
        for _ in range(25):
            await sched.step()
            if s_interactive.is_done:
                break

        assert s_interactive.is_done
        assert s_interactive.phase == SessionPhase.COMPLETED
        # Interactive completed in Q0
        assert s_interactive.metrics.generated_tokens >= 8

        # Check telemetry includes Phase 7 blocks
        telemetry = sched.get_telemetry()
        assert "mlfq" in telemetry
        assert "slo" in telemetry
        assert "fair_share" in telemetry
        assert telemetry["mlfq"]["enabled"] is True
        assert telemetry["fair_share"]["total_tenants"] >= 2

    asyncio.run(_run())

# ============================================================================
# 6. HTTP API Endpoint Tests
# ============================================================================

def test_api_phase_7_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Test /api/scheduler/mlfq
    r_mlfq = client.get("/api/scheduler/mlfq")
    assert r_mlfq.status_code == 200
    data_mlfq = r_mlfq.json()
    assert "queue_depths" in data_mlfq
    assert "total_boosts" in data_mlfq

    # 2. Test /api/scheduler/slo
    r_slo = client.get("/api/scheduler/slo")
    assert r_slo.status_code == 200
    data_slo = r_slo.json()
    assert "default_target_ttft_ms" in data_slo
    assert "slo_compliance_pct" in data_slo

    # 3. Test /api/scheduler/tenants
    r_tenants = client.get("/api/scheduler/tenants")
    assert r_tenants.status_code == 200
    data_tenants = r_tenants.json()
    assert "total_tenants" in data_tenants
    assert "tenants" in data_tenants

    # 4. Test /api/speculative
    r_spec = client.get("/api/speculative")
    assert r_spec.status_code == 200
    assert isinstance(r_spec.json(), dict)

def test_speculative_session_token_alignment():
    """Verify that both target and draft backend states are aligned with emitted tokens (CRIT-1)."""
    draft_backend = MockBackend(token_sequence=[10, 20, 30, 40], name="draft")
    target_backend = MockBackend(token_sequence=[10, 20, 999, 40], name="target")

    draft_backend.new_session("sess_align", "test prompt")
    target_backend.new_session("sess_align", "test prompt")

    spec_engine = SpeculativeEngine("target-7b", "draft-512m", k_draft=4)
    res = spec_engine.speculative_step("sess_align", target_backend, draft_backend, k_draft=4)

    # Expected emitted: [10, 20, 999]
    assert res.emitted_tokens == [10, 20, 999]

    # Verify draft backend state has exactly [10, 20, 999] (no ghost rejected token 30 or 40)
    draft_sess = draft_backend.get_session("sess_align")
    assert draft_sess.generated_tokens == [10, 20, 999]

    # Verify target backend state has exactly [10, 20, 999]
    target_sess = target_backend.get_session("sess_align")
    assert target_sess.generated_tokens == [10, 20, 999]

def test_speculative_self_model_no_deadlock(tmp_path):
    """Verify speculative decode with target model tag == draft model tag does not deadlock (CRIT-2)."""
    async def _run():
        from unittest.mock import MagicMock
        from pathlib import Path
        from llmhub.config import LLMHubConfig
        from llmhub.core.registry import ModelRegistry
        from llmhub.core.template import ChatTemplateEngine
        from llmhub.server.model_manager import ModelManager

        config = LLMHubConfig(base_dir=tmp_path)
        reg = ModelRegistry(config.db_path)
        mm = ModelManager(config=config, registry=reg)

        meta = MagicMock()
        meta.chat_template = None
        meta.architecture = "llama"
        backend = MockBackend(token_sequence=[1, 2, 3, 4, 5])

        loaded = LoadedModel(
            tag="shared-model",
            file_path=Path("m.gguf"),
            backend=backend,
            metadata=meta,
            template_engine=ChatTemplateEngine(None),
            n_ctx=2048,
            n_threads=4,
            weights_bytes=100 * 1024 * 1024,
            kv_cache_bytes=20 * 1024 * 1024,
            overhead_bytes=10 * 1024 * 1024,
            total_memory_bytes=130 * 1024 * 1024,
            pinned=False,
        )
        mm.loaded_models["shared-model"] = loaded

        sched = TokenRoundRobinScheduler(
            model_manager=mm,
            default_quantum=4,
            enable_continuous_batching=False,
            enable_mlfq=False,
        )

        sess = SchedSession(
            session_id="spec_self",
            model_tag="shared-model",
            prompt="speculative test",
            max_tokens=8,
            use_speculative=True,
            draft_model_tag="shared-model",  # Identical tag
        )

        await sched.submit_session(sess)

        # Step through prefill and decode turns without deadlock
        for _ in range(10):
            await sched.step()
            if sess.is_done:
                break

        assert sess.is_done
        assert sess.phase == SessionPhase.COMPLETED

    asyncio.run(_run())

def test_tenant_rate_limited_returns_none():
    """Verify that when all tenants are rate limited, selection returns None (WARN-1)."""
    mgr = TenantFairShareManager()
    t = mgr.get_or_create_tenant("exhausted_tenant", weight=1.0)
    t.rate_limit_tokens_per_min = 50
    t.record_usage(100)
    assert t.is_rate_limited

    sess = SchedSession(session_id="s_rate", model_tag="m", prompt="", tenant_id="exhausted_tenant")
    selected = mgr.select_next_tenant_session([sess])
    assert selected is None
