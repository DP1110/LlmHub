"""Comprehensive unit and integration tests for Phase 6 Continuous Batching and Prefix Caching."""

import asyncio
from pathlib import Path
import time
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from llmhub.batch.engine import BatchSlotState, ContinuousBatchEngine
from llmhub.batch.prefix_cache import PrefixCache
from llmhub.core.template import ChatTemplateEngine
from llmhub.scheduler.scheduler import TokenRoundRobinScheduler
from llmhub.scheduler.session import SchedSession, SessionPhase
from llmhub.server.app import app, state
from llmhub.server.backend import Backend, DecodeResult, SessionState
from llmhub.server.model_manager import LoadedModel, ModelManager

class MockBatchBackend(Backend):
    def __init__(self, model_name: str = "mock-batch"):
        self.model_name = model_name
        self._loaded = True
        self.sessions: dict = {}
        self.metadata = MagicMock()
        self.metadata.chat_template = None
        self.metadata.architecture = "llama"

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, *args, **kwargs):
        self._loaded = True

    def unload(self):
        self._loaded = False
        self.sessions.clear()

    def tokenize(self, text: str):
        return [ord(c) for c in text]

    def new_session(self, session_id, prompt, max_tokens=128, **kwargs):
        tokens = kwargs.get("prompt_tokens") or ([ord(c) for c in prompt] if prompt else [1, 2, 3])
        sess = SessionState(
            session_id=session_id,
            model_tag=self.model_name,
            prompt=prompt,
            prompt_tokens=list(tokens),
            max_tokens=max_tokens,
        )
        self.sessions[session_id] = sess
        return sess

    def prefill_chunk(self, session_id, max_tokens_chunk=256):
        sess = self.sessions.get(session_id)
        if not sess:
            return 0, True
        remaining = len(sess.prompt_tokens) - sess.prefill_offset
        chunk = min(max_tokens_chunk, max(0, remaining))
        sess.prefill_offset += chunk
        done = sess.prefill_offset >= len(sess.prompt_tokens)
        if done:
            sess.prefill_completed = True
        return chunk, done

    def decode_step(self, session_id, n_tokens=1):
        sess = self.sessions.get(session_id)
        if not sess:
            return DecodeResult(session_id, [], [], True)

        curr = len(sess.generated_tokens)
        tokens = [curr + i for i in range(n_tokens)]
        deltas = [f" tok_{t}" for t in tokens]
        sess.generated_tokens.extend(tokens)

        done = len(sess.generated_tokens) >= sess.max_tokens
        return DecodeResult(
            session_id=session_id,
            tokens=tokens,
            text_deltas=deltas,
            done=done,
            stop_reason="stop" if done else None,
            ttft_ms=5.0,
            decode_tok_s=60.0,
            total_tokens_generated=len(sess.generated_tokens),
        )

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def close_session(self, session_id):
        self.sessions.pop(session_id, None)

    def get_info(self):
        return {"status": "loaded", "active_sessions": len(self.sessions)}

def make_batch_model(tag: str, backend: Backend, max_slots: int = 4) -> LoadedModel:
    lm = LoadedModel(
        tag=tag,
        file_path=Path(f"{tag}.gguf"),
        backend=backend,
        metadata=backend.metadata,
        template_engine=ChatTemplateEngine(None),
        n_ctx=2048,
        n_threads=4,
        weights_bytes=100 * 1024 * 1024,
        kv_cache_bytes=20 * 1024 * 1024,
        overhead_bytes=10 * 1024 * 1024,
        total_memory_bytes=130 * 1024 * 1024,
        pinned=False,
    )
    lm.batch_engine = ContinuousBatchEngine(model_tag=tag, max_slots=max_slots)
    return lm

def test_prefix_cache_exact_and_partial_match():
    cache = PrefixCache(max_cached_tokens=1024, min_prefix_tokens=8)

    # Insert a 64-token prefix (e.g. system prompt)
    system_tokens = list(range(100, 164))
    cache.insert_prefix("test-model", system_tokens)

    # 1. Exact match query
    matched, entry = cache.find_longest_prefix("test-model", system_tokens)
    assert matched == 64
    assert entry is not None
    assert cache.cache_hits == 1

    # 2. Query with prompt extending the system prompt (+ 10 new user tokens)
    user_prompt = system_tokens + list(range(200, 210))
    matched_ext, entry_ext = cache.find_longest_prefix("test-model", user_prompt)
    assert matched_ext == 64
    assert entry_ext is not None
    assert cache.cache_hits == 2
    assert cache.tokens_saved == 128

    # 3. Query with completely different tokens
    diff_tokens = list(range(500, 520))
    matched_diff, entry_diff = cache.find_longest_prefix("test-model", diff_tokens)
    assert matched_diff == 0
    assert entry_diff is None
    assert cache.cache_misses == 1

def test_prefix_cache_lru_eviction():
    # Cache with maximum capacity of 30 tokens
    cache = PrefixCache(max_cached_tokens=30, min_prefix_tokens=8)

    t1 = list(range(10))
    t2 = list(range(10, 20))
    t3 = list(range(20, 30))
    t4 = list(range(30, 45))  # 15 tokens -> forces eviction

    cache.insert_prefix("m", t1)
    cache.insert_prefix("m", t2)
    cache.insert_prefix("m", t3)
    assert len(cache._entries) == 3

    # Access t1 to make t2 the least recently accessed
    cache.find_longest_prefix("m", t1)

    # Insert t4, which should trigger eviction of oldest entry
    cache.insert_prefix("m", t4)
    stats = cache.get_stats()
    assert stats["cached_tokens"] <= 30

def test_batch_slot_allocation_and_saturation():
    engine = ContinuousBatchEngine("m1", max_slots=3)

    s0 = engine.allocate_slot("sess-0", "prompt", [1, 2, 3, 4, 5, 6, 7, 8])
    s1 = engine.allocate_slot("sess-1", "prompt", [1, 2, 3, 4, 5, 6, 7, 8])
    s2 = engine.allocate_slot("sess-2", "prompt", [1, 2, 3, 4, 5, 6, 7, 8])
    assert s0 is not None and s0.slot_id == 0
    assert s1 is not None and s1.slot_id == 1
    assert s2 is not None and s2.slot_id == 2

    # 4th slot should return None (saturated)
    s3 = engine.allocate_slot("sess-3", "prompt", [1, 2, 3, 4, 5, 6, 7, 8])
    assert s3 is None

    # Free slot 1
    engine.free_slot(1)
    assert engine.slots[1].state == BatchSlotState.FREE

    # Now allocation should succeed in slot 1
    s3_retry = engine.allocate_slot("sess-3", "prompt", [1, 2, 3, 4, 5, 6, 7, 8])
    assert s3_retry is not None
    assert s3_retry.slot_id == 1

def test_continuous_batch_multi_slot_parallel_stepping():
    backend = MockBatchBackend("m1")
    engine = ContinuousBatchEngine("m1", max_slots=3)

    # Allocate 3 sessions
    engine.allocate_slot("s0", "prompt zero", [1, 2, 3], max_tokens=4)
    engine.allocate_slot("s1", "prompt one", [4, 5, 6], max_tokens=4)
    engine.allocate_slot("s2", "prompt two", [7, 8, 9], max_tokens=4)

    # Step 1: Initialize ALLOCATED slots
    res1 = engine.step_batch(backend)
    assert res1.active_slots_count == 3
    assert all(s.state == BatchSlotState.PREFILLING for s in engine.slots)

    # Step 2: Prefill chunks
    res2 = engine.step_batch(backend, chunk_size=256)
    assert res2.total_prefill_tokens == 9
    assert all(s.state == BatchSlotState.DECODING for s in engine.slots)

    # Step 3: Parallel decode turn (each slot decodes 2 tokens)
    res3 = engine.step_batch(backend, default_quantum=2)
    assert res3.decoding_slots_count == 3
    assert res3.total_generated_tokens == 6  # 2 tokens * 3 slots!
    assert "s0" in res3.session_deltas
    assert "s1" in res3.session_deltas
    assert "s2" in res3.session_deltas
    assert len(res3.session_deltas["s0"]) == 2

    # Step 4: Next decode turn (each slot decodes 2 tokens -> all reach max_tokens=4 and complete!)
    res4 = engine.step_batch(backend, default_quantum=2)
    assert res4.total_generated_tokens == 6
    assert len(res4.completed_sessions) == 3
    assert "s0" in res4.completed_sessions
    assert "s1" in res4.completed_sessions
    assert "s2" in res4.completed_sessions
    assert all(s.state == BatchSlotState.COMPLETED for s in engine.slots)

def test_continuous_batch_dynamic_joining_and_leaving():
    backend = MockBatchBackend("m1")
    engine = ContinuousBatchEngine("m1", max_slots=2)

    # Session A: short (max_tokens=2)
    # Session B: long (max_tokens=6)
    engine.allocate_slot("sA", "hi", [1, 2], max_tokens=2)
    engine.allocate_slot("sB", "hello", [3, 4], max_tokens=6)

    # Init & Prefill
    engine.step_batch(backend)
    engine.step_batch(backend)
    assert engine.slots[0].state == BatchSlotState.DECODING
    assert engine.slots[1].state == BatchSlotState.DECODING

    # Turn 1 decode: 2 tokens each
    res_dec = engine.step_batch(backend, default_quantum=2)
    # sA reaches 2 tokens and completes! sB reaches 2 tokens and remains decoding!
    assert "sA" in res_dec.completed_sessions
    assert "sB" not in res_dec.completed_sessions
    assert engine.slots[0].state == BatchSlotState.COMPLETED
    assert engine.slots[1].state == BatchSlotState.DECODING

    # Free sA slot
    engine.free_slot(0)
    assert engine.slots[0].state == BatchSlotState.FREE

    # Dynamically admit session C while Session B is still running in slot 1!
    slotC = engine.allocate_slot("sC", "welcome", [5, 6], max_tokens=4)
    assert slotC is not None
    assert slotC.slot_id == 0

    # Step: in this iteration, slot 0 is initialized while slot 1 continues
    engine.step_batch(backend)
    assert engine.slots[0].state == BatchSlotState.PREFILLING
    assert engine.slots[1].state == BatchSlotState.DECODING

def test_scheduler_continuous_batching_integration():
    async def _run():
        backend = MockBatchBackend("model-batch")
        lm = make_batch_model("model-batch", backend, max_slots=4)
        mgr = ModelManager()
        mgr.loaded_models["model-batch"] = lm

        sched = TokenRoundRobinScheduler(model_manager=mgr, default_quantum=4, enable_continuous_batching=True)

        # Submit 3 sessions for the same model
        s1 = SchedSession("s1", "model-batch", "prompt one", max_tokens=4)
        s2 = SchedSession("s2", "model-batch", "prompt two", max_tokens=4)
        s3 = SchedSession("s3", "model-batch", "prompt three", max_tokens=4)

        await sched.submit_session(s1)
        await sched.submit_session(s2)
        await sched.submit_session(s3)

        assert len(sched.runnable_queue) == 3

        # Step until all 3 sessions complete
        step_count = 0
        while not s1.is_done or not s2.is_done or not s3.is_done:
            await sched.step()
            step_count += 1
            if step_count > 20:
                break

        assert s1.phase == SessionPhase.COMPLETED
        assert s2.phase == SessionPhase.COMPLETED
        assert s3.phase == SessionPhase.COMPLETED
        assert s1.metrics.generated_tokens == 4
        assert s2.metrics.generated_tokens == 4
        assert s3.metrics.generated_tokens == 4
        assert len(sched.runnable_queue) == 0
    asyncio.run(_run())

def test_scheduler_multi_model_batch_interleaving():
    async def _run():
        bA = MockBatchBackend("mA")
        bB = MockBatchBackend("mB")
        lmA = make_batch_model("mA", bA, max_slots=2)
        lmB = make_batch_model("mB", bB, max_slots=2)

        mgr = ModelManager()
        mgr.loaded_models["mA"] = lmA
        mgr.loaded_models["mB"] = lmB

        sched = TokenRoundRobinScheduler(model_manager=mgr, default_quantum=2, enable_continuous_batching=True)

        sA1 = SchedSession("sA1", "mA", "hello A1", max_tokens=4, quantum_tokens=2)
        sA2 = SchedSession("sA2", "mA", "hello A2", max_tokens=4, quantum_tokens=2)
        sB1 = SchedSession("sB1", "mB", "hello B1", max_tokens=4, quantum_tokens=2)
        sB2 = SchedSession("sB2", "mB", "hello B2", max_tokens=4, quantum_tokens=2)

        await sched.submit_session(sA1)
        await sched.submit_session(sA2)
        await sched.submit_session(sB1)
        await sched.submit_session(sB2)

        step_count = 0
        while any(not s.is_done for s in [sA1, sA2, sB1, sB2]):
            await sched.step()
            step_count += 1
            if step_count > 30:
                break

        assert all(s.phase == SessionPhase.COMPLETED for s in [sA1, sA2, sB1, sB2])
        # Scheduler interleaved batches across mA and mB
        assert sched.total_context_switches >= 2
    asyncio.run(_run())

def test_api_batch_endpoint():
    backend = MockBatchBackend("api-batch-model")
    lm = make_batch_model("api-batch-model", backend, max_slots=4)
    state.model_manager.loaded_models["api-batch-model"] = lm

    try:
        client = TestClient(app)

        # GET /api/batch
        resp = client.get("/api/batch")
        assert resp.status_code == 200
        data = resp.json()
        assert "batch_engines" in data
        assert "api-batch-model" in data["batch_engines"]
        b_info = data["batch_engines"]["api-batch-model"]
        assert b_info["max_slots"] == 4
        assert "prefix_cache" in b_info
        assert len(b_info["slots"]) == 4

        # GET /api/ps should now also include batch_summary
        ps_resp = client.get("/api/ps")
        assert ps_resp.status_code == 200
        ps_data = ps_resp.json()
        assert "batch_summary" in ps_data
        assert "api-batch-model" in ps_data["batch_summary"]
        assert ps_data["batch_summary"]["api-batch-model"]["max_slots"] == 4
    finally:
        state.model_manager.loaded_models.pop("api-batch-model", None)

def test_prefix_cache_non_aligned_sub_prefix():
    cache = PrefixCache(max_cached_tokens=1024, min_prefix_tokens=8)
    common_prefix = [100 + i for i in range(80)]  # 80 tokens
    prompt_a = common_prefix + [1, 2, 3, 4, 5]    # 85 tokens (non-aligned)
    prompt_b = common_prefix + [10, 20, 30]       # 83 tokens (non-aligned)

    cache.insert_prefix("m1", prompt_a)
    # Check that prompt_b matches at least the 64-token checkpoint of common_prefix
    matched, entry = cache.find_longest_prefix("m1", prompt_b)
    assert matched >= 64
    assert entry is not None
    assert cache.tokens_saved >= 64

def test_continuous_batch_per_slot_error_isolation():
    class FaultyMockBackend(MockBatchBackend):
        def decode_step(self, session_id, n_tokens=1):
            if session_id == "failing-sess":
                raise RuntimeError("Simulated GPU/CPU execution fault in slot")
            return super().decode_step(session_id, n_tokens)

    backend = FaultyMockBackend("faulty-model")
    engine = ContinuousBatchEngine("faulty-model", max_slots=2)

    s1 = engine.allocate_slot("good-sess", "good prompt", [1, 2, 3, 4], max_tokens=10)
    s2 = engine.allocate_slot("failing-sess", "bad prompt", [5, 6, 7, 8], max_tokens=10)
    assert s1 and s2

    # Step 1: Init ALLOCATED slots -> PREFILLING
    res1 = engine.step_batch(backend, default_quantum=2)
    assert all(s.state == BatchSlotState.PREFILLING for s in [s1, s2])

    # Step 2: Prefill chunks -> DECODING
    res2 = engine.step_batch(backend, default_quantum=2)
    assert all(s.state == BatchSlotState.DECODING for s in [s1, s2])

    # Step 3: Decode step -> FaultyMockBackend raises on failing-sess
    res3 = engine.step_batch(backend, default_quantum=2)
    assert "failing-sess" in res3.failed_sessions
    assert s2.state == BatchSlotState.FAILED
    assert s1.state in (BatchSlotState.DECODING, BatchSlotState.COMPLETED)
    assert "good-sess" in res3.session_deltas

def test_waiting_session_cancellation_no_active_sessions_underflow():
    async def _run():
        backend = MockBatchBackend("test-underflow")
        lm = make_batch_model("test-underflow", backend, max_slots=2)
        mgr = ModelManager()
        mgr.loaded_models["test-underflow"] = lm

        sched = TokenRoundRobinScheduler(model_manager=mgr, default_quantum=2, enable_continuous_batching=True)

        s1 = SchedSession("s1", "test-underflow", "hello", max_tokens=4)
        await sched.submit_session(s1)
        assert lm.active_sessions == 0
        assert s1.phase == SessionPhase.WAITING

        # Client disconnects/cancels BEFORE scheduler step
        cancelled = await sched.cancel_session("s1")
        assert cancelled
        assert s1.phase == SessionPhase.CANCELLED
        # active_sessions MUST remain 0 and never underflow to negative or decrement idle count
        assert lm.active_sessions == 0

    asyncio.run(_run())

def test_continuous_batch_session_cancellation_under_lock():
    async def _run():
        backend = MockBatchBackend("test-cancel")
        lm = make_batch_model("test-cancel", backend, max_slots=2)
        mgr = ModelManager()
        mgr.loaded_models["test-cancel"] = lm

        sched = TokenRoundRobinScheduler(model_manager=mgr, default_quantum=2, enable_continuous_batching=True)

        s1 = SchedSession("s1", "test-cancel", "hello", max_tokens=20)
        await sched.submit_session(s1)

        # Step 1: Allocates slot and steps prefill/decode
        await sched.step()
        assert lm.active_sessions == 1
        assert s1.slot_id is not None

        # Cancel while active in slot
        cancelled = await sched.cancel_session("s1")
        assert cancelled
        assert s1.phase == SessionPhase.CANCELLED
        assert lm.active_sessions == 0
        assert s1.slot_id is None

        # Verify slot in batch engine is freed
        slot = lm.batch_engine.get_slot_by_session("s1")
        assert slot is None
        free_slot = lm.batch_engine.find_free_slot()
        assert free_slot is not None

    asyncio.run(_run())

