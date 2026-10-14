"""Comprehensive tests for Token-level Round-Robin Scheduler, CPU Threading Modes, and Simulator."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from fastapi.testclient import TestClient

from llmhub.core.template import ChatTemplateEngine
from llmhub.scheduler.policy import (
    DeficitRoundRobinPolicy,
    ThreadingMode,
    ThreadingStrategy,
)
from llmhub.scheduler.scheduler import (
    QueueFullError,
    SessionTimeoutError,
    TokenRoundRobinScheduler,
)
from llmhub.scheduler.session import SchedSession, SessionPhase, SessionPriority
from llmhub.scheduler.simulator import (
    DeterministicSchedulerSimulator,
    SimSessionSpec,
)
from llmhub.server.app import app, state
from llmhub.server.backend import Backend, DecodeResult, SessionState
from llmhub.server.model_manager import LoadedModel, ModelManager

class MockSchedBackend(Backend):
    def __init__(self, model_name: str = "mock-model"):
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

    def new_session(self, session_id, prompt, max_tokens=128, **kwargs):
        sess = SessionState(
            session_id=session_id,
            model_tag=self.model_name,
            prompt=prompt,
            prompt_tokens=[1] * len(prompt.split()),
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
            ttft_ms=10.0,
            decode_tok_s=50.0,
            total_tokens_generated=len(sess.generated_tokens),
        )

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def close_session(self, session_id):
        self.sessions.pop(session_id, None)

    def get_info(self):
        return {"status": "loaded", "active_sessions": len(self.sessions)}

def make_loaded_model(tag: str, backend: Backend) -> LoadedModel:
    return LoadedModel(
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

def test_quantum_stepping_and_lifecycle():
    async def _run():
        b = MockSchedBackend("m1")
        lm = make_loaded_model("m1", b)
        mgr = ModelManager()
        mgr.loaded_models["m1"] = lm

        sched = TokenRoundRobinScheduler(model_manager=mgr, default_quantum=4, default_chunk_size=256)

        sess = SchedSession(
            session_id="s1",
            model_tag="m1",
            prompt="Hello world test",
            max_tokens=8,
            quantum_tokens=4,
        )
        await sched.submit_session(sess)
        assert len(sched.runnable_queue) == 1
        assert sess.phase == SessionPhase.WAITING

        # Step 1: Initialize session
        await sched.step()
        assert sess.phase == SessionPhase.PREFILLING

        # Step 2: Prefill chunk
        await sched.step()
        assert sess.phase == SessionPhase.DECODING

        # Step 3: First quantum (4 tokens)
        await sched.step()
        assert sess.metrics.generated_tokens == 4
        assert sess.metrics.quanta_executed == 1
        assert sess.phase == SessionPhase.DECODING

        # Step 4: Second quantum (4 tokens -> reaches max_tokens 8)
        await sched.step()
        assert sess.metrics.generated_tokens == 8
        assert sess.phase == SessionPhase.COMPLETED
        assert sess.stop_reason in ("stop", "length")
        assert len(sched.runnable_queue) == 0
        assert len(sched.completed_sessions) == 1

        # Check token stream queue received tokens
        tokens_received = []
        while not sess.token_queue.empty():
            item = sess.token_queue.get_nowait()
            if "delta" in item:
                tokens_received.append(item["delta"])
        assert len(tokens_received) == 8
    asyncio.run(_run())

def test_chunked_prefill_interleaving():
    async def _run():
        b = MockSchedBackend("m1")
        lm = make_loaded_model("m1", b)
        mgr = ModelManager()
        mgr.loaded_models["m1"] = lm

        # Prompt with 600 words, chunk size 256 -> requires 3 prefill steps
        long_prompt = "word " * 600
        sched = TokenRoundRobinScheduler(model_manager=mgr, default_chunk_size=256)
        sess = SchedSession(
            session_id="long-s1",
            model_tag="m1",
            prompt=long_prompt,
            max_tokens=4,
            chunk_size=256,
        )
        await sched.submit_session(sess)

        # Init
        await sched.step()
        assert sess.phase == SessionPhase.PREFILLING

        # Chunk 1 (256 tokens)
        await sched.step()
        assert sess.phase == SessionPhase.PREFILLING
        assert sess.metrics.prefill_tokens == 256

        # Chunk 2 (256 tokens -> total 512)
        await sched.step()
        assert sess.phase == SessionPhase.PREFILLING
        assert sess.metrics.prefill_tokens == 512

        # Chunk 3 (remaining 88 tokens -> total 600, prefill completed)
        await sched.step()
        assert sess.phase == SessionPhase.DECODING
        assert sess.metrics.prefill_tokens == 600
    asyncio.run(_run())

def test_multi_model_interleaved_decode():
    async def _run():
        b1 = MockSchedBackend("model-A")
        b2 = MockSchedBackend("model-B")
        lm1 = make_loaded_model("model-A", b1)
        lm2 = make_loaded_model("model-B", b2)

        mgr = ModelManager()
        mgr.loaded_models["model-A"] = lm1
        mgr.loaded_models["model-B"] = lm2

        sched = TokenRoundRobinScheduler(model_manager=mgr, default_quantum=2)

        sA = SchedSession(session_id="sA", model_tag="model-A", prompt="ping", max_tokens=4, quantum_tokens=2)
        sB = SchedSession(session_id="sB", model_tag="model-B", prompt="pong", max_tokens=4, quantum_tokens=2)

        await sched.submit_session(sA)
        await sched.submit_session(sB)

        # Step through until both complete
        step_count = 0
        while not sA.is_done or not sB.is_done:
            await sched.step()
            step_count += 1
            if step_count > 20:
                break

        assert sA.phase == SessionPhase.COMPLETED
        assert sB.phase == SessionPhase.COMPLETED
        assert sA.metrics.generated_tokens == 4
        assert sB.metrics.generated_tokens == 4
        # Context switches occurred between model-A and model-B
        assert sched.total_context_switches >= 2
    asyncio.run(_run())

def test_deficit_round_robin_weighting():
    policy = DeficitRoundRobinPolicy(default_quantum=4)
    s1 = SchedSession("s1", "m", "", weight=1.0, quantum_tokens=4)
    s2 = SchedSession("s2", "m", "", weight=2.0, quantum_tokens=4)

    policy.replenish_credits(s1)
    policy.replenish_credits(s2)

    # s1 gets 4 credits, s2 gets 8 credits
    assert s1.deficit == 4.0
    assert s2.deficit == 8.0

    # s1 consumes 4 credits
    s1.deficit -= 4.0
    assert s1.deficit == 0.0

    # s2 still has 8 credits >= 4, so it can run again
    next_s = policy.select_next_session([s1, s2])
    assert next_s == s2

def test_session_cancellation():
    async def _run():
        b = MockSchedBackend("m1")
        lm = make_loaded_model("m1", b)
        mgr = ModelManager()
        mgr.loaded_models["m1"] = lm

        sched = TokenRoundRobinScheduler(model_manager=mgr)
        sess = SchedSession("cancel-s1", "m1", "prompt", max_tokens=20)
        await sched.submit_session(sess)

        # Start session
        await sched.step()
        await sched.step()

        # Cancel mid-execution
        cancelled = await sched.cancel_session("cancel-s1")
        assert cancelled is True
        assert sess.phase == SessionPhase.CANCELLED
        assert sess.is_done is True
        assert "cancel-s1" not in b.sessions
        assert lm.active_sessions == 0
        assert sess not in sched.runnable_queue
        assert sess in sched.completed_sessions

        # Next step should not execute cancelled session
        stepped = await sched.step()
        assert not stepped
    asyncio.run(_run())

def test_session_timeout():
    async def _run():
        b = MockSchedBackend("m1")
        lm = make_loaded_model("m1", b)
        mgr = ModelManager()
        mgr.loaded_models["m1"] = lm

        sched = TokenRoundRobinScheduler(model_manager=mgr)
        sess = SchedSession("timeout-s1", "m1", "prompt", timeout_seconds=0.01)
        await sched.submit_session(sess)

        # Simulate timeout passage
        sess.metrics.enqueue_time = time.monotonic() - 1.0
        assert sess.is_expired is True

        # Step detects timeout and cleans up
        await sched.step()
        assert sess.phase == SessionPhase.TIMED_OUT
        assert sess.is_done is True
        assert sess not in sched.runnable_queue
        assert sess in sched.completed_sessions
        assert lm.active_sessions == 0
    asyncio.run(_run())

def test_dynamic_set_n_threads():
    async def _run():
        b = MockSchedBackend("m1")
        b.set_threads_called = []
        b.set_n_threads = lambda n: b.set_threads_called.append(n)

        lm = make_loaded_model("m1", b)
        lm.n_threads = 1
        mgr = ModelManager()
        mgr.loaded_models["m1"] = lm

        sched = TokenRoundRobinScheduler(model_manager=mgr, threading_mode=ThreadingMode.TIME_SLICE)
        sched.threading_strategy.total_physical_cores = 6

        sess = SchedSession("s1", "m1", "prompt")
        await sched.submit_session(sess)

        await sched.step()
        # Threading allocation should have dynamically adjusted backend to 6 threads
        assert lm.n_threads == 6
        assert 6 in b.set_threads_called
    asyncio.run(_run())


def test_queue_backpressure_rejection():
    async def _run():
        mgr = ModelManager()
        sched = TokenRoundRobinScheduler(model_manager=mgr, max_queue_depth=2)

        s1 = SchedSession("s1", "m1", "")
        s2 = SchedSession("s2", "m1", "")
        s3 = SchedSession("s3", "m1", "")

        await sched.submit_session(s1)
        await sched.submit_session(s2)

        # 3rd session should be rejected with QueueFullError
        with pytest.raises(QueueFullError) as exc_info:
            await sched.submit_session(s3)
        assert "Scheduler queue is full" in str(exc_info.value)
    asyncio.run(_run())

def test_cpu_threading_modes():
    strat = ThreadingStrategy()
    strat.total_physical_cores = 8

    # (a) Time-Slice: all cores to each active model
    strat.mode = ThreadingMode.TIME_SLICE
    alloc_a = strat.calculate_threads({"m1", "m2"})
    assert alloc_a["m1"] == 8
    assert alloc_a["m2"] == 8

    # (b) Partition: split physical cores evenly
    strat.mode = ThreadingMode.PARTITION
    alloc_b = strat.calculate_threads({"m1", "m2"})
    assert alloc_b["m1"] == 4
    assert alloc_b["m2"] == 4

    alloc_b4 = strat.calculate_threads({"m1", "m2", "m3", "m4"})
    assert alloc_b4["m1"] == 2
    assert alloc_b4["m4"] == 2

    # (c) Hybrid: single model gets all, multiple models partition
    strat.mode = ThreadingMode.HYBRID
    alloc_c1 = strat.calculate_threads({"m1"})
    assert alloc_c1["m1"] == 8

    alloc_c2 = strat.calculate_threads({"m1", "m2"})
    assert alloc_c2["m1"] == 4
    assert alloc_c2["m2"] == 4

def test_deterministic_simulator_jains_fairness():
    sim = DeterministicSchedulerSimulator(
        threading_mode=ThreadingMode.HYBRID,
        physical_cores=4,
        base_decode_ms_per_tok=10.0,
    )

    # 4 concurrent models, 2 sessions each with equal weight
    specs = []
    for m_idx in range(4):
        for s_idx in range(2):
            specs.append(SimSessionSpec(
                session_id=f"m{m_idx}_s{s_idx}",
                model_tag=f"model-{m_idx}",
                prompt_tokens=64,
                completion_tokens=32,
                weight=1.0,
                priority=5,
                quantum_tokens=4,
            ))

    res = sim.run_simulation(specs)
    assert res.num_models == 4
    assert res.total_tokens == 4 * 2 * 32
    # Verify mathematical fairness
    assert res.jains_fairness_index >= 0.95
    assert res.total_context_switches > 0

def test_api_scheduler_endpoints():
    client = TestClient(app)

    # GET /api/scheduler
    resp = client.get("/api/scheduler")
    assert resp.status_code == 200
    data = resp.json()
    assert "queue_depth" in data
    assert "threading_mode" in data
    assert "jains_fairness_index" in data

    # POST /api/scheduler/mode
    resp_mode = client.post("/api/scheduler/mode?mode=partition")
    assert resp_mode.status_code == 200
    assert resp_mode.json()["threading_mode"] == "partition"
    assert state.scheduler.threading_strategy.mode == ThreadingMode.PARTITION

    # Invalid mode returns 400
    resp_inv = client.post("/api/scheduler/mode?mode=invalid_mode")
    assert resp_inv.status_code == 400
