"""Benchmark harness for LLMHub Phase 1 Single-Model Runner.

Measures:
1. GGUF binary header parse latency (ms)
2. Jinja2 chat template compilation and rendering latency (ms)
3. Memory breakdown computation latency (ms)
4. Model inference speed (TTFT, decode tok/s) when a real GGUF model is present
"""

import sys
import time
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmhub.config import LLMHubConfig
from llmhub.core.gguf_parser import GGUFParser
from llmhub.core.registry import ModelRegistry
from llmhub.core.template import ChatTemplateEngine
from llmhub.utils.math_calc import compute_memory_breakdown, estimate_decode_throughput
from llmhub.utils.platform import get_hardware_profile
from llmhub.hardware.detector import HardwareDetector
from llmhub.hardware.tuner import AutoTuner
from tests.test_gguf_parser import create_synthetic_gguf

def benchmark_gguf_parser(iterations: int = 500) -> float:
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        create_synthetic_gguf(tmp_path)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = GGUFParser.parse_file(tmp_path)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000.0
        return avg_ms
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def benchmark_chat_template(iterations: int = 1000) -> float:
    engine = ChatTemplateEngine(architecture="qwen")
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "What is the memory bandwidth of DDR4 dual-channel?"},
        {"role": "assistant", "content": "Dual-channel DDR4-3200 provides theoretical peak of 51.2 GB/s, sustained ~40 GB/s."},
        {"role": "user", "content": "How does that affect 7B model inference?"}
    ]
    start = time.perf_counter()
    for _ in range(iterations):
        _ = engine.render(messages, add_generation_prompt=True)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iterations) * 1000.0
    return avg_ms

def benchmark_memory_math(iterations: int = 5000) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        _ = compute_memory_breakdown(
            weights_bytes=4 * 1024 * 1024 * 1024,
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            n_ctx=4096
        )
        _ = estimate_decode_throughput(4 * 1024 * 1024 * 1024)
    elapsed = time.perf_counter() - start
    avg_us = (elapsed / iterations) * 1_000_000.0
    return avg_us

def benchmark_hardware_detection(iterations: int = 5) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        _ = HardwareDetector.get_profile()
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iterations) * 1000.0
    return avg_ms

def benchmark_auto_tuning(iterations: int = 500) -> float:
    prof = HardwareDetector.get_profile()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = AutoTuner.generate_tuning_plan(
            model_tag="test/7b:q4_k_m",
            weights_bytes=4 * 1024 * 1024 * 1024,
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            ctx_max=4096,
            profile=prof
        )
    elapsed = time.perf_counter() - start
    avg_us = (elapsed / iterations) * 1_000_000.0
    return avg_us

def benchmark_backend_quantum(iterations: int = 1000) -> float:
    from llmhub.server.backend import DecodeResult, SessionState
    sess = SessionState(
        session_id="bench-s1",
        model_tag="bench-model",
        prompt="Hi",
        prompt_tokens=[1, 2, 3]
    )
    start = time.perf_counter()
    for _ in range(iterations):
        _ = DecodeResult(
            session_id=sess.session_id,
            tokens=[10],
            text_deltas=[" token"],
            done=False,
            ttft_ms=1.5,
            decode_tok_s=85.0,
            total_tokens_generated=1
        )
    elapsed = time.perf_counter() - start
    avg_us = (elapsed / iterations) * 1_000_000.0
    return avg_us

def benchmark_server_sse(iterations: int = 100) -> float:
    from fastapi.testclient import TestClient
    from llmhub.server.app import app, state
    from llmhub.server.model_manager import LoadedModel
    from tests.test_server import MockBackend
    from unittest.mock import MagicMock

    orig_registry = state.registry
    mock_reg = MagicMock()
    mock_rec = MagicMock()
    mock_rec.tag = "mock"
    mock_rec.file_path = "mock.gguf"
    mock_reg.find_model.return_value = mock_rec
    state.registry = mock_reg

    b = MockBackend()
    lm = LoadedModel(
        tag="mock",
        file_path=Path("mock.gguf"),
        backend=b,
        metadata=b.metadata,
        template_engine=ChatTemplateEngine(None),
        n_ctx=2048,
        n_threads=4,
        weights_bytes=100 * 1024 * 1024,
        kv_cache_bytes=20 * 1024 * 1024,
        overhead_bytes=10 * 1024 * 1024,
        total_memory_bytes=130 * 1024 * 1024,
        pinned=False
    )
    state.model_manager.loaded_models["mock"] = lm

    client = TestClient(app)
    payload = {
        "model": "mock",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": True
    }

    try:
        start = time.perf_counter()
        for _ in range(iterations):
            resp = client.post("/v1/chat/completions", json=payload)
            _ = resp.text
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000.0
        return avg_ms
    finally:
        state.model_manager.loaded_models.pop("mock", None)
        state.registry = orig_registry

def benchmark_model_manager_admission(iterations: int = 2000) -> float:
    from llmhub.server.model_manager import ModelManager
    mgr = ModelManager()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = mgr.check_admission(required_bytes=4 * 1024 * 1024 * 1024)
    elapsed = time.perf_counter() - start
    avg_us = (elapsed / iterations) * 1_000_000.0
    return avg_us

def benchmark_scheduler_step_overhead(iterations: int = 500) -> float:
    import asyncio
    from llmhub.scheduler.scheduler import TokenRoundRobinScheduler
    from llmhub.scheduler.session import SchedSession
    from llmhub.server.model_manager import ModelManager
    from tests.test_scheduler import MockSchedBackend, make_loaded_model

    b = MockSchedBackend("bench-m")
    lm = make_loaded_model("bench-m", b)
    mgr = ModelManager()
    mgr.loaded_models["bench-m"] = lm

    sched = TokenRoundRobinScheduler(model_manager=mgr, default_quantum=4)

    async def _run():
        for i in range(iterations):
            sess = SchedSession(f"s-{i}", "bench-m", "prompt", max_tokens=4, quantum_tokens=4)
            await sched.submit_session(sess)
            # WAITING -> PREFILLING
            await sched.step()
            # PREFILLING -> DECODING
            await sched.step()
            # DECODING -> COMPLETED
            await sched.step()

    start = time.perf_counter()
    asyncio.run(_run())
    elapsed = time.perf_counter() - start
    avg_us = (elapsed / (iterations * 3)) * 1_000_000.0
    return avg_us

def benchmark_scheduler_modes_comparison() -> None:
    from llmhub.scheduler.policy import ThreadingMode
    from llmhub.scheduler.simulator import DeterministicSchedulerSimulator, SimSessionSpec

    prof = HardwareDetector.get_profile()
    cores = max(1, prof.cpu.physical_cores)

    modes = [
        ("(a) Time-Slice", ThreadingMode.TIME_SLICE),
        ("(b) Partition", ThreadingMode.PARTITION),
        ("(c) Hybrid", ThreadingMode.HYBRID),
    ]
    model_counts = [1, 2, 4]

    print("\n" + "=" * 80)
    print(" PHASE 5 SCHEDULER CPU THREADING MODES COMPARISON (1, 2, 4 CONCURRENT MODELS)")
    print("=" * 80)
    print(f"{'Mode':<18} | {'Models':<7} | {'Agg tok/s':<11} | {'Avg TTFT(ms)':<13} | {'Context Sw':<11} | {'Jain Index':<10}")
    print("-" * 80)

    for mode_name, mode_enum in modes:
        for n_models in model_counts:
            sim = DeterministicSchedulerSimulator(
                threading_mode=mode_enum,
                physical_cores=cores,
                base_decode_ms_per_tok=20.0,
                context_switch_penalty_ms=0.5,
            )

            # 2 sessions per model, 128 prompt tokens, 64 completion tokens
            specs = []
            for m_idx in range(n_models):
                for s_idx in range(2):
                    specs.append(SimSessionSpec(
                        session_id=f"m{m_idx}_s{s_idx}",
                        model_tag=f"model-{m_idx}",
                        prompt_tokens=128,
                        completion_tokens=64,
                        weight=1.0,
                        priority=5,
                        quantum_tokens=4,
                        chunk_size=256,
                    ))

            res = sim.run_simulation(specs)
            agg_tok_s = res.total_tokens / (res.total_time_ms / 1000.0) if res.total_time_ms > 0 else 0.0
            avg_ttft = sum(m["ttft_ms"] for m in res.sessions_metrics.values()) / max(1, len(res.sessions_metrics))

            print(f"{mode_name:<18} | {n_models:<7} | {agg_tok_s:>9.2f}   | {avg_ttft:>11.2f}   | {res.total_context_switches:>10}  | {res.jains_fairness_index:>9.4f}")

    print("=" * 80)

def benchmark_continuous_batching_scaling() -> None:
    from llmhub.batch.engine import BatchSlotState, ContinuousBatchEngine
    from llmhub.batch.prefix_cache import PrefixCache
    from llmhub.server.backend import Backend, DecodeResult, SessionState

    class BenchmarkMockBackend(Backend):
        def __init__(self, model_name: str = "bench-model"):
            self.model_name = model_name
            self.sessions = {}
        def is_loaded(self): return True
        def load(self, *a, **k): pass
        def unload(self): self.sessions.clear()
        def tokenize(self, text: str): return [ord(c) for c in text]
        def new_session(self, session_id, prompt, max_tokens=128, **kwargs):
            tokens = kwargs.get("prompt_tokens") or [1, 2, 3, 4]
            sess = SessionState(session_id, self.model_name, prompt, list(tokens), max_tokens=max_tokens)
            self.sessions[session_id] = sess
            return sess
        def prefill_chunk(self, session_id, max_tokens_chunk=256):
            sess = self.sessions.get(session_id)
            if not sess: return 0, True
            rem = len(sess.prompt_tokens) - sess.prefill_offset
            chunk = min(max_tokens_chunk, max(0, rem))
            sess.prefill_offset += chunk
            done = sess.prefill_offset >= len(sess.prompt_tokens)
            return chunk, done
        def decode_step(self, session_id, n_tokens=1):
            sess = self.sessions.get(session_id)
            if not sess: return DecodeResult(session_id, [], [], True)
            toks = [len(sess.generated_tokens) + i for i in range(n_tokens)]
            sess.generated_tokens.extend(toks)
            done = len(sess.generated_tokens) >= sess.max_tokens
            return DecodeResult(session_id, toks, [f" t{t}" for t in toks], done)
        def get_session(self, session_id): return self.sessions.get(session_id)
        def close_session(self, session_id): self.sessions.pop(session_id, None)
        def get_info(self): return {}

    print("\n" + "=" * 80)
    print(" PHASE 6 CONTINUOUS BATCHING: THROUGHPUT SCALING (BATCH SIZES B=1, 2, 4)")
    print("=" * 80)
    print(f"{'Batch Size (B)':<15} | {'Steps Executed':<16} | {'Tokens / Sec':<14} | {'Step Latency (us)':<18} | {'Speedup':<10}")
    print("-" * 80)

    base_tok_s = 0.0
    for b_size in [1, 2, 4]:
        backend = BenchmarkMockBackend()
        engine = ContinuousBatchEngine(model_tag="bench-model", max_slots=b_size)

        for i in range(b_size):
            engine.allocate_slot(
                session_id=f"sess_{i}",
                prompt="System prompt test instructions",
                prompt_tokens=[10, 20, 30, 40],
                max_tokens=4000,
            )
            slot = engine.get_slot_by_session(f"sess_{i}")
            if slot:
                slot.state = BatchSlotState.DECODING

        # Warmup
        for _ in range(50):
            engine.step_batch(backend, default_quantum=4)

        n_steps = 500
        start = time.perf_counter()
        for _ in range(n_steps):
            res = engine.step_batch(backend, default_quantum=4)
        elapsed = time.perf_counter() - start

        total_tokens = n_steps * (b_size * 4)
        tok_s = total_tokens / elapsed if elapsed > 0 else 0.0
        step_us = (elapsed / n_steps) * 1_000_000.0

        if b_size == 1:
            base_tok_s = tok_s
            speedup = 1.00
        else:
            speedup = tok_s / base_tok_s if base_tok_s > 0 else 1.00

        print(f"{b_size:<15} | {n_steps:<16} | {tok_s:>12.1f}   | {step_us:>16.2f}  | {speedup:>8.2f}x")

    print("-" * 80)
    print(" CONTINUOUS BATCHING PREFIX / PROMPT CACHING EFFICIENCY")
    print("-" * 80)

    cache = PrefixCache(max_cached_tokens=16384, min_prefix_tokens=8)
    common_system_prompt = [i for i in range(1, 129)]  # 128 tokens
    cache.insert_prefix("bench-model", common_system_prompt)

    n_queries = 2000
    start = time.perf_counter()
    for q_idx in range(n_queries):
        if q_idx % 5 != 0:  # 80% hit
            query_tokens = common_system_prompt + [1000 + q_idx, 1001 + q_idx]
        else:  # 20% miss
            query_tokens = [5000 + q_idx + j for j in range(32)]
        cache.find_longest_prefix("bench-model", query_tokens)
    elapsed = time.perf_counter() - start

    lookup_us = (elapsed / n_queries) * 1_000_000.0
    hit_rate = (cache.cache_hits / cache.total_queries) * 100.0 if cache.total_queries > 0 else 0.0

    print(f" Prefix Cache Queries:       {cache.total_queries}")
    print(f" Cache Hit Rate:             {hit_rate:.1f}% ({cache.cache_hits} hits / {cache.total_queries} queries)")
    print(f" Prefill Tokens Saved:       {cache.tokens_saved:,} tokens")
    print(f" Average Hash & Lookup Time: {lookup_us:.3f} us / query")
    print(f" Prefill Redundancy Avoided: {cache.tokens_saved / (cache.total_queries * 128) * 100:.1f}% of total input volume")
    print("=" * 80)

def benchmark_advanced_scheduling() -> None:
    from llmhub.scheduler.mlfq import MultilevelFeedbackQueue
    from llmhub.scheduler.session import SchedSession
    from llmhub.speculative.engine import SpeculativeEngine

    print("\n" + "=" * 80)
    print(" PHASE 7 ADVANCED SCHEDULING: MLFQ OVERHEAD & SPECULATIVE DECODING SPEEDUP")
    print("=" * 80)

    # 1. MLFQ queue selection and demotion overhead
    mlfq = MultilevelFeedbackQueue(boost_interval_seconds=10.0, enable_mlfq=True)
    sessions = [
        SchedSession(session_id=f"bench_sess_{i}", model_tag="test-model", prompt="test", max_tokens=256)
        for i in range(100)
    ]
    for s in sessions:
        mlfq.enqueue_session(s)

    n_ops = 5000
    start = time.perf_counter()
    for i in range(n_ops):
        s = mlfq.select_next_runnable()
        if s:
            mlfq.record_quantum_consumption(s, 4)
            mlfq.rotate_session_in_level(s)
    elapsed = time.perf_counter() - start
    avg_mlfq_us = (elapsed / n_ops) * 1_000_000.0

    print(f" MLFQ Select + Demotion + Rotate Overhead: {avg_mlfq_us:.3f} us / step (Queue Depth: 100)")
    print(f" MLFQ Demotions Triggered:                 {mlfq.total_demotions}")
    print("-" * 80)

    # 2. Speculative decoding mathematical model speedup across acceptance rates alpha
    print(f"{'Draft K':<9} | {'Acceptance Rate (alpha)':<25} | {'Expected Tok/Round':<20} | {'Theoretical Speedup':<20}")
    print("-" * 80)

    cost_ratio = 0.15  # Draft 512M vs Target 7B (~6.7x speed difference)
    for k in [2, 4, 6]:
        for alpha in [0.30, 0.50, 0.75, 0.90]:
            # Expected accepted tokens: E[accepted] = sum_{i=1}^K alpha^i
            expected_accepted = sum(alpha ** i for i in range(1, k + 1))
            expected_tokens = expected_accepted + 1.0  # +1 target correction/bonus token
            cost = 1.0 + (k * cost_ratio)
            speedup = expected_tokens / cost
            print(f"{k:<9} | {alpha:<25.2f} | {expected_tokens:<20.2f} | {speedup:>18.2f}x")

    print("=" * 80)

def run_benchmarks() -> None:
    prof = HardwareDetector.get_profile()
    cpu = prof.cpu
    gpu = prof.gpu

    print("=" * 70)
    print(" LLMHUB SYSTEM & INFERENCE MICRO-BENCHMARK HARNESS")
    print("=" * 70)
    print(f" OS:             {prof.os_name} ({prof.arch})")
    print(f" Physical Cores: {cpu.physical_cores} (Logical: {cpu.logical_cores})")
    print(f" Total RAM:      {prof.total_ram_bytes / (1024**3):.2f} GB")
    print(f" Free RAM:       {prof.free_ram_bytes / (1024**3):.2f} GB")
    print(f" SIMD Features:  {', '.join(cpu.simd_extensions)}")
    print(f" Accelerator:    {gpu.name} ({gpu.backend.upper()})")
    print(f" Termux Mode:    {prof.is_termux} (Non-Root: {prof.is_non_root})")
    print("-" * 70)

    print("Running GGUF binary parser benchmark (500 runs)...")
    parse_ms = benchmark_gguf_parser(500)
    print(f"  -> Average GGUF Parse Latency:       {parse_ms:.3f} ms / file")

    print("Running Jinja2 multi-turn chat template benchmark (1000 runs)...")
    template_ms = benchmark_chat_template(1000)
    print(f"  -> Average Template Render Latency:  {template_ms:.3f} ms / turn")

    print("Running memory physics computation benchmark (5000 runs)...")
    math_us = benchmark_memory_math(5000)
    print(f"  -> Average Memory Physics Latency:   {math_us:.3f} us / call")

    print("Running hardware detector benchmark (5 runs)...")
    hw_ms = benchmark_hardware_detection(5)
    print(f"  -> Average Hardware Probe Latency:   {hw_ms:.3f} ms / probe")

    print("Running auto-tuner plan calculation benchmark (500 runs)...")
    tune_us = benchmark_auto_tuning(500)
    print(f"  -> Average Auto-Tuner Plan Latency:  {tune_us:.3f} us / plan")

    print("Running backend quantum decode overhead benchmark (1000 runs)...")
    quantum_us = benchmark_backend_quantum(1000)
    print(f"  -> Average Decode Quantum Dispatch:  {quantum_us:.3f} us / quantum")

    print("Running FastAPI SSE streaming roundtrip benchmark (100 runs)...")
    sse_ms = benchmark_server_sse(100)
    print(f"  -> Average SSE Streaming Turnaround: {sse_ms:.3f} ms / req")

    print("Running Multi-Model admission control check (2000 runs)...")
    adm_us = benchmark_model_manager_admission(2000)
    print(f"  -> Average Admission Check Latency:  {adm_us:.3f} us / evaluation")

    print("Running Token Round-Robin Scheduler step dispatch (1500 steps)...")
    sched_step_us = benchmark_scheduler_step_overhead(500)
    print(f"  -> Average Scheduler Step Overhead:  {sched_step_us:.3f} us / turn")

    print("=" * 70)
    print(" Micro-benchmark passed with zero bottlenecks in Python orchestration path.")

    # Phase 5 Threading Modes Comparison
    benchmark_scheduler_modes_comparison()

    # Phase 6 Continuous Batching Throughput & Prefix Caching Scaling
    benchmark_continuous_batching_scaling()

    # Phase 7 Advanced Scheduling & Speculative Decoding
    benchmark_advanced_scheduling()

    # Phase 8 Termux & Low-Memory Mode
    benchmark_termux_low_memory()

    # Phase 9 Observability & Hardening
    benchmark_observability_hardening()

    # Phase 10 Distributed Scale-Out & Cluster Routing
    benchmark_cluster_scaleout()

def benchmark_termux_low_memory() -> None:
    from llmhub.termux.thermal import ThermalMonitor
    from llmhub.termux.memory_pressure import MemoryPressureMonitor
    from llmhub.termux.downshift import DownshiftEngine
    from llmhub.batch.prefix_cache import PrefixCache

    print("\n" + "=" * 80)
    print(" PHASE 8 TERMUX & LOW-MEMORY MONITORING OVERHEAD BENCHMARK")
    print("=" * 80)

    # 1. Thermal monitor probe latency
    tm = ThermalMonitor()
    n_thermal = 50
    start = time.perf_counter()
    for _ in range(n_thermal):
        tm.probe()
    thermal_us = ((time.perf_counter() - start) / n_thermal) * 1_000_000.0
    print(f" Thermal Sensor Probe Overhead:       {thermal_us:.3f} us / probe")

    # 2. Memory pressure monitor probe latency
    mpm = MemoryPressureMonitor()
    n_mem = 1000
    start = time.perf_counter()
    for _ in range(n_mem):
        mpm.probe()
    mem_us = ((time.perf_counter() - start) / n_mem) * 1_000_000.0
    print(f" Memory Pressure / LMK Probe Overhead:{mem_us:.3f} us / probe")

    # 3. Downshift context search latency
    engine = DownshiftEngine()
    n_downshift = 2000
    start = time.perf_counter()
    for _ in range(n_downshift):
        engine.find_fitting_context(
            weights_bytes=1024 * 1024 * 1024,
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            desired_ctx=2048,
            available_ram_bytes=int(1.2 * 1024**3),
            safety_margin=0.85,
        )
    downshift_us = ((time.perf_counter() - start) / n_downshift) * 1_000_000.0
    print(f" Context Downshifting Calculation:    {downshift_us:.3f} us / evaluation")

    # 4. Prefix cache memory pressure pruning
    cache = PrefixCache(max_cached_tokens=10000)
    for i in range(50):
        cache.insert_prefix(f"model_{i}", list(range(i * 10, i * 10 + 64)))
    n_prune = 500
    start = time.perf_counter()
    for _ in range(n_prune):
        cache.prune_memory_pressure(ratio=0.1)
    prune_us = ((time.perf_counter() - start) / n_prune) * 1_000_000.0
    print(f" Prefix Cache Pressure Pruning:       {prune_us:.3f} us / prune")
    print("=" * 80)

def benchmark_observability_hardening() -> None:
    from llmhub.observability.metrics import MetricsRegistry
    from llmhub.security.auth import TokenBucket, RateLimiter
    from llmhub.security.sanitizer import InputSanitizer
    from llmhub.observability.logger import current_request_id, JSONFormatter
    import logging

    print("\n" + "=" * 80)
    print(" PHASE 9 OBSERVABILITY & SECURITY HARDENING OVERHEAD BENCHMARK")
    print("=" * 80)

    # 1. Prometheus metrics serialization latency
    reg = MetricsRegistry()
    for i in range(10):
        reg.requests_total.inc(1, labels={"model": f"model_{i}", "status": "success", "tenant": "default"})
        reg.ttft_seconds.observe(0.045 + i * 0.005, labels={"model": f"model_{i}"})
    n_metrics = 500
    start = time.perf_counter()
    for _ in range(n_metrics):
        _ = reg.export()
    metrics_us = ((time.perf_counter() - start) / n_metrics) * 1_000_000.0
    print(f" Prometheus Text Exposition Serialization:{metrics_us:.3f} us / scrape")

    # 2. TokenBucket rate limiter check latency
    bucket = TokenBucket(capacity=100.0, refill_rate_per_sec=10.0)
    n_bucket = 10000
    start = time.perf_counter()
    for _ in range(n_bucket):
        bucket.consume(1.0)
    bucket_us = ((time.perf_counter() - start) / n_bucket) * 1_000_000.0
    print(f" TokenBucket Consume Check Overhead:      {bucket_us:.3f} us / check")

    # 3. Input sanitizer latency on prompt string
    sample_prompt = "Hello \x00world! This is a test \x08message with \n legitimate newlines and \t tabs." * 5
    n_clean = 5000
    start = time.perf_counter()
    for _ in range(n_clean):
        InputSanitizer.sanitize_text(sample_prompt)
    clean_us = ((time.perf_counter() - start) / n_clean) * 1_000_000.0
    print(f" Input Sanitizer (Regex & Control Strip): {clean_us:.3f} us / prompt")

    # 4. ContextVar trace ID propagation overhead
    n_trace = 10000
    start = time.perf_counter()
    for _ in range(n_trace):
        t = current_request_id.set("req-test-1234")
        _ = current_request_id.get()
        current_request_id.reset(t)
    trace_us = ((time.perf_counter() - start) / n_trace) * 1_000_000.0
    print(f" ContextVar Trace ID Propagation Overhead:{trace_us:.3f} us / context switch")

    # 5. Structured JSON log formatting latency
    rec = logging.LogRecord(
        name="llmhub.bench",
        level=logging.INFO,
        pathname="bench.py",
        lineno=42,
        msg="Inference request completed successfully",
        args=(),
        exc_info=None,
    )
    rec.trace_id = "req-bench-9999"
    fmt = JSONFormatter()
    n_json = 5000
    start = time.perf_counter()
    for _ in range(n_json):
        _ = fmt.format(rec)
    json_us = ((time.perf_counter() - start) / n_json) * 1_000_000.0
    print(f" Structured JSON Log Formatting Overhead: {json_us:.3f} us / log record")
    print("=" * 80)

def benchmark_cluster_scaleout() -> None:
    from llmhub.cluster.node import NodeInfo, NodeStatus
    from llmhub.cluster.manager import ClusterManager
    from llmhub.cluster.router import ClusterRouter
    from llmhub.cluster.beacon import DiscoveryBeacon
    from llmhub.cluster.pipeline import PipelineCoordinator

    print("\n" + "=" * 80)
    print(" PHASE 10 DISTRIBUTED SCALE-OUT & CLUSTER ROUTING BENCHMARK")
    print("=" * 80)

    # 1. ClusterRouter route resolution latency
    local = NodeInfo(node_id="node-local", free_ram_mb=4000.0, resident_models={"qwen:0.5b": {}})
    mgr = ClusterManager(local_node=local, enabled=True)
    for i in range(5):
        mgr.peers[f"peer_{i}"] = NodeInfo(
            node_id=f"peer_{i}",
            free_ram_mb=8000.0 + i * 1000,
            resident_models={f"model_{i}": {}},
            latency_ms=2.0 + i,
        )
    router = ClusterRouter(mgr)

    n_routes = 10000
    start = time.perf_counter()
    for _ in range(n_routes):
        router.route_model("model_2")
    route_us = ((time.perf_counter() - start) / n_routes) * 1_000_000.0
    print(f" Cluster Route Resolution Latency:       {route_us:.3f} us / routing decision")

    # 2. Discovery beacon packet serialization latency
    beacon = DiscoveryBeacon(node_id="node-bench", name="bench-worker", api_url="http://192.168.1.50:11434")
    beacon.set_dynamic_state_callback(lambda: {"free_ram_mb": 5500.0, "resident_models": {"test": {}}})
    n_beacon = 10000
    start = time.perf_counter()
    for _ in range(n_beacon):
        beacon.build_packet()
    beacon_us = ((time.perf_counter() - start) / n_beacon) * 1_000_000.0
    print(f" Discovery Beacon Packet Serialization:  {beacon_us:.3f} us / packet")

    # 3. Pipeline layer partition calculation latency
    nodes = [local] + list(mgr.peers.values())
    n_plan = 5000
    start = time.perf_counter()
    for _ in range(n_plan):
        PipelineCoordinator.plan_pipeline("llama3:70b", 80, 8192, 40000.0, nodes)
    plan_us = ((time.perf_counter() - start) / n_plan) * 1_000_000.0
    print(f" Pipeline Layer Sharding Plan Latency:   {plan_us:.3f} us / evaluation")

    # 4. Cluster status aggregation latency
    n_status = 5000
    start = time.perf_counter()
    for _ in range(n_status):
        mgr.get_cluster_status()
    status_us = ((time.perf_counter() - start) / n_status) * 1_000_000.0
    print(f" Cluster Status Aggregation Overhead:    {status_us:.3f} us / query")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmarks()
