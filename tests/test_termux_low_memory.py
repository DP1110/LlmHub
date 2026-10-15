"""Tests for Phase 8: Termux, Android non-root sandbox, wakelock, thermals, and low-memory mode."""

import asyncio
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from llmhub.batch.prefix_cache import PrefixCache
from llmhub.config import LLMHubConfig, is_termux
from llmhub.hardware.detector import HardwareDetector
from llmhub.scheduler.policy import ThreadingMode, ThreadingStrategy
from llmhub.scheduler.scheduler import TokenRoundRobinScheduler
from llmhub.server.app import create_app, state
from llmhub.server.backend import Backend, DecodeResult, SessionState
from llmhub.server.model_manager import ModelManager
from llmhub.termux.controller import TermuxController
from llmhub.termux.downshift import DownshiftEngine
from llmhub.termux.memory_pressure import MemoryPressureLevel, MemoryPressureMonitor, MemoryPressureStats
from llmhub.termux.thermal import ThermalMonitor, ThermalReading, ThermalState, ThermalZoneInfo
from llmhub.termux.wakelock import WakelockManager

class MockBackend(Backend):
    def __init__(self) -> None:
        self.loaded = False
        self.n_threads = 4
        self.closed_sessions: List[str] = []

    def load(self, model_path: Path, n_ctx: int, n_threads: int, use_mmap: bool = True, use_mlock: bool = False) -> None:
        self.loaded = True
        self.n_threads = n_threads

    def unload(self) -> None:
        self.loaded = False

    def new_session(self, session_id: str, prompt_tokens: List[int]) -> SessionState:
        return SessionState(session_id=session_id, prompt_tokens=prompt_tokens)

    def decode_step(self, session: SessionState, n_tokens: int = 1) -> DecodeResult:
        token = 100 + len(session.generated_tokens)
        session.generated_tokens.append(token)
        return DecodeResult(
            session_id=session.session_id,
            tokens=[token],
            text_deltas=[f"t{token}"],
            done=len(session.generated_tokens) >= 5,
        )

    def prefill_chunk(self, session: SessionState, chunk_tokens: List[int]) -> int:
        session.prefilled_tokens += len(chunk_tokens)
        return len(chunk_tokens)

    def set_n_threads(self, n_threads: int) -> None:
        self.n_threads = n_threads

    def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)

def test_termux_environment_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Termux detection via environment variable and filesystem markers."""
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    # Default without env or path
    with patch("os.path.exists", return_value=False):
        assert not is_termux()

    # With TERMUX_VERSION
    monkeypatch.setenv("TERMUX_VERSION", "0.118.0")
    assert is_termux()

    # With Termux path marker
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    with patch("os.path.exists", side_effect=lambda p: str(p) == "/data/data/com.termux"):
        assert is_termux()

def test_wakelock_lifecycle_mocked() -> None:
    """Verify acquire, release, idempotency, and context manager semantics."""
    executed_commands: List[List[str]] = []

    def mock_runner(cmd: List[str]) -> int:
        executed_commands.append(cmd)
        return 0

    wm = WakelockManager(command_runner=mock_runner, force_supported=True)
    assert wm.is_supported
    assert not wm.is_held

    # Acquire
    assert wm.acquire()
    assert wm.is_held
    assert len(executed_commands) == 1

    # Idempotent acquire (should not re-execute command)
    assert wm.acquire()
    assert len(executed_commands) == 1

    # Release
    assert wm.release()
    assert not wm.is_held
    assert len(executed_commands) == 2

    # Idempotent release
    assert wm.release()
    assert len(executed_commands) == 2

    # Context manager
    with wm:
        assert wm.is_held
    assert not wm.is_held

def test_wakelock_async_context_manager() -> None:
    """Verify async context manager behavior."""
    async def _run() -> None:
        executed: List[List[str]] = []
        wm = WakelockManager(command_runner=lambda cmd: executed.append(cmd) or 0, force_supported=True)

        async with wm:
            assert wm.is_held
        assert not wm.is_held

    asyncio.run(_run())

def test_wakelock_unsupported_fallback() -> None:
    """Verify graceful no-op on non-Termux systems."""
    wm = WakelockManager(force_supported=False)
    assert not wm.is_supported
    assert not wm.acquire()
    assert not wm.is_held
    assert wm.release()

def test_thermal_monitor_state_classification() -> None:
    """Verify thermal temperature thresholds and state classification."""
    tm = ThermalMonitor(warning_temp_c=75.0, critical_temp_c=85.0, warm_temp_c=65.0)

    # 50C -> NORMAL
    tm.inject_temperature(50.0)
    assert tm.probe().state == ThermalState.NORMAL
    assert tm.calculate_target_threads(4, ThermalState.NORMAL) == 4

    # 68C -> WARM
    tm.inject_temperature(68.0)
    assert tm.probe().state == ThermalState.WARM
    assert tm.calculate_target_threads(4, ThermalState.WARM) == 3

    # 78C -> HOT (50% thread reduction)
    tm.inject_temperature(78.0)
    assert tm.probe().state == ThermalState.HOT
    assert tm.calculate_target_threads(4, ThermalState.HOT) == 2

    # 88C -> CRITICAL (1 thread limit)
    tm.inject_temperature(88.0)
    assert tm.probe().state == ThermalState.CRITICAL
    assert tm.calculate_target_threads(4, ThermalState.CRITICAL) == 1

def test_thermal_monitor_hysteresis() -> None:
    """Verify hysteresis algorithm prevents rapid oscillation between thermal states."""
    tm = ThermalMonitor(
        warning_temp_c=75.0,
        critical_temp_c=85.0,
        warm_temp_c=65.0,
        cooldown_margin_c=3.0,
        consecutive_cool_required=2,
    )

    # Escalate to HOT
    tm.inject_temperature(80.0)
    reading, threads = tm.update_and_get_target_threads(4)
    assert tm.current_state == ThermalState.HOT
    assert threads == 2

    # Temperature drops to 73C (below 75C warning, but above 75 - 3 = 72C cooldown threshold)
    tm.inject_temperature(73.0)
    reading, threads = tm.update_and_get_target_threads(4)
    # Hysteresis: stays in HOT because it hasn't reached safe cooldown margin
    assert tm.current_state == ThermalState.HOT

    # Temperature drops to 70C (below 72C, first cool reading)
    tm.inject_temperature(70.0)
    reading, threads = tm.update_and_get_target_threads(4)
    # Still HOT (requires 2 consecutive cool readings)
    assert tm.current_state == ThermalState.HOT

    # Second consecutive cool reading
    tm.inject_temperature(70.0)
    reading, threads = tm.update_and_get_target_threads(4)
    # Now cooled down to WARM
    assert tm.current_state == ThermalState.WARM

    # WN-1: A spike above cooldown threshold immediately resets cool counter
    tm.inject_temperature(60.0)  # 1st cool towards NORMAL
    tm.update_and_get_target_threads(4)
    assert tm._cool_reading_count == 1
    tm.inject_temperature(74.0)  # Temperature spike back to WARM
    tm.update_and_get_target_threads(4)
    assert tm._cool_reading_count == 0

def test_memory_pressure_monitor_classification() -> None:
    """Verify memory pressure classification and LMK risk detection."""
    mpm = MemoryPressureMonitor()

    # Nominal: 8GB total, 4GB available
    mpm.inject_stats(
        MemoryPressureStats(
            level=MemoryPressureLevel.NOMINAL,
            total_ram_bytes=8 * 1024**3,
            available_ram_bytes=4 * 1024**3,
            free_ram_bytes=3 * 1024**3,
            available_percent=50.0,
            oom_score_adj=0,
        )
    )
    assert mpm.probe().level == MemoryPressureLevel.NOMINAL

    # Moderate: 4GB total, 800MB available
    mpm.inject_stats(
        MemoryPressureStats(
            level=MemoryPressureLevel.MODERATE,
            total_ram_bytes=4 * 1024**3,
            available_ram_bytes=800 * 1024**2,
            free_ram_bytes=400 * 1024**2,
            available_percent=19.5,
            oom_score_adj=200,
        )
    )
    assert mpm.probe().level == MemoryPressureLevel.MODERATE

    # Critical: 4GB total, 300MB available
    mpm.inject_stats(
        MemoryPressureStats(
            level=MemoryPressureLevel.CRITICAL,
            total_ram_bytes=4 * 1024**3,
            available_ram_bytes=300 * 1024**2,
            free_ram_bytes=100 * 1024**2,
            available_percent=7.3,
            oom_score_adj=905,
        )
    )
    assert mpm.probe().level == MemoryPressureLevel.CRITICAL

def test_context_downshifting_under_pressure() -> None:
    """Verify dynamic context clamping given memory pressure levels."""
    engine = DownshiftEngine()

    # Nominal keeps 2048
    assert engine.adjust_context_for_pressure(2048, MemoryPressureLevel.NOMINAL) == 2048

    # Moderate clamps to 1024
    assert engine.adjust_context_for_pressure(2048, MemoryPressureLevel.MODERATE) == 1024
    assert engine.adjust_context_for_pressure(512, MemoryPressureLevel.MODERATE) == 512

    # Critical clamps to 512
    assert engine.adjust_context_for_pressure(2048, MemoryPressureLevel.CRITICAL) == 512

    # CR-1: When requested context is smaller than min_ctx (e.g. 256), never upshift!
    assert engine.adjust_context_for_pressure(256, MemoryPressureLevel.CRITICAL) == 256
    assert engine.adjust_context_for_pressure(256, MemoryPressureLevel.MODERATE) == 256

    # Low-memory mode flag clamps to 1024
    assert engine.adjust_context_for_pressure(4096, MemoryPressureLevel.NOMINAL, low_memory_mode=True) == 1024

def test_find_fitting_context_algorithm() -> None:
    """Verify find_fitting_context selects the largest viable context window."""
    engine = DownshiftEngine()

    # 1 GB model weights with 1.2 GB usable RAM
    # 2048 context ~256 MB KV cache -> 1.0 + 0.256 + 0.064 = 1.32 GB (shortfall)
    # 1024 context ~128 MB KV cache -> 1.0 + 0.128 + 0.064 = 1.192 GB (fits!)
    fitting_ctx, rationale = engine.find_fitting_context(
        weights_bytes=1024 * 1024 * 1024,
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        desired_ctx=2048,
        available_ram_bytes=int(1.25 * 1024**3),
        safety_margin=0.95,
        min_ctx=512,
    )
    assert fitting_ctx is not None
    assert fitting_ctx <= 1024

    # Extremely constrained RAM: cannot fit even 512 context
    impossible_ctx, impossible_rationale = engine.find_fitting_context(
        weights_bytes=1024 * 1024 * 1024,
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        desired_ctx=2048,
        available_ram_bytes=500 * 1024 * 1024,  # only 500 MB RAM for 1GB weights
        safety_margin=0.85,
    )
    assert impossible_ctx is None

def test_quant_downshifting_suggestions() -> None:
    """Verify quantization suggestions scale down from Q8 to Q2 under RAM constraints."""
    engine = DownshiftEngine()

    # 7B model on 4GB device (available ~3.5GB)
    res = engine.suggest_downshifted_quant(
        param_count_billions=7.0,
        available_ram_bytes=int(3.5 * 1024**3),
        safety_margin=0.85,
    )
    assert res is not None
    quant, bpw, desc = res
    # Q4_K_M (4.5 bpw * 7B / 8 = 3.93GB > 2.97GB limit), so should pick Q3_K_M or Q2_K
    assert quant in ("Q3_K_M", "Q2_K")

    # 70B model on 4GB device cannot fit any quantization
    impossible = engine.suggest_downshifted_quant(
        param_count_billions=70.0,
        available_ram_bytes=int(4.0 * 1024**3),
    )
    assert impossible is None

def test_prefix_cache_pressure_pruning() -> None:
    """Verify prefix cache shedding entries under memory pressure."""
    cache = PrefixCache(max_cached_tokens=4096)
    # Insert 10 entries of 64 tokens each = 640 tokens
    for i in range(10):
        tokens = list(range(i * 100, i * 100 + 64))
        cache.insert_prefix(f"model_{i}", tokens)

    initial_tokens = cache.get_stats()["cached_tokens"]
    assert initial_tokens > 0

    # Prune 50%
    evicted = cache.prune_memory_pressure(ratio=0.5)
    assert evicted > 0
    after_tokens = cache.get_stats()["cached_tokens"]
    assert after_tokens <= initial_tokens // 2

def test_threading_strategy_thermal_cap() -> None:
    """Verify ThreadingStrategy caps thread allocation when max_threads_cap is set."""
    ts = ThreadingStrategy(mode=ThreadingMode.TIME_SLICE)
    base_cores = ts.total_physical_cores

    allocs = ts.calculate_threads({"model_a"})
    assert allocs["model_a"] == base_cores

    # Throttle due to thermals
    ts.set_max_threads_cap(2)
    allocs_throttled = ts.calculate_threads({"model_a"})
    assert allocs_throttled["model_a"] == min(base_cores, 2)

    # Reset
    ts.set_max_threads_cap(None)
    assert ts.calculate_threads({"model_a"})["model_a"] == base_cores

def test_termux_controller_tick_and_status() -> None:
    """Verify TermuxController tick updates status and invokes thread callback."""
    config = LLMHubConfig()
    controller = TermuxController(config=config, base_threads=4)

    adjusted_threads: List[int] = []
    controller.set_thread_callback(lambda t: adjusted_threads.append(t))

    # Inject HOT temperature into thermal monitor
    controller.thermal.inject_temperature(80.0)
    status = controller.tick()

    assert status["thermal"]["state"] == "hot"
    assert status["thermal"]["effective_threads"] == 2
    assert 2 in adjusted_threads

    assert "wakelock" in status
    assert "memory_pressure" in status

def test_api_termux_endpoints() -> None:
    """Verify /api/termux/status, wakelock, and /api/system/low-memory endpoints."""
    app = create_app()
    client = TestClient(app)

    # 1. Healthz includes termux info
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.json()
    assert "termux" in data
    assert "thermal_state" in data["termux"]
    assert "memory_pressure" in data["termux"]

    # 2. /api/termux/status
    res = client.get("/api/termux/status")
    assert res.status_code == 200
    st = res.json()
    assert "wakelock" in st
    assert "thermal" in st
    assert "memory_pressure" in st

    # 3. /api/termux/wakelock/acquire and release
    res_acq = client.post("/api/termux/wakelock/acquire")
    assert res_acq.status_code == 200

    res_rel = client.post("/api/termux/wakelock/release")
    assert res_rel.status_code == 200

    # 4. /api/system/low-memory toggle
    res_lm = client.post("/api/system/low-memory?enabled=true")
    assert res_lm.status_code == 200
    assert res_lm.json()["low_memory_mode"] is True


def test_core_lazy_load_without_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure llmhub.core imports without requiring llama_cpp."""
    import sys
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    import llmhub.core as core
    # HFDownloader and others should be accessible
    assert hasattr(core, "HFDownloader")
    assert hasattr(core, "GGUFParser")
    # ModelRunner should be lazy-loaded on access
    runner_cls = core.ModelRunner
    assert runner_cls is not None


def test_model_runner_error_when_neither_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelRunner should raise clear error when neither llama_cpp nor llama-cli exists."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod

    dummy_gguf = tmp_path / "test.gguf"
    # Write a minimal dummy file
    dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)

    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)

    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    # Patch parser so it doesn't fail on dummy bytes
    with patch("llmhub.core.runner.GGUFParser.parse_file") as mock_parse:
        mock_meta = MagicMock()
        mock_meta.chat_template = ""
        mock_meta.architecture = "llama"
        mock_parse.return_value = mock_meta

        with pytest.raises(RuntimeError, match="Neither 'llama-cpp-python' nor native 'llama-cli'"):
            ModelRunner(model_path=dummy_gguf)


def test_model_runner_cli_fallback_when_llama_cpp_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelRunner should successfully bind to llama-cli when llama_cpp is absent."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod

    dummy_gguf = tmp_path / "test.gguf"
    dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)

    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)

    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli" if cmd == "llama-cli" else None)

    with patch("llmhub.core.runner.GGUFParser.parse_file") as mock_parse:
        mock_meta = MagicMock()
        mock_meta.chat_template = ""
        mock_meta.architecture = "llama"
        mock_meta.block_count = 12
        mock_meta.head_count_kv = 4
        mock_meta.head_dim = 64
        mock_parse.return_value = mock_meta

        runner = ModelRunner(model_path=dummy_gguf)
        assert runner.llm is None
        assert runner._cli_binary == "/usr/bin/llama-cli"

        # Test stream_completion with mocked subprocess
        import io
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("Hello world!\n")
            mock_proc.poll.return_value = 0
            mock_popen.return_value = mock_proc

            chunks = list(runner.stream_completion(prompt="Write a greeting"))
            assert len(chunks) >= 2
            # Verify stream tokens reconstructed full text
            full_text = "".join(c["token"] for c in chunks)
            assert full_text == "Hello world!\n"
            assert chunks[-1]["done"] is True


def test_core_lazy_load_dir_introspection() -> None:
    """Ensure dir(llmhub.core) and inspect include ModelRunner per PEP 562."""
    import llmhub.core as core
    dir_entries = dir(core)
    assert "ModelRunner" in dir_entries
    assert "ModelRunner" in core.__all__


def test_model_runner_cli_generator_early_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure subprocess is terminated and handles cleaned up when generator is closed early."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod
    import shutil
    import io

    dummy = tmp_path / "test.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    with patch("llmhub.core.runner.GGUFParser.parse_file"):
        runner = ModelRunner(model_path=dummy)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("Long generation text that will be interrupted...")
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            gen = runner.stream_completion(prompt="Hello")
            first_chunk = next(gen)
            assert len(first_chunk["token"]) > 0

            # Simulate early client disconnect / generator close
            gen.close()

            # Verify subprocess cleanup was invoked
            assert mock_proc.terminate.called or mock_proc.kill.called


def test_runner_cli_flags_and_prompt_echo_stripping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure -st flag is passed to llama-cli and echoed prompt is automatically stripped."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod
    import shutil
    import io

    dummy = tmp_path / "test_echo.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    with patch("llmhub.core.runner.GGUFParser.parse_file"):
        runner = ModelRunner(model_path=dummy)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            # Simulate llama-cli echoing prompt first before generating response
            mock_proc.stdout = io.StringIO("What is AI?\nAI is artificial intelligence.")
            mock_proc.stderr = io.StringIO("")
            mock_proc.poll.return_value = 0
            mock_popen.return_value = mock_proc

            chunks = list(runner.stream_completion(prompt="What is AI?"))
            # The prompt "What is AI?" should have been stripped, leaving only the generated response
            full_text = "".join(c["token"] for c in chunks)
            assert "What is AI?" not in full_text
            assert "AI is artificial intelligence." in full_text
            assert chunks[-1]["done"] is True

            # Verify -st flag was passed in command
            called_cmd = mock_popen.call_args[0][0]
            assert "-st" in called_cmd or "--single-turn" in called_cmd


def test_runner_cli_error_surfacing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure non-zero exit code with 0 tokens raises descriptive RuntimeError with stderr."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod
    import shutil
    import io

    dummy = tmp_path / "test_err.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    with patch("llmhub.core.runner.GGUFParser.parse_file"):
        runner = ModelRunner(model_path=dummy)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("")
            mock_proc.stderr = io.StringIO("error: failed to load model weights: out of memory\n")
            mock_proc.poll.return_value = 1
            mock_proc.returncode = 1
            mock_popen.return_value = mock_proc

            with pytest.raises(RuntimeError, match="out of memory"):
                list(runner.stream_completion(prompt="Hello"))


def test_llamacpp_backend_cli_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LlamaCppBackend falls back to native llama-cli and decodes successfully."""
    from llmhub.server.llamacpp import LlamaCppBackend
    import llmhub.server.llamacpp as llamacpp_mod
    import shutil
    import io

    dummy = tmp_path / "model.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(llamacpp_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(llamacpp_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    with patch("llmhub.server.llamacpp.GGUFParser.parse_file") as mock_parse:
        mock_meta = MagicMock()
        mock_meta.model_name = "test-model"
        mock_meta.architecture = "llama"
        mock_meta.block_count = 16
        mock_meta.head_count_kv = 4
        mock_meta.head_dim = 64
        mock_parse.return_value = mock_meta

        backend = LlamaCppBackend()
        backend.load(dummy, n_ctx=1024, n_threads=4)
        assert backend.is_loaded()
        assert backend._cli_binary == "/usr/bin/llama-cli"

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("Hello from Termux native runtime!")
            mock_proc.stderr = io.StringIO("")
            mock_proc.poll.return_value = 0
            mock_popen.return_value = mock_proc

            sess = backend.new_session("sess-1", prompt="Greet me")
            assert sess.session_id == "sess-1"

            # Execute decode steps
            res1 = backend.decode_step("sess-1", n_tokens=2)
            assert len(res1.text_deltas) >= 1

            # Drain remainder
            res2 = backend.decode_step("sess-1", n_tokens=10)
            assert "Hello from Termux native runtime!" in sess.generated_text

            backend.close_session("sess-1")
            assert "sess-1" not in backend.sessions
            backend.unload()
            assert not backend.is_loaded()


def test_llamacpp_backend_cli_error_surfacing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LlamaCppBackend raises RuntimeError when llama-cli exits non-zero with 0 tokens."""
    from llmhub.server.llamacpp import LlamaCppBackend
    import llmhub.server.llamacpp as llamacpp_mod
    import shutil
    import io

    dummy = tmp_path / "model_fail.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(llamacpp_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(llamacpp_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    with patch("llmhub.server.llamacpp.GGUFParser.parse_file"):
        backend = LlamaCppBackend()
        backend.load(dummy, n_ctx=1024, n_threads=4)

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("")
            mock_proc.stderr = io.StringIO("CANNOT LINK EXECUTABLE 'llama-cli': library 'libopenblas.so' not found\n")
            mock_proc.poll.return_value = 1
            mock_proc.returncode = 1
            mock_popen.return_value = mock_proc

            sess = backend.new_session("sess-fail", prompt="Hello")

            with pytest.raises(RuntimeError, match="libopenblas.so"):
                backend.decode_step("sess-fail", n_tokens=1)


def test_runner_cli_partial_prompt_match_at_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure partial prompt prefix match at EOF is flushed and not lost."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod
    import shutil
    import io

    dummy = tmp_path / "test_partial.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    # Invalidate CLI flags cache to test dynamic lookup without --no-display-prompt
    runner_mod._CLI_FLAGS_CACHE.clear()

    with patch("llmhub.core.runner.GGUFParser.parse_file"):
        runner = ModelRunner(model_path=dummy)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            # Prompt is "Long prompt text", output starts matching "Long" but terminates
            mock_proc.stdout = io.StringIO("Long")
            mock_proc.stderr = io.StringIO("")
            mock_proc.poll.return_value = 0
            mock_popen.return_value = mock_proc

            with patch("subprocess.run") as mock_run:
                # Simulate --help output that does not include --no-display-prompt
                mock_run.return_value = MagicMock(stdout="-st, --single-turn", stderr="")

                chunks = list(runner.stream_completion(prompt="Long prompt text"))
                full_text = "".join(c["token"] for c in chunks)
                # "Long" was not full prompt, so it should be preserved
                assert "Long" in full_text


def test_runner_cli_reverse_prompt_stop_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure stop sequences are passed via -r and stop execution early."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod
    import shutil
    import io

    dummy = tmp_path / "test_stop.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    runner_mod._CLI_FLAGS_CACHE["/usr/bin/llama-cli"] = {"-st", "-r", "--no-display-prompt"}

    with patch("llmhub.core.runner.GGUFParser.parse_file"):
        runner = ModelRunner(model_path=dummy)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("Here is the answer.<|im_end|>More ignored text")
            mock_proc.stderr = io.StringIO("")
            mock_proc.poll.return_value = 0
            mock_popen.return_value = mock_proc

            chunks = list(runner.stream_completion(prompt="Question", stop=["<|im_end|>"]))
            full_text = "".join(c["token"] for c in chunks)
            assert "Here is the answer." in full_text
            assert "More ignored text" not in full_text

            # Check -r was passed
            called_cmd = mock_popen.call_args[0][0]
            assert "-r" in called_cmd


def test_runner_cli_chat_template_stop_token_not_triggered_by_prompt_echo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure stop tokens inside the echoed prompt (like <|im_end|>) do NOT abort generation."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod
    import shutil
    import io

    dummy = tmp_path / "test_chat.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    runner_mod._CLI_FLAGS_CACHE["/usr/bin/llama-cli"] = {"-st"}

    prompt = "<|im_start|>system\nYou are an AI.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    stream_output = (
        "\n<|im_start|>system\nYou are an AI.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        "Hello there! How can I help you today?<|im_end|>Extra garbage"
    )

    with patch("llmhub.core.runner.GGUFParser.parse_file"):
        runner = ModelRunner(model_path=dummy)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO(stream_output)
            mock_proc.stderr = io.StringIO("")
            mock_proc.poll.return_value = 0
            mock_popen.return_value = mock_proc

            chunks = list(runner.stream_completion(prompt=prompt, stop=["<|im_end|>"]))
            full_text = "".join(c["token"] for c in chunks)

            assert "<|im_start|>" not in full_text
            assert "<|im_end|>" not in full_text
            assert "Hello there! How can I help you today?" in full_text
            assert "Extra garbage" not in full_text
            assert chunks[-1]["done"] is True


def test_runner_cli_model_response_overlapping_prompt_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure response words overlapping with prompt prefix are never truncated (CRIT-1 fix)."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod
    import shutil
    import io

    dummy = tmp_path / "test_overlap.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    runner_mod._CLI_FLAGS_CACHE["/usr/bin/llama-cli"] = {"-st"}

    prompt = "Please explain quantum computing in simple terms"
    # Response begins with "Please find", overlapping first 7 characters with prompt
    stream_output = "Please find below an explanation of quantum computing in simple terms."

    with patch("llmhub.core.runner.GGUFParser.parse_file"):
        runner = ModelRunner(model_path=dummy)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO(stream_output)
            mock_proc.stderr = io.StringIO("")
            mock_proc.poll.return_value = 0
            mock_popen.return_value = mock_proc

            chunks = list(runner.stream_completion(prompt=prompt))
            full_text = "".join(c["token"] for c in chunks)

            assert "Please find below an explanation of quantum computing in simple terms." in full_text


def test_runner_cli_mid_generation_sigkill_surfaces_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mid-stream SIGKILL (-9) is not swallowed as normal completion (CRIT-2 fix)."""
    from llmhub.core.runner import ModelRunner
    import llmhub.core.runner as runner_mod
    import shutil
    import io

    dummy = tmp_path / "test_sigkill.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    monkeypatch.setattr(runner_mod, "HAS_LLAMA_CPP", False)
    monkeypatch.setattr(runner_mod, "llama_cpp", None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/llama-cli")

    runner_mod._CLI_FLAGS_CACHE["/usr/bin/llama-cli"] = {"-st"}

    with patch("llmhub.core.runner.GGUFParser.parse_file"):
        runner = ModelRunner(model_path=dummy)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("Some valid words generated before being killed")
            mock_proc.stderr = io.StringIO("Out of memory: Kill process 12345\n")
            mock_proc.poll.return_value = -9
            mock_proc.returncode = -9
            mock_popen.return_value = mock_proc

            with pytest.raises(RuntimeError, match="Android Low Memory Killer OOM"):
                list(runner.stream_completion(prompt="Generate a long essay"))





