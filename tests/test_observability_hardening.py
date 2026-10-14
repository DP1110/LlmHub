"""Unit and integration tests for Phase 9: Observability and Hardening.

Covers:
- Prometheus Metrics Exporter (Counter, Gauge, Histogram, MetricsRegistry, exposition formatting)
- Structured JSON Logging, contextvars trace ID propagation, RotatingFileHandler
- Crash recovery, PID management, stale lock detection, run-dir cleanup
- Graceful model unloading and request draining (ModelDrainingError, drain_event)
- Bearer token authentication and TokenBucket rate limiter (RPM/TPM, headers)
- Input sanitization (null byte stripping, control codes, chat schema, prompt bounds)
- Strict configuration schema validation
- FastAPI server middleware integration (/metrics, X-Request-ID, 401, 413, 429, 503)
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import pytest
from starlette.testclient import TestClient

from llmhub.config import LLMHubConfig, validate_config_schema, ConfigValidationError
from llmhub.core.lifecycle import PIDManager, ServerAlreadyRunningError, is_process_alive
from llmhub.observability.logger import (
    current_request_id,
    get_logger,
    setup_logging,
    JSONFormatter,
    RequestContextFilter,
)
from llmhub.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    get_metrics_registry,
)
from llmhub.security.auth import APIKeyAuthenticator, RateLimiter, TokenBucket
from llmhub.security.sanitizer import (
    InputSanitizer,
    PayloadTooLargeError,
    ValidationError,
)
from llmhub.server.app import create_app, state
from llmhub.server.backend import Backend, DecodeResult, SessionState
from llmhub.server.model_manager import LoadedModel, ModelDrainingError, ModelManager


# ============================================================================
# 1. Prometheus Metrics Exporter Tests (US-045)
# ============================================================================

def test_prometheus_counter():
    c = Counter("test_counter_total", "Test counter metric")
    assert c.get() == 0.0

    c.inc(1.0)
    assert c.get() == 1.0

    c.inc(2.5, labels={"model": "llama", "status": "success"})
    assert c.get(labels={"model": "llama", "status": "success"}) == 2.5

    with pytest.raises(ValueError):
        c.inc(-1.0)

    exposition = c.generate_exposition()
    assert "# HELP test_counter_total Test counter metric" in exposition
    assert "# TYPE test_counter_total counter" in exposition
    assert any('test_counter_total{model="llama",status="success"} 2.5' in line for line in exposition)


def test_prometheus_gauge():
    g = Gauge("test_gauge", "Test gauge metric")
    assert g.get() == 0.0

    g.set(42.5)
    assert g.get() == 42.5

    g.inc(5.0)
    assert g.get() == 47.5

    g.dec(10.0)
    assert g.get() == 37.5

    g.set(100.0, labels={"zone": "cpu"})
    assert g.get(labels={"zone": "cpu"}) == 100.0

    exposition = g.generate_exposition()
    assert "# HELP test_gauge Test gauge metric" in exposition
    assert "# TYPE test_gauge gauge" in exposition
    assert any('test_gauge{zone="cpu"} 100.0' in line for line in exposition)


def test_prometheus_histogram():
    h = Histogram("test_latency_seconds", "Test histogram metric", buckets=(0.05, 0.1, 0.5, 1.0))
    h.observe(0.04)
    h.observe(0.08)
    h.observe(0.6)
    h.observe(2.5)

    exposition = h.generate_exposition()
    assert "# HELP test_latency_seconds Test histogram metric" in exposition
    assert "# TYPE test_latency_seconds histogram" in exposition
    # Check bucket accumulation
    assert any('test_latency_seconds_bucket{le="0.05"} 1' in line for line in exposition)
    assert any('test_latency_seconds_bucket{le="0.1"} 2' in line for line in exposition)
    assert any('test_latency_seconds_bucket{le="0.5"} 2' in line for line in exposition)
    assert any('test_latency_seconds_bucket{le="1.0"} 3' in line for line in exposition)
    assert any('test_latency_seconds_bucket{le="+Inf"} 4' in line for line in exposition)
    assert any('test_latency_seconds_count 4' in line for line in exposition)


def test_metrics_registry_and_export():
    reg = MetricsRegistry()
    reg.requests_total.inc(3, labels={"model": "qwen", "status": "success", "tenant": "t1"})
    reg.active_sessions.set(5)
    reg.ttft_seconds.observe(0.08, labels={"model": "qwen"})

    # Dynamic collector hook
    called = []
    def sample_collector():
        called.append(True)
        reg.loaded_models_count.set(2)

    reg.register_collector(sample_collector)

    output = reg.export()
    assert len(called) == 1
    assert "llmhub_requests_total" in output
    assert "llmhub_active_sessions" in output
    assert "llmhub_loaded_models_count 2.0" in output
    assert output.endswith("\n")


# ============================================================================
# 2. Structured JSON Logging & Request Tracing Tests (US-046)
# ============================================================================

def test_json_formatter_and_trace_id():
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="llmhub.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Inference completed for model %s",
        args=("qwen",),
        exc_info=None,
    )
    record.trace_id = "req-abc-123"

    formatted = fmt.format(record)
    data = json.loads(formatted)
    assert data["level"] == "INFO"
    assert data["logger"] == "llmhub.test"
    assert data["message"] == "Inference completed for model qwen"
    assert data["trace_id"] == "req-abc-123"
    assert "timestamp" in data


def test_request_context_filter():
    ctx_filter = RequestContextFilter()
    record = logging.LogRecord(
        name="llmhub.test", level=logging.INFO, pathname="t.py", lineno=1, msg="hi", args=(), exc_info=None
    )
    
    # When contextvar is empty
    ctx_filter.filter(record)
    assert record.trace_id == "-"

    # When contextvar is set
    token = current_request_id.set("req-trace-xyz")
    try:
        ctx_filter.filter(record)
        assert record.trace_id == "req-trace-xyz"
    finally:
        current_request_id.reset(token)


def test_setup_logging_file_creation(tmp_path):
    logs_dir = tmp_path / "logs"
    setup_logging(log_level="DEBUG", logs_dir=logs_dir, json_format=True)
    log_file = logs_dir / "llmhub.log"
    assert log_file.exists()

    logger = get_logger("unit_test")
    logger.info("Test log record line")

    # Flush handlers
    for h in logging.getLogger("llmhub").handlers:
        h.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "Test log record line" in content


# ============================================================================
# 3. Crash Recovery, PID Watchdog, & Run-Dir Cleanup Tests (US-047)
# ============================================================================

def test_pid_manager_lifecycle(tmp_path):
    run_dir = tmp_path / "run"
    mgr = PIDManager(run_dir)

    # 1. Acquire
    mgr.acquire(host="127.0.0.1", port=11434, version="0.9.0")
    assert mgr.pid_file.exists()
    info = mgr.read_pid_info()
    assert info is not None
    assert info["pid"] == os.getpid()
    assert info["port"] == 11434

    # 2. Re-acquire with current process PID should succeed
    mgr.acquire(host="127.0.0.1", port=11434, version="0.9.0")

    # 3. Release
    mgr.release()
    assert not mgr.pid_file.exists()


def test_pid_manager_detects_stale_lock(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_file = run_dir / "llmhub.pid"

    # Write a PID that does not exist
    stale_pid = 999999
    while is_process_alive(stale_pid):
        stale_pid += 1

    with open(pid_file, "w", encoding="utf-8") as f:
        json.dump({
            "pid": stale_pid,
            "host": "127.0.0.1",
            "port": 11434,
            "version": "0.8.0",
            "started_at": "2026-10-14T00:00:00Z"
        }, f)

    mgr = PIDManager(run_dir)
    # Must detect stale PID, remove it, and cleanly acquire for current process
    mgr.acquire(host="127.0.0.1", port=11434, version="0.9.0")
    new_info = mgr.read_pid_info()
    assert new_info["pid"] == os.getpid()
    mgr.release()


def test_pid_manager_prevents_duplicate_server(tmp_path):
    run_dir = tmp_path / "run"
    mgr = PIDManager(run_dir)

    # Spawn a temporary foreign python process
    import subprocess
    import sys
    dummy_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        pid_file = run_dir / "llmhub.pid"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(pid_file, "w", encoding="utf-8") as f:
            json.dump({"pid": dummy_proc.pid, "host": "127.0.0.1", "port": 11434, "version": "0.9.0"}, f)

        with pytest.raises(ServerAlreadyRunningError) as exc_info:
            mgr.acquire(host="127.0.0.1", port=11434, version="0.9.0")
        assert f"PID {dummy_proc.pid}" in str(exc_info.value)
    finally:
        dummy_proc.terminate()
        dummy_proc.wait()


def test_pid_manager_cleanup_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    mgr = PIDManager(run_dir)

    (run_dir / "stale.lock").write_text("lock", encoding="utf-8")
    (run_dir / "orphan-123.sock").write_text("sock", encoding="utf-8")
    (run_dir / "temp.tmp").write_text("tmp", encoding="utf-8")
    (run_dir / "keep_me.txt").write_text("keep", encoding="utf-8")

    cleaned = mgr.cleanup_run_dir()
    assert "stale.lock" in cleaned
    assert "orphan-123.sock" in cleaned
    assert "temp.tmp" in cleaned
    assert (run_dir / "keep_me.txt").exists()


# ============================================================================
# 4. Graceful Model Unloading & Request Draining Tests (US-048)
# ============================================================================

class MockDummyBackend(Backend):
    def __init__(self):
        super().__init__()
        self._loaded = True

    def load(self, model_path, n_ctx=2048, n_threads=4, use_mmap=True, use_mlock=False):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def new_session(self, session_id, prompt, max_tokens=512, temperature=0.7, top_p=0.9, stop=None):
        st = SessionState(session_id=session_id, prompt=prompt, max_tokens=max_tokens)
        self.sessions[session_id] = st
        return st

    def decode_token(self, session_id):
        return DecodeResult(token="tok", is_eos=True)

    def close_session(self, session_id):
        self.sessions.pop(session_id, None)

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def tokenize(self, text):
        return [1, 2, 3]

    def detokenize(self, tokens):
        return "text"

    def is_loaded(self) -> bool:
        return self._loaded

    def prefill_chunk(self, session_id, max_tokens_chunk=256):
        return 3, True

    def decode_step(self, session_id, n_tokens=1):
        return DecodeResult(session_id, [10], ["token"], True, "stop")

    def get_info(self):
        return {"status": "loaded" if self._loaded else "unloaded", "active_sessions": len(self.sessions)}

    def get_memory_info(self):
        return {"status": "loaded" if self._loaded else "unloaded", "active_sessions": len(self.sessions)}


def test_loaded_model_draining_blocks_new_requests(tmp_path):
    async def _run():
        config = LLMHubConfig(base_dir=tmp_path)
        mm = ModelManager(config=config, backend_factory=MockDummyBackend)

        dummy_model = LoadedModel(
            tag="test/model:q4_k_m",
            file_path=tmp_path / "model.gguf",
            backend=MockDummyBackend(),
            metadata=None,
            template_engine=None,
            n_ctx=2048,
            n_threads=4,
            weights_bytes=1000,
            kv_cache_bytes=100,
            overhead_bytes=10,
            total_memory_bytes=1110,
            active_sessions=1,
        )
        mm.loaded_models["test/model:q4_k_m"] = dummy_model

        # Mark draining
        dummy_model.is_draining = True

        # Attempt to retrieve model for new request must fail with ModelDrainingError
        with pytest.raises(ModelDrainingError):
            await mm.get_or_load_model("test/model:q4_k_m")

    asyncio.run(_run())


def test_unload_model_graceful_drain_success(tmp_path):
    async def _run():
        config = LLMHubConfig(base_dir=tmp_path)
        mm = ModelManager(config=config, backend_factory=MockDummyBackend)

        backend = MockDummyBackend()
        dummy_model = LoadedModel(
            tag="test/drain:q4_k_m",
            file_path=tmp_path / "model.gguf",
            backend=backend,
            metadata=None,
            template_engine=None,
            n_ctx=2048,
            n_threads=4,
            weights_bytes=1000,
            kv_cache_bytes=100,
            overhead_bytes=10,
            total_memory_bytes=1110,
            active_sessions=1,
        )
        mm.loaded_models["test/drain:q4_k_m"] = dummy_model

        # Asynchronously finish the active session after 0.05s
        async def finish_session_after_delay():
            await asyncio.sleep(0.05)
            dummy_model.dec_active_sessions()

        asyncio.create_task(finish_session_after_delay())

        # unload with drain=True
        unloaded = await mm.unload_model("test/drain:q4_k_m", drain=True, timeout=2.0)
        assert unloaded is True
        assert "test/drain:q4_k_m" not in mm.loaded_models
        assert backend._loaded is False

    asyncio.run(_run())


def test_unload_model_drain_timeout_raises(tmp_path):
    async def _run():
        config = LLMHubConfig(base_dir=tmp_path)
        mm = ModelManager(config=config, backend_factory=MockDummyBackend)

        dummy_model = LoadedModel(
            tag="test/stuck:q4_k_m",
            file_path=tmp_path / "model.gguf",
            backend=MockDummyBackend(),
            metadata=None,
            template_engine=None,
            n_ctx=2048,
            n_threads=4,
            weights_bytes=1000,
            kv_cache_bytes=100,
            overhead_bytes=10,
            total_memory_bytes=1110,
            active_sessions=2,
        )
        mm.loaded_models["test/stuck:q4_k_m"] = dummy_model

        # Does not finish sessions; timeout=0.1s must raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await mm.unload_model("test/stuck:q4_k_m", drain=True, timeout=0.1, force=False)
        assert "Timed out draining" in str(exc_info.value)
        # Model remains resident
        assert "test/stuck:q4_k_m" in mm.loaded_models

    asyncio.run(_run())


# ============================================================================
# 5. API Key Auth & TokenBucket Rate Limiting Tests (US-049)
# ============================================================================

def test_token_bucket_consume_and_refill():
    # 2 tokens capacity, refills 10 tokens/sec
    bucket = TokenBucket(capacity=2.0, refill_rate_per_sec=10.0)

    # 1. First consume succeeds
    ok, retry, rem, lim = bucket.consume(1.0)
    assert ok is True
    assert rem == 1
    assert lim == 2

    # 2. Second consume succeeds (bucket empty now)
    ok, retry, rem, lim = bucket.consume(1.0)
    assert ok is True
    assert rem == 0

    # 3. Third consume immediately fails with retry_after > 0
    ok, retry, rem, lim = bucket.consume(1.0)
    assert ok is False
    assert retry > 0.0
    assert rem == 0

    # 4. Wait 0.15s (1.5 tokens refilled) -> consume succeeds
    time.sleep(0.15)
    ok, retry, rem, lim = bucket.consume(1.0)
    assert ok is True


def test_rate_limiter_rpm_and_tpm():
    rl = RateLimiter(default_rpm=3, default_tpm=1000)

    # 3 requests allowed under RPM
    assert rl.check_request("client_1", est_tokens=10)[0] is True
    assert rl.check_request("client_1", est_tokens=10)[0] is True
    assert rl.check_request("client_1", est_tokens=10)[0] is True

    # 4th request rejected
    ok, retry_after, remaining, limit = rl.check_request("client_1", est_tokens=10)
    assert ok is False
    assert retry_after > 0.0

    # Independent client bucket is unaffected
    assert rl.check_request("client_2", est_tokens=10)[0] is True


def test_api_key_authenticator():
    auth = APIKeyAuthenticator(
        enabled=True,
        api_keys=["valid-key-123", "secret-token-abc"],
        exempt_localhost=True,
        exempt_paths={"/healthz", "/metrics"},
    )

    # 1. Exempt path
    ok, cid, err = auth.authenticate_header(auth_header=None, client_host="192.168.1.5", path="/healthz")
    assert ok is True

    # 2. Localhost exemption
    ok, cid, err = auth.authenticate_header(auth_header=None, client_host="127.0.0.1", path="/v1/chat/completions")
    assert ok is True

    # 3. Non-localhost missing auth
    ok, cid, err = auth.authenticate_header(auth_header=None, client_host="192.168.1.5", path="/v1/chat/completions")
    assert ok is False
    assert "Missing Authorization" in (err or "")

    # 4. Non-localhost invalid token
    ok, cid, err = auth.authenticate_header(auth_header="Bearer bad-token", client_host="192.168.1.5", path="/v1/chat/completions")
    assert ok is False
    assert "Invalid API key" in (err or "")

    # 5. Non-localhost valid token
    ok, cid, err = auth.authenticate_header(auth_header="Bearer valid-key-123", client_host="192.168.1.5", path="/v1/chat/completions")
    assert ok is True
    assert cid == "valid-key-123"


# ============================================================================
# 6. Request Validation, Size Limits, & Input Sanitization Tests (US-050)
# ============================================================================

def test_input_sanitizer_null_bytes_and_control_chars():
    dirty = "Hello\x00 World!\x08 This is \x1fvalid \nwith newlines\rand\ttabs."
    clean = InputSanitizer.sanitize_text(dirty)
    assert "\x00" not in clean
    assert "\x08" not in clean
    assert "\x1f" not in clean
    assert "\n" in clean
    assert "\r" in clean
    assert "\t" in clean
    assert "Hello World! This is valid \nwith newlines\rand\ttabs." == clean


def test_validate_request_size():
    # Pass
    InputSanitizer.validate_request_size(1024, max_bytes=2048)

    # Exceed
    with pytest.raises(PayloadTooLargeError):
        InputSanitizer.validate_request_size(4096, max_bytes=2048)


def test_validate_chat_messages():
    # Valid messages
    valid = [
        {"role": "system", "content": "You are assistant."},
        {"role": "user", "content": "Hello\x00 world!"},
    ]
    res = InputSanitizer.validate_chat_messages(valid)
    assert len(res) == 2
    assert res[1]["content"] == "Hello world!"

    # Empty list
    with pytest.raises(ValidationError):
        InputSanitizer.validate_chat_messages([])

    # Invalid role
    with pytest.raises(ValidationError):
        InputSanitizer.validate_chat_messages([{"role": "attacker", "content": "malicious"}])


def test_validate_prompt_bounds():
    prompt = "A" * 100
    InputSanitizer.validate_prompt_bounds(prompt, max_ctx=2048)

    huge_prompt = "A" * 50000
    with pytest.raises(ValidationError):
        InputSanitizer.validate_prompt_bounds(huge_prompt, max_ctx=512)


# ============================================================================
# 7. Strict Configuration Schema Validation Tests (US-051)
# ============================================================================

def test_config_schema_validation_valid(tmp_path):
    config = LLMHubConfig(base_dir=tmp_path)
    errors = validate_config_schema(config.settings)
    assert errors == []


def test_config_schema_validation_invalid_ranges():
    bad_settings = {
        "server": {"port": 999999},
        "runtime": {"memory_safety_margin": 1.8, "default_threads": 0},
        "scheduler": {"threading_mode": "quantum-leap"},
        "security": {"default_rpm": -5},
        "termux": {"thermal_warning_temp_c": 80.0, "thermal_critical_temp_c": 70.0},
        "unknown_section": {},
    }
    errors = validate_config_schema(bad_settings)
    assert any("server.port" in e for e in errors)
    assert any("runtime.memory_safety_margin" in e for e in errors)
    assert any("runtime.default_threads" in e for e in errors)
    assert any("scheduler.threading_mode" in e for e in errors)
    assert any("security.default_rpm" in e for e in errors)
    assert any("thermal_critical_temp_c" in e for e in errors)
    assert any("Unknown configuration section" in e for e in errors)


def test_config_strict_validation_raises(tmp_path):
    config = LLMHubConfig(base_dir=tmp_path)
    config.settings["server"]["port"] = -1
    with pytest.raises(ConfigValidationError):
        config.validate(strict=True)


# ============================================================================
# 8. FastAPI Integration & Middleware Tests (US-052)
# ============================================================================

@pytest.fixture
def test_client():
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_get_metrics_endpoint(test_client):
    resp = test_client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    text = resp.text
    assert "llmhub_requests_total" in text
    assert "llmhub_active_sessions" in text
    assert "llmhub_loaded_models_count" in text


def test_request_tracing_header(test_client):
    # Custom X-Request-ID
    resp = test_client.get("/healthz", headers={"X-Request-ID": "test-trace-999"})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == "test-trace-999"

    # Auto-generated X-Request-ID
    resp2 = test_client.get("/healthz")
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Request-ID", "").startswith("req-")


def test_payload_too_large_413(test_client):
    # Set max_request_body_bytes to 500 bytes for testing
    old_max = state.config.settings["security"]["max_request_body_bytes"]
    state.config.settings["security"]["max_request_body_bytes"] = 500
    try:
        huge_body = "x" * 1000
        resp = test_client.post(
            "/v1/chat/completions",
            content=huge_body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(huge_body))}
        )
        assert resp.status_code == 413
        assert "Payload too large" in resp.json()["error"]["message"]
    finally:
        state.config.settings["security"]["max_request_body_bytes"] = old_max


def test_auth_middleware_401_and_success(test_client):
    # Enable auth with key "secret-phase-9"
    sec_cfg = state.config.settings["security"]
    old_auth = sec_cfg.get("auth_enabled", False)
    old_keys = sec_cfg.get("api_keys", [])
    old_localhost = sec_cfg.get("exempt_localhost", True)

    sec_cfg["auth_enabled"] = True
    sec_cfg["api_keys"] = ["secret-phase-9"]
    sec_cfg["exempt_localhost"] = False  # Disable localhost bypass for test
    state.authenticator = APIKeyAuthenticator(
        enabled=True,
        api_keys=["secret-phase-9"],
        exempt_localhost=False,
    )

    try:
        # 1. Exempt /healthz should work without token
        resp = test_client.get("/healthz")
        assert resp.status_code == 200

        # 2. Non-exempt endpoint /api/ps without token -> 401
        resp = test_client.get("/api/ps")
        assert resp.status_code == 401

        # 3. Invalid token -> 401
        resp = test_client.get("/api/ps", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

        # 4. Valid token -> 200
        resp = test_client.get("/api/ps", headers={"Authorization": "Bearer secret-phase-9"})
        assert resp.status_code == 200
    finally:
        sec_cfg["auth_enabled"] = old_auth
        sec_cfg["api_keys"] = old_keys
        sec_cfg["exempt_localhost"] = old_localhost
        state.authenticator = APIKeyAuthenticator(
            enabled=old_auth,
            api_keys=old_keys,
            exempt_localhost=old_localhost,
        )


def test_rate_limiting_middleware_429(test_client):
    old_auth = state.authenticator
    old_rl = state.rate_limiter
    try:
        # Configure tiny RPM limit of 2 requests
        state.authenticator = APIKeyAuthenticator(enabled=False, exempt_localhost=False)
        state.rate_limiter = RateLimiter(default_rpm=2, default_tpm=1000)

        # 1st and 2nd requests pass
        r1 = test_client.get("/api/ps")
        assert r1.status_code == 200
        r2 = test_client.get("/api/ps")
        assert r2.status_code == 200

        # 3rd request rejected with 429
        r3 = test_client.get("/api/ps")
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers
        assert r3.headers.get("X-RateLimit-Limit") == "2"
        assert r3.headers.get("X-RateLimit-Remaining") == "0"
    finally:
        state.authenticator = old_auth
        state.rate_limiter = old_rl


def test_x_request_id_present_on_error_responses(test_client):
    """Verify that tracing middleware is outermost and adds X-Request-ID even on 401, 413, 429."""
    # 1. Test 401
    old_auth = state.authenticator
    try:
        state.authenticator = APIKeyAuthenticator(enabled=True, api_keys=["key1"], exempt_localhost=False)
        r401 = test_client.get("/api/ps")
        assert r401.status_code == 401
        assert "X-Request-ID" in r401.headers
        assert r401.headers["X-Request-ID"].startswith("req-")
    finally:
        state.authenticator = old_auth

    # 2. Test 413
    old_max = state.config.settings["security"]["max_request_body_bytes"]
    try:
        state.config.settings["security"]["max_request_body_bytes"] = 10
        r413 = test_client.post(
            "/v1/chat/completions",
            content="excessively large payload for test",
            headers={"Content-Length": "34"}
        )
        assert r413.status_code == 413
        assert "X-Request-ID" in r413.headers
    finally:
        state.config.settings["security"]["max_request_body_bytes"] = old_max

    # 3. Test 429
    old_rl = state.rate_limiter
    try:
        state.rate_limiter = RateLimiter(default_rpm=1, default_tpm=100)
        test_client.get("/api/ps")
        r429 = test_client.get("/api/ps")
        assert r429.status_code == 429
        assert "X-Request-ID" in r429.headers
    finally:
        state.rate_limiter = old_rl


def test_chunked_transfer_exceeds_max_bytes_returns_413(test_client):
    """Verify streaming/chunked requests without Content-Length are stopped if they exceed max_bytes."""
    old_max = state.config.settings["security"]["max_request_body_bytes"]
    try:
        state.config.settings["security"]["max_request_body_bytes"] = 50

        def chunk_gen():
            yield b'{"model": "test", "messages": [{"role": "user", '
            yield b'"content": "this is a very long chunked streaming payload that clearly exceeds 50 bytes"}]}'

        resp = test_client.post(
            "/v1/chat/completions",
            content=chunk_gen(),
            headers={"Transfer-Encoding": "chunked", "Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        assert "Payload too large" in resp.json()["error"]["message"]
        assert "X-Request-ID" in resp.headers
    finally:
        state.config.settings["security"]["max_request_body_bytes"] = old_max


def test_forwarded_header_prevents_localhost_auth_bypass(test_client):
    """Verify that X-Forwarded-For or Forwarded header prevents localhost auth exemption."""
    old_auth = state.authenticator
    try:
        state.authenticator = APIKeyAuthenticator(
            enabled=True,
            api_keys=["valid-proxy-key"],
            exempt_localhost=True,
        )
        # 1. Localhost direct without forward header -> allowed by localhost exemption
        r_direct = test_client.get("/api/ps")
        assert r_direct.status_code == 200

        # 2. Localhost with X-Forwarded-For header -> NOT exempt, returns 401
        r_fwd = test_client.get("/api/ps", headers={"X-Forwarded-For": "198.51.100.1"})
        assert r_fwd.status_code == 401

        # 3. Localhost with Forwarded header and valid Bearer token -> 200
        r_fwd_auth = test_client.get(
            "/api/ps",
            headers={
                "Forwarded": "for=198.51.100.1;proto=https",
                "Authorization": "Bearer valid-proxy-key",
            },
        )
        assert r_fwd_auth.status_code == 200
    finally:
        state.authenticator = old_auth

