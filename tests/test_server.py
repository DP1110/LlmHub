"""Tests for LLMHub FastAPI HTTP server endpoints and LLMHubClient."""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from pathlib import Path
from llmhub.core.template import ChatTemplateEngine
from llmhub.server.app import app, state
from llmhub.server.backend import DecodeResult, SessionState
from llmhub.server.client import LLMHubClient, ServerUnavailableError
from llmhub.server.model_manager import LoadedModel

@pytest.fixture
def client():
    return TestClient(app)

def test_healthz_endpoint(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "is_model_loaded" in data

def test_v1_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)

def test_api_models_endpoint(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert isinstance(data["models"], list)

def test_api_ps_endpoint(client):
    resp = client.get("/api/ps")
    assert resp.status_code == 200
    data = resp.json()
    assert "backend_status" in data
    assert "active_sessions" in data
    assert "system_ram_free_mb" in data

class MockBackend:
    def __init__(self):
        self._loaded = True
        self.metadata = MagicMock()
        self.metadata.chat_template = None
        self.metadata.architecture = "llama"
        self.sessions = {}

    def is_loaded(self):
        return self._loaded

    def load(self, *args, **kwargs):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def new_session(self, session_id, prompt, max_tokens=512, **kwargs):
        sess = SessionState(
            session_id=session_id,
            model_tag="mock-model",
            prompt=prompt,
            prompt_tokens=[1, 2, 3],
            max_tokens=max_tokens
        )
        self.sessions[session_id] = sess
        return sess

    def prefill_chunk(self, session_id, max_tokens_chunk=256):
        return 3, True

    def decode_step(self, session_id, n_tokens=1):
        sess = self.sessions.get(session_id)
        if not sess:
            return DecodeResult(session_id, [], [], True)
        
        step_idx = len(sess.generated_tokens)
        if step_idx == 0:
            sess.generated_tokens.append(100)
            return DecodeResult(
                session_id=session_id,
                tokens=[100],
                text_deltas=["Hello"],
                done=False,
                ttft_ms=10.0,
                decode_tok_s=25.0,
                total_tokens_generated=1
            )
        elif step_idx == 1:
            sess.generated_tokens.append(101)
            sess.done = True
            return DecodeResult(
                session_id=session_id,
                tokens=[101],
                text_deltas=[" world"],
                done=True,
                stop_reason="stop",
                ttft_ms=10.0,
                decode_tok_s=25.0,
                total_tokens_generated=2
            )
        else:
            return DecodeResult(session_id, [], [], True, "stop")

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def close_session(self, session_id):
        self.sessions.pop(session_id, None)

    def get_info(self):
        return {
            "status": "loaded",
            "active_sessions": len(self.sessions),
            "memory": {"total_mb": 100.0, "weights_mb": 80.0, "kv_cache_mb": 20.0},
            "bandwidth": {"theoretical_max_tok_s": 50.0, "bottleneck": "Memory-Bandwidth Bound"}
        }

def setup_mock_model(tag: str = "mock-model") -> LoadedModel:
    b = MockBackend()
    lm = LoadedModel(
        tag=tag,
        file_path=Path(f"{tag}.gguf"),
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
    state.model_manager.loaded_models[tag] = lm
    return lm

@patch.object(state, "registry")
def test_chat_completions_json(mock_registry, client):
    mock_record = MagicMock()
    mock_record.tag = "mock-model"
    mock_record.file_path = "mock.gguf"
    mock_registry.find_model.return_value = mock_record

    lm = setup_mock_model("mock-model")
    try:
        payload = {
            "model": "mock-model",
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
            "stream": False
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello world"
        assert data["usage"]["total_tokens"] > 0
    finally:
        state.model_manager.loaded_models.pop("mock-model", None)

@patch.object(state, "registry")
def test_chat_completions_streaming_sse(mock_registry, client):
    mock_record = MagicMock()
    mock_record.tag = "mock-model"
    mock_record.file_path = "mock.gguf"
    mock_registry.find_model.return_value = mock_record

    lm = setup_mock_model("mock-model")
    try:
        payload = {
            "model": "mock-model",
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
            "stream": True
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = resp.text.strip().split("\n\n")
        chunks = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("data: "):
                raw = line_str[6:].strip()
                if raw == "[DONE]":
                    chunks.append("[DONE]")
                else:
                    chunks.append(json.loads(raw))

        assert len(chunks) >= 5
        # Chunk 0: initial role delta
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        # Chunk 1: first token
        assert chunks[1]["choices"][0]["delta"]["content"] == "Hello"
        # Chunk 2: second token
        assert chunks[2]["choices"][0]["delta"]["content"] == " world"
        # Chunk 3: stop chunk with usage
        assert chunks[3]["choices"][0]["finish_reason"] == "stop"
        assert "usage" in chunks[3]
        # Chunk 4: stream done marker
        assert chunks[-1] == "[DONE]"
    finally:
        state.model_manager.loaded_models.pop("mock-model", None)

@patch.object(state, "registry")
def test_v1_get_model_by_id(mock_registry, client):
    mock_record = MagicMock()
    mock_record.tag = "qwen2.5:q4_k_m"
    mock_record.org = "Qwen"
    mock_record.name = "Qwen2.5"
    mock_registry.find_model.return_value = mock_record

    resp = client.get("/v1/models/qwen2.5:q4_k_m")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "qwen2.5:q4_k_m"
    assert data["object"] == "model"

    mock_registry.find_model.return_value = None
    resp_404 = client.get("/v1/models/nonexistent")
    assert resp_404.status_code == 404

@patch.object(state, "registry")
def test_chat_message_with_null_content(mock_registry, client):
    mock_record = MagicMock()
    mock_record.tag = "mock-model"
    mock_record.file_path = "mock.gguf"
    mock_registry.find_model.return_value = mock_record

    lm = setup_mock_model("mock-model")
    try:
        payload = {
            "model": "mock-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": None},
                {"role": "user", "content": "How are you?"}
            ],
            "stream": False
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
    finally:
        state.model_manager.loaded_models.pop("mock-model", None)

def test_api_pin_and_unpin_endpoints(client):
    lm = setup_mock_model("model-to-pin")
    try:
        assert lm.pinned is False
        # Pin
        resp = client.post("/api/models/model-to-pin/pin")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pinned"
        assert lm.pinned is True

        # Unpin
        resp_unpin = client.post("/api/models/model-to-pin/unpin")
        assert resp_unpin.status_code == 200
        assert resp_unpin.json()["status"] == "unpinned"
        assert lm.pinned is False

        # Unpin non-resident
        resp_404 = client.post("/api/models/non-existent/unpin")
        assert resp_404.status_code == 404
    finally:
        state.model_manager.loaded_models.pop("model-to-pin", None)

def test_api_unload_endpoint(client):
    lm = setup_mock_model("model-to-unload")
    try:
        assert "model-to-unload" in state.model_manager.loaded_models
        resp = client.post("/api/models/model-to-unload/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"
        assert "model-to-unload" not in state.model_manager.loaded_models
    finally:
        state.model_manager.loaded_models.pop("model-to-unload", None)

def test_llmhub_client_not_running():
    cli = LLMHubClient(base_url="http://127.0.0.1:59999")
    assert not cli.is_server_running()
    with pytest.raises(ServerUnavailableError):
        cli.check_connection()


def test_web_gui_and_logo_endpoints(client):
    # 1. Test GET / serves the Web GUI
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert "LLMHub" in resp_root.text
    assert "<!DOCTYPE html>" in resp_root.text

    # 2. Test GET /dashboard serves the same Web GUI
    resp_dash = client.get("/dashboard")
    assert resp_dash.status_code == 200
    assert "LLMHub" in resp_dash.text

    # 3. Test GET /static/logo.svg
    resp_svg = client.get("/static/logo.svg")
    assert resp_svg.status_code == 200
    assert "svg" in resp_svg.text

    # 4. Test GET /static/logo.jpg
    resp_jpg = client.get("/static/logo.jpg")
    assert resp_jpg.status_code == 200
    assert len(resp_jpg.content) > 1000

    # 5. Test root /logo.svg and /favicon.ico
    assert client.get("/logo.svg").status_code == 200
    assert client.get("/favicon.ico").status_code == 200


@patch.object(state, "registry")
def test_chat_completions_auto_loads_model_on_termux(mock_registry, client, tmp_path):
    """Test that POST /v1/chat/completions auto-loads model in Termux unprivileged environment without 500 error."""
    model_file = tmp_path / "termux-auto.gguf"
    model_file.write_bytes(b"\x00" * 1024)

    mock_record = MagicMock()
    mock_record.tag = "termux-auto"
    mock_record.file_path = str(model_file)
    mock_registry.find_model.return_value = mock_record

    # Ensure model is not currently resident
    state.model_manager.loaded_models.pop("termux-auto", None)
    old_factory = state.model_manager.backend_factory
    state.model_manager.backend_factory = lambda: MockBackend()

    mock_meta = MagicMock()
    mock_meta.block_count = 16
    mock_meta.head_count_kv = 4
    mock_meta.head_dim = 64
    mock_meta.chat_template = None
    mock_meta.architecture = "llama"
    mock_meta.model_name = "termux-auto"

    try:
        with patch.object(state.model_manager.registry, "find_model", return_value=mock_record):
            with patch("llmhub.core.gguf_parser.GGUFParser.parse_file", return_value=mock_meta):
                with patch("llmhub.hardware.detector.HardwareDetector.is_termux", return_value=True):
                    with patch("llmhub.hardware.detector.HardwareDetector.is_root", return_value=False):
                        payload = {
                            "model": "termux-auto",
                            "messages": [{"role": "user", "content": "Hello from Termux"}],
                            "stream": False,
                        }
                        resp = client.post("/v1/chat/completions", json=payload)
                        assert resp.status_code == 200
                        data = resp.json()
                        assert data["object"] == "chat.completion"
                        assert data["choices"][0]["message"]["content"] == "Hello world"
                        assert "termux-auto" in state.model_manager.loaded_models
    finally:
        state.model_manager.loaded_models.pop("termux-auto", None)
        state.model_manager.backend_factory = old_factory


def test_chat_completions_sse_streaming_headers(client):
    lm = setup_mock_model("mock-streaming")
    try:
        payload = {
            "model": "mock-streaming",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert resp.headers.get("cache-control") == "no-cache, no-transform"
        assert resp.headers.get("x-accel-buffering") == "no"
    finally:
        state.model_manager.loaded_models.pop("mock-streaming", None)


def test_api_models_includes_resident_and_scanned(client):
    lm = setup_mock_model("mock-resident-only")
    try:
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        tags = [m["tag"] for m in data.get("models", [])]
        assert "mock-resident-only" in tags
        resident_flags = {m["tag"]: m.get("is_resident") for m in data.get("models", [])}
        assert resident_flags["mock-resident-only"] is True

        # Also verify in /v1/models
        v1_resp = client.get("/v1/models")
        assert v1_resp.status_code == 200
        v1_ids = [m["id"] for m in v1_resp.json().get("data", [])]
        assert "mock-resident-only" in v1_ids

        # Also verify in /api/ps
        ps_resp = client.get("/api/ps")
        assert ps_resp.status_code == 200
        ps_data = ps_resp.json()
        assert "resident_models" in ps_data
        assert "mock-resident-only" in ps_data["resident_models"]
    finally:
        state.model_manager.loaded_models.pop("mock-resident-only", None)



