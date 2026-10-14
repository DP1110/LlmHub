"""Tests for LLMHub FastAPI HTTP server endpoints and LLMHubClient."""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from llmhub.server.app import app, state
from llmhub.server.backend import DecodeResult, SessionState
from llmhub.server.client import LLMHubClient, ServerUnavailableError

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

@patch.object(state, "registry")
def test_chat_completions_json(mock_registry, client):
    mock_record = MagicMock()
    mock_record.tag = "mock-model"
    mock_record.file_path = "mock.gguf"
    mock_registry.find_model.return_value = mock_record

    orig_backend = state.backend
    state.backend = MockBackend()
    state.current_model_tag = "mock-model"

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
        state.backend = orig_backend

@patch.object(state, "registry")
def test_chat_completions_streaming_sse(mock_registry, client):
    mock_record = MagicMock()
    mock_record.tag = "mock-model"
    mock_record.file_path = "mock.gguf"
    mock_registry.find_model.return_value = mock_record

    orig_backend = state.backend
    state.backend = MockBackend()
    state.current_model_tag = "mock-model"

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
        state.backend = orig_backend

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

    orig_backend = state.backend
    state.backend = MockBackend()
    state.current_model_tag = "mock-model"

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
        state.backend = orig_backend

def test_llmhub_client_not_running():
    cli = LLMHubClient(base_url="http://127.0.0.1:59999")
    assert not cli.is_server_running()
    with pytest.raises(ServerUnavailableError):
        cli.check_connection()
