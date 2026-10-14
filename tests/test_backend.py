"""Tests for LLMHub Backend ABC, LlamaCppBackend, and RemoteBackend."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from llmhub.server.backend import Backend, DecodeResult, SessionState
from llmhub.server.llamacpp import LlamaCppBackend
from llmhub.server.remote import RemoteBackend

def test_session_state_creation():
    state = SessionState(
        session_id="test-sess-1",
        model_tag="test-model",
        prompt="Hello world",
        prompt_tokens=[1, 2, 3],
        max_tokens=128,
        temperature=0.8,
        top_p=0.95
    )
    assert state.session_id == "test-sess-1"
    assert state.model_tag == "test-model"
    assert len(state.prompt_tokens) == 3
    assert state.prefill_completed is False
    assert state.prefill_offset == 0
    assert state.done is False

def test_decode_result_metrics():
    res = DecodeResult(
        session_id="test-sess-1",
        tokens=[10, 11, 12, 13],
        text_deltas=[" The", " quick", " brown", " fox"],
        done=False,
        ttft_ms=15.4,
        decode_tok_s=42.5,
        total_tokens_generated=4
    )
    assert len(res.tokens) == 4
    assert len(res.text_deltas) == 4
    assert res.ttft_ms == 15.4
    assert res.decode_tok_s == 42.5
    assert not res.done

class DummyMockLlama:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def tokenize(self, text_bytes, add_bos=True):
        # 1 token per 4 bytes approximately
        return list(range(100, 100 + max(1, len(text_bytes) // 4)))

    def __call__(self, prompt, max_tokens=512, stream=True, **kwargs):
        chunks = [
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " from"}]},
            {"choices": [{"text": " backend"}]},
            {"choices": [{"text": "!"}]},
        ]
        for c in chunks:
            yield c

    def close(self):
        self.closed = True

@patch("llama_cpp.Llama", side_effect=DummyMockLlama)
@patch("llmhub.core.gguf_parser.GGUFParser.parse_file")
def test_llamacpp_backend_lifecycle(mock_parse, mock_llama, tmp_path):
    mock_meta = MagicMock()
    mock_meta.block_count = 24
    mock_meta.head_count_kv = 8
    mock_meta.head_dim = 64
    mock_meta.chat_template = None
    mock_meta.architecture = "qwen2"
    mock_meta.model_name = "test-model"
    mock_parse.return_value = mock_meta

    dummy_model_file = tmp_path / "model.gguf"
    dummy_model_file.write_bytes(b"\x00" * 1024 * 1024)  # 1MB dummy

    backend = LlamaCppBackend()
    assert not backend.is_loaded()

    # Load
    backend.load(dummy_model_file, n_ctx=1024, n_threads=4)
    assert backend.is_loaded()
    assert backend.n_ctx == 1024
    assert backend.n_threads == 4

    # New session
    sess = backend.new_session(
        session_id="s1",
        prompt="Tell me a joke.",
        max_tokens=10
    )
    assert sess.session_id == "s1"
    assert len(sess.prompt_tokens) > 0
    assert not sess.prefill_completed

    # Chunked prefill
    chunk_size, is_done = backend.prefill_chunk("s1", max_tokens_chunk=2)
    assert chunk_size <= 2

    # Finish prefill
    _, is_done = backend.prefill_chunk("s1", max_tokens_chunk=1000)
    assert is_done is True

    # Decode step 1 token
    res1 = backend.decode_step("s1", n_tokens=1)
    assert len(res1.text_deltas) == 1
    assert res1.text_deltas[0] == "Hello"
    assert not res1.done

    # Decode step 2 tokens
    res2 = backend.decode_step("s1", n_tokens=2)
    assert len(res2.text_deltas) == 2
    assert "".join(res2.text_deltas) == " from backend"

    # Decode until completion
    res3 = backend.decode_step("s1", n_tokens=5)
    assert res3.done is True
    assert backend.sessions["s1"].generated_text == "Hello from backend!"

    # Info
    info = backend.get_info()
    assert info["status"] == "loaded"
    assert info["architecture"] == "qwen2"
    assert info["active_sessions"] == 1
    assert "memory" in info
    assert "bandwidth" in info

    # Close session
    backend.close_session("s1")
    assert "s1" not in backend.sessions

    # Unload
    backend.unload()
    assert not backend.is_loaded()

@patch("httpx.Client")
def test_remote_backend_lifecycle(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.__enter__.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client.get.return_value = mock_resp

    backend = RemoteBackend(base_url="http://127.0.0.1:11434")
    assert not backend.is_loaded()

    backend.load(Path("qwen2.5-0.5b"))
    assert backend.is_loaded()
    assert backend.model_name == "qwen2.5-0.5b"

    # New session with mocked streaming response
    mock_stream_resp = MagicMock()
    mock_stream_resp.status_code = 200
    mock_stream_resp.iter_lines.return_value = [
        'data: {"choices": [{"delta": {"content": "Remote"}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {"content": " response"}, "finish_reason": "stop"}]}',
        'data: [DONE]'
    ]
    mock_client.send.return_value = mock_stream_resp

    sess = backend.new_session("rem-s1", prompt="Ping")
    assert sess.session_id == "rem-s1"

    # Prefill
    chunk, done = backend.prefill_chunk("rem-s1", max_tokens_chunk=100)
    assert done is True

    # Decode step
    res1 = backend.decode_step("rem-s1", n_tokens=1)
    assert res1.text_deltas == ["Remote"]
    assert not res1.done

    res2 = backend.decode_step("rem-s1", n_tokens=1)
    assert res2.text_deltas == [" response"]
    assert res2.done is True

    backend.close_session("rem-s1")
    assert "rem-s1" not in backend.sessions

    backend.unload()
    assert not backend.is_loaded()
