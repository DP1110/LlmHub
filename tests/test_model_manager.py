"""Tests for Multi-Model Manager: admission control, LRU eviction, pinning, and keep-alive."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from llmhub.server.backend import Backend, DecodeResult, SessionState
from llmhub.server.model_manager import LoadedModel, MemoryAdmissionError, ModelManager

class MockBackend(Backend):
    def __init__(self):
        self._loaded = False
        self.sessions = {}

    def is_loaded(self):
        return self._loaded

    def load(self, model_path, n_ctx=2048, **kwargs):
        self._loaded = True
        self.model_path = model_path
        self.n_ctx = n_ctx

    def unload(self):
        self._loaded = False
        self.sessions.clear()

    def new_session(self, session_id, prompt, **kwargs):
        sess = SessionState(session_id, "mock", prompt, [1, 2, 3])
        self.sessions[session_id] = sess
        return sess

    def prefill_chunk(self, session_id, max_tokens_chunk=256):
        return 3, True

    def decode_step(self, session_id, n_tokens=1):
        return DecodeResult(session_id, [10], ["token"], True, "stop")

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def close_session(self, session_id):
        self.sessions.pop(session_id, None)

    def get_info(self):
        return {"status": "loaded" if self._loaded else "unloaded", "active_sessions": len(self.sessions)}

def create_mock_metadata(layers=16, kv_heads=4, head_dim=64):
    meta = MagicMock()
    meta.block_count = layers
    meta.head_count_kv = kv_heads
    meta.head_dim = head_dim
    meta.chat_template = None
    meta.architecture = "llama"
    meta.model_name = "test"
    return meta

@pytest.fixture
def mock_registry(tmp_path):
    reg = MagicMock()
    
    def fake_find(spec):
        rec = MagicMock()
        rec.tag = spec
        rec.org = "test-org"
        rec.name = spec
        # Create dummy file with controlled size
        p = tmp_path / f"{spec}.gguf"
        if not p.exists():
            p.write_bytes(b"\x00" * (100 * 1024 * 1024))  # 100 MB weights
        rec.file_path = str(p)
        return rec

    reg.find_model.side_effect = fake_find
    return reg

def test_multi_model_concurrent_residence(mock_registry, tmp_path):
    async def _run():
        with patch("llmhub.core.gguf_parser.GGUFParser.parse_file", return_value=create_mock_metadata()):
            with patch.object(ModelManager, "get_free_ram_bytes", return_value=16 * 1024 * 1024 * 1024):  # 16 GB free
                mgr = ModelManager(
                    registry=mock_registry,
                    backend_factory=lambda: MockBackend(),
                    safety_margin=0.85
                )

                m1 = await mgr.load_model("model-1", n_ctx=1024)
                m2 = await mgr.load_model("model-2", n_ctx=2048)
                m3 = await mgr.load_model("model-3", n_ctx=4096)

                assert len(mgr.loaded_models) == 3
                assert "model-1" in mgr.loaded_models
                assert "model-2" in mgr.loaded_models
                assert "model-3" in mgr.loaded_models

                # Verify isolated backends and contexts
                assert m1.backend != m2.backend
                assert m1.n_ctx == 1024
                assert m2.n_ctx == 2048
                assert m3.n_ctx == 4096
                assert m1.backend.is_loaded()
                assert m2.backend.is_loaded()
                assert m3.backend.is_loaded()
    asyncio.run(_run())

def test_memory_admission_rejection(mock_registry):
    async def _run():
        with patch("llmhub.core.gguf_parser.GGUFParser.parse_file", return_value=create_mock_metadata()):
            # Mock free RAM: only 50 MB available
            with patch.object(ModelManager, "get_free_ram_bytes", return_value=50 * 1024 * 1024):
                mgr = ModelManager(
                    registry=mock_registry,
                    backend_factory=lambda: MockBackend(),
                    safety_margin=0.85
                )

                # Model requires ~100MB weights + KV cache > 50MB * 0.85 (42.5 MB usable)
                with pytest.raises(MemoryAdmissionError) as exc_info:
                    await mgr.load_model("large-model", n_ctx=2048)

                assert "Insufficient memory" in str(exc_info.value)
                assert exc_info.value.required_bytes > exc_info.value.available_bytes
                assert len(mgr.loaded_models) == 0
    asyncio.run(_run())

def test_lru_eviction_under_memory_pressure(mock_registry):
    async def _run():
        with patch("llmhub.core.gguf_parser.GGUFParser.parse_file", return_value=create_mock_metadata()):
            # Set free RAM to 250 MB. Usable = 250 * 0.85 = 212.5 MB.
            # Each model takes ~110 MB total. So 1 model fits (110 <= 212.5), but 2 models (220 MB) require eviction!
            with patch.object(ModelManager, "get_free_ram_bytes", return_value=250 * 1024 * 1024):
                mgr = ModelManager(
                    registry=mock_registry,
                    backend_factory=lambda: MockBackend(),
                    safety_margin=0.85
                )

                # Load model-A first
                mA = await mgr.load_model("model-A")
                mA.last_active_time = time.monotonic() - 100.0  # Older usage
                assert "model-A" in mgr.loaded_models

                # Loading model-B requires evicting model-A because 110 + 110 > 212.5 MB usable!
                # After model-A is evicted, model-B fits.
                mB = await mgr.load_model("model-B")
                assert "model-B" in mgr.loaded_models
                assert "model-A" not in mgr.loaded_models
                assert not mA.backend.is_loaded()
    asyncio.run(_run())

def test_non_destructive_eviction_on_oversized_request(mock_registry, tmp_path):
    async def _run():
        with patch("llmhub.core.gguf_parser.GGUFParser.parse_file", return_value=create_mock_metadata()):
            with patch.object(ModelManager, "get_free_ram_bytes", return_value=250 * 1024 * 1024):
                mgr = ModelManager(
                    registry=mock_registry,
                    backend_factory=lambda: MockBackend(),
                    safety_margin=0.85
                )

                # Load healthy model-A
                mA = await mgr.load_model("model-A")
                assert "model-A" in mgr.loaded_models

                # Create an impossible 500 MB model that cannot fit even if model-A is evicted (usable = 212.5 MB)
                p_huge = tmp_path / "huge-model.gguf"
                p_huge.write_bytes(b"\x00" * (500 * 1024 * 1024))
                mock_rec = MagicMock()
                mock_rec.tag = "huge-model"
                mock_rec.file_path = str(p_huge)
                mock_registry.find_model.side_effect = lambda spec: mock_rec if spec == "huge-model" else mock_registry.find_model(spec)

                with pytest.raises(MemoryAdmissionError) as exc_info:
                    await mgr.load_model("huge-model")

                assert "No models were evicted" in str(exc_info.value)
                # Verify model-A was NOT destructively evicted!
                assert "model-A" in mgr.loaded_models
                assert mA.backend.is_loaded()
    asyncio.run(_run())

def test_pin_unloaded_model_no_deadlock(mock_registry):
    async def _run():
        with patch("llmhub.core.gguf_parser.GGUFParser.parse_file", return_value=create_mock_metadata()):
            with patch.object(ModelManager, "get_free_ram_bytes", return_value=16 * 1024 * 1024 * 1024):
                mgr = ModelManager(
                    registry=mock_registry,
                    backend_factory=lambda: MockBackend()
                )

                # Pin model that is NOT currently loaded - should load and pin without recursive lock deadlock!
                res = await mgr.pin_model("unloaded-model")
                assert res is True
                assert "unloaded-model" in mgr.loaded_models
                assert mgr.loaded_models["unloaded-model"].pinned is True
    asyncio.run(_run())

def test_model_pinning_prevents_lru_eviction(mock_registry):
    async def _run():
        with patch("llmhub.core.gguf_parser.GGUFParser.parse_file", return_value=create_mock_metadata()):
            with patch.object(ModelManager, "get_free_ram_bytes", return_value=250 * 1024 * 1024):
                mgr = ModelManager(
                    registry=mock_registry,
                    backend_factory=lambda: MockBackend(),
                    safety_margin=0.85
                )

                # Load model-pinned and pin it
                m_pin = await mgr.load_model("model-pinned", pinned=True)
                m_pin.last_active_time = time.monotonic() - 100.0  # Very old
                assert m_pin.pinned is True

                # Loading another model that requires memory cannot evict model-pinned, raising MemoryAdmissionError!
                with pytest.raises(MemoryAdmissionError):
                    await mgr.load_model("model-incoming")

                # Pinned model is still loaded and safe
                assert "model-pinned" in mgr.loaded_models
                assert m_pin.backend.is_loaded()
    asyncio.run(_run())

def test_active_sessions_prevent_eviction(mock_registry):
    async def _run():
        with patch("llmhub.core.gguf_parser.GGUFParser.parse_file", return_value=create_mock_metadata()):
            with patch.object(ModelManager, "get_free_ram_bytes", return_value=250 * 1024 * 1024):
                mgr = ModelManager(
                    registry=mock_registry,
                    backend_factory=lambda: MockBackend(),
                    safety_margin=0.85
                )

                m_busy = await mgr.load_model("model-busy")
                m_busy.active_sessions = 2  # Active user request in flight

                with pytest.raises(MemoryAdmissionError):
                    await mgr.load_model("model-incoming")

                assert "model-busy" in mgr.loaded_models
    asyncio.run(_run())

def test_keep_alive_reaper(mock_registry):
    async def _run():
        with patch("llmhub.core.gguf_parser.GGUFParser.parse_file", return_value=create_mock_metadata()):
            with patch.object(ModelManager, "get_free_ram_bytes", return_value=16 * 1024 * 1024 * 1024):
                mgr = ModelManager(
                    registry=mock_registry,
                    backend_factory=lambda: MockBackend(),
                    default_keep_alive=5.0  # 5 seconds
                )

                m1 = await mgr.load_model("m1", keep_alive=1.0)
                m2 = await mgr.load_model("m2", keep_alive=100.0)
                m3 = await mgr.load_model("m3", keep_alive=1.0, pinned=True)

                # Simulate passage of time
                m1.last_active_time = time.monotonic() - 2.0  # Expired
                m2.last_active_time = time.monotonic()        # Fresh
                m3.last_active_time = time.monotonic() - 2.0  # Expired time, but pinned!

                reaped = await mgr.reap_expired_models()
                assert "m1" in reaped
                assert "m2" not in reaped
                assert "m3" not in reaped

                assert "m1" not in mgr.loaded_models
                assert "m2" in mgr.loaded_models
                assert "m3" in mgr.loaded_models
    asyncio.run(_run())

def test_model_manager_get_ps_matrix():
    mgr = ModelManager()
    ps = mgr.get_ps_info()
    assert "resident_models_count" in ps
    assert "system_ram_free_mb" in ps
    assert "models" in ps
    assert isinstance(ps["models"], list)
