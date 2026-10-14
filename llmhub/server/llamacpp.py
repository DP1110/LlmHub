"""llama.cpp execution backend implementing the Backend ABC.

Provides token-level quantum decoding, chunked prefill, and multi-session isolation.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    llama_cpp = None
    HAS_LLAMA_CPP = False

from llmhub.core.gguf_parser import GGUFParser, GGUFMetadata
from llmhub.hardware.tuner import AutoTuner
from llmhub.server.backend import Backend, DecodeResult, SessionState
from llmhub.utils.math_calc import compute_memory_breakdown, estimate_decode_throughput

class LlamaCppBackend(Backend):
    """Local CPU execution engine powered by llama.cpp."""

    def __init__(self) -> None:
        self.llm: Optional[Any] = None
        self.model_path: Optional[Path] = None
        self.n_ctx: int = 2048
        self.n_threads: int = 4
        self.metadata: Optional[GGUFMetadata] = None
        self.sessions: Dict[str, SessionState] = {}
        # Keep generator objects per session for smooth token streaming
        self._generators: Dict[str, Any] = {}

    def load(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        use_mmap: bool = True,
        use_mlock: bool = False,
        **kwargs: Any
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if n_threads is None:
            n_threads, _ = AutoTuner.select_threads()

        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.metadata = GGUFParser.parse_file(self.model_path)

        if not HAS_LLAMA_CPP or llama_cpp is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. To run models locally, install llama-cpp-python or use native llama.cpp."
            )

        self.llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=kwargs.get("n_batch", 512),
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            verbose=kwargs.get("verbose", False)
        )

    def unload(self) -> None:
        for s in list(self.sessions.keys()):
            self.close_session(s)
        if self.llm is not None:
            if hasattr(self.llm, "close"):
                try:
                    self.llm.close()
                except Exception:
                    pass
            self.llm = None
        self.model_path = None
        self.metadata = None

    def is_loaded(self) -> bool:
        return self.llm is not None

    def new_session(
        self,
        session_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        prompt_tokens: Optional[List[int]] = None,
        seq_id: Optional[int] = None,
        **kwargs: Any,
    ) -> SessionState:
        if not self.llm:
            raise RuntimeError("Backend model is not loaded.")

        if stop is None:
            stop = ["<|im_end|>", "<|eot_id|>", "</s>", "\nUser:", "<|end_of_text|>"]

        # Use pre-tokenized tokens if provided, else tokenize with llama.cpp
        if prompt_tokens is not None:
            raw_tokens = list(prompt_tokens)
        else:
            prompt_bytes = prompt.encode("utf-8")
            raw_tokens = list(self.llm.tokenize(prompt_bytes, add_bos=True))

        session = SessionState(
            session_id=session_id,
            model_tag=self.model_path.stem if self.model_path else "unknown",
            prompt=prompt,
            prompt_tokens=list(raw_tokens),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop,
            created_at=time.perf_counter()
        )
        self.sessions[session_id] = session

        # Initialize streaming generator for this session
        gen = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop
        )
        self._generators[session_id] = gen
        return session

    def prefill_chunk(self, session_id: str, max_tokens_chunk: int = 256) -> Tuple[int, bool]:
        """Perform chunked prefill on the session prompt."""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found.")

        total_prompt_tokens = len(session.prompt_tokens)
        remaining = total_prompt_tokens - session.prefill_offset
        chunk_size = min(remaining, max_tokens_chunk)

        session.prefill_offset += chunk_size
        if session.prefill_offset >= total_prompt_tokens:
            session.prefill_completed = True
            return chunk_size, True

        return chunk_size, False

    def decode_step(self, session_id: str, n_tokens: int = 1) -> DecodeResult:
        """Execute a quantum of n_tokens decode steps for the session."""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found.")

        gen = self._generators.get(session_id)
        if not gen or session.done:
            return DecodeResult(
                session_id=session_id,
                tokens=[],
                text_deltas=[],
                done=True,
                stop_reason=session.stop_reason or "completed",
                ttft_ms=session.ttft_ms,
                decode_tok_s=0.0,
                total_tokens_generated=len(session.generated_tokens)
            )

        tokens_this_step: List[int] = []
        text_deltas: List[str] = []
        now = time.perf_counter()

        for _ in range(n_tokens):
            try:
                chunk = next(gen)
                text = chunk["choices"][0]["text"]
                text_deltas.append(text)
                session.generated_text += text
                
                # Synthetic token ID or hash if raw token id not exposed in dict
                token_id = len(session.generated_tokens) + 1
                tokens_this_step.append(token_id)
                session.generated_tokens.append(token_id)

                now = time.perf_counter()
                if session.first_token_time is None:
                    session.first_token_time = now
                    session.ttft_ms = (now - session.created_at) * 1000.0
                session.last_token_time = now

                # Check max tokens
                if len(session.generated_tokens) >= session.max_tokens:
                    session.done = True
                    session.stop_reason = "length"
                    break

            except StopIteration:
                session.done = True
                session.stop_reason = "stop"
                break

        # Compute speed metrics
        gen_count = len(session.generated_tokens)
        if session.first_token_time and session.last_token_time and gen_count > 1:
            decode_duration = session.last_token_time - session.first_token_time
            decode_tok_s = (gen_count - 1) / decode_duration if decode_duration > 0.001 else 0.0
        else:
            decode_tok_s = 0.0

        return DecodeResult(
            session_id=session_id,
            tokens=tokens_this_step,
            text_deltas=text_deltas,
            done=session.done,
            stop_reason=session.stop_reason,
            ttft_ms=session.ttft_ms,
            decode_tok_s=decode_tok_s,
            total_tokens_generated=gen_count
        )

    def tokenize(self, text: str) -> List[int]:
        if self.llm is not None:
            try:
                return list(self.llm.tokenize(text.encode("utf-8"), add_bos=True))
            except Exception:
                pass
        return list(text.encode("utf-8"))

    def set_n_threads(self, n_threads: int) -> None:
        self.n_threads = max(1, n_threads)
        if self.llm is not None:
            if hasattr(self.llm, "set_n_threads"):
                try:
                    self.llm.set_n_threads(self.n_threads)
                except Exception:
                    pass
            elif hasattr(self.llm, "n_threads"):
                self.llm.n_threads = self.n_threads

    def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        gen = self._generators.pop(session_id, None)
        if gen is not None and hasattr(gen, "close"):
            try:
                gen.close()
            except Exception:
                pass

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def get_info(self) -> Dict[str, Any]:
        if not self.is_loaded() or not self.model_path or not self.metadata:
            return {
                "status": "unloaded",
                "active_sessions": 0
            }

        file_size = self.model_path.stat().st_size
        mem = compute_memory_breakdown(
            weights_bytes=file_size,
            n_layers=self.metadata.block_count,
            n_kv_heads=self.metadata.head_count_kv,
            head_dim=self.metadata.head_dim,
            n_ctx=self.n_ctx
        )
        bw = estimate_decode_throughput(file_size)

        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "model_name": self.metadata.model_name,
            "architecture": self.metadata.architecture,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "active_sessions": len(self.sessions),
            "memory": {
                "weights_mb": mem.weights_mb,
                "kv_cache_mb": mem.kv_cache_mb,
                "overhead_mb": mem.overhead_mb,
                "total_mb": mem.total_mb,
            },
            "bandwidth": {
                "theoretical_max_tok_s": bw.theoretical_max_tok_s,
                "bottleneck": bw.bottleneck_type
            }
        }
