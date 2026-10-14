"""Backend abstract interface and session state definitions.

Designed for future migration of execution hot-paths to C / C++ / Rust / Go:
Every operation (load, unload, session creation, prefill chunking, decode quantum)
maps 1:1 to low-level C ABI functions without intermediate Python runtime locks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class SessionState:
    session_id: str
    model_tag: str
    prompt: str
    prompt_tokens: List[int]
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=list)
    prefill_completed: bool = False
    prefill_offset: int = 0
    done: bool = False
    stop_reason: Optional[str] = None
    created_at: float = 0.0
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    ttft_ms: float = 0.0

@dataclass
class DecodeResult:
    session_id: str
    tokens: List[int]
    text_deltas: List[str]
    done: bool
    stop_reason: Optional[str] = None
    ttft_ms: float = 0.0
    decode_tok_s: float = 0.0
    total_tokens_generated: int = 0

class Backend(ABC):
    """Abstract interface for LLM execution engines."""

    @abstractmethod
    def load(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        use_mmap: bool = True,
        use_mlock: bool = False,
        **kwargs: Any
    ) -> None:
        """Load model weights and initialize computational context."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Free model weights, KV caches, and compute buffers."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is resident in memory."""
        pass

    @abstractmethod
    def new_session(
        self,
        session_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> SessionState:
        """Initialize an inference session with prompt tokenization and KV cache reservation."""
        pass

    @abstractmethod
    def prefill_chunk(self, session_id: str, max_tokens_chunk: int = 256) -> Tuple[int, bool]:
        """Process a chunk of prompt tokens (chunked prefill).
        
        Returns: (tokens_processed_in_this_step, is_prefill_complete)
        """
        pass

    @abstractmethod
    def decode_step(self, session_id: str, n_tokens: int = 1) -> DecodeResult:
        """Generate up to n_tokens for the session (token quantum execution)."""
        pass

    @abstractmethod
    def close_session(self, session_id: str) -> None:
        """Release session KV cache and resources."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve active session state by session_id."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return operational metrics, memory allocation, and active sessions count."""
        pass

    def set_n_threads(self, n_threads: int) -> None:
        """Dynamically update CPU execution threads for active context."""
        pass

