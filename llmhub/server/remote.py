"""Remote HTTP execution backend for LLMHub distributed nodes and OpenAI-compatible endpoints."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx

from llmhub.server.backend import Backend, DecodeResult, SessionState

class RemoteBackend(Backend):
    """Executes inference via an upstream remote LLMHub instance or OpenAI endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name: Optional[str] = None
        self._is_loaded: bool = False
        self.sessions: Dict[str, SessionState] = {}
        self._streams: Dict[str, Any] = {}
        self._clients: Dict[str, httpx.Client] = {}

    def load(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        use_mmap: bool = True,
        use_mlock: bool = False,
        **kwargs: Any
    ) -> None:
        self.model_name = str(model_path.name) if isinstance(model_path, Path) else str(model_path)
        # Verify connectivity to remote server
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{self.base_url}/healthz", headers=headers)
            if resp.status_code not in (200, 404):
                raise ConnectionError(f"Failed to connect to remote endpoint at {self.base_url}: HTTP {resp.status_code}")
        self._is_loaded = True

    def unload(self) -> None:
        for s in list(self.sessions.keys()):
            self.close_session(s)
        self._is_loaded = False
        self.model_name = None

    def is_loaded(self) -> bool:
        return self._is_loaded

    def new_session(
        self,
        session_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> SessionState:
        if not self._is_loaded:
            raise RuntimeError("Remote backend is not connected.")

        session = SessionState(
            session_id=session_id,
            model_tag=self.model_name or "remote",
            prompt=prompt,
            prompt_tokens=[1] * (len(prompt) // 4 + 1),  # Estimated token count
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop or [],
            created_at=time.perf_counter()
        )
        self.sessions[session_id] = session

        # Open streaming connection to remote /v1/chat/completions
        client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._clients[session_id] = client

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name or "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stop": stop
        }

        req = client.build_request("POST", f"{self.base_url}/v1/chat/completions", json=payload, headers=headers)
        resp = client.send(req, stream=True)
        if resp.status_code != 200:
            body = resp.read().decode("utf-8", errors="replace")
            resp.close()
            client.close()
            raise RuntimeError(f"Remote server error HTTP {resp.status_code}: {body}")

        # Store lines iterator
        self._streams[session_id] = (resp, iter(resp.iter_lines()))
        return session

    def prefill_chunk(self, session_id: str, max_tokens_chunk: int = 256) -> Tuple[int, bool]:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found.")
        session.prefill_completed = True
        return len(session.prompt_tokens), True

    def decode_step(self, session_id: str, n_tokens: int = 1) -> DecodeResult:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found.")

        stream_data = self._streams.get(session_id)
        if not stream_data or session.done:
            return DecodeResult(
                session_id=session_id,
                tokens=[],
                text_deltas=[],
                done=True,
                stop_reason=session.stop_reason or "completed",
                ttft_ms=session.ttft_ms,
                total_tokens_generated=len(session.generated_tokens)
            )

        resp, line_iter = stream_data
        tokens_this_step: List[int] = []
        text_deltas: List[str] = []

        while len(tokens_this_step) < n_tokens and not session.done:
            try:
                line = next(line_iter)
            except (StopIteration, GeneratorExit):
                session.done = True
                session.stop_reason = "stop"
                break
            except Exception:
                session.done = True
                session.stop_reason = "error"
                break

            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    session.done = True
                    session.stop_reason = "stop"
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text = delta.get("content", "")
                    finish_reason = choices[0].get("finish_reason")
                    if text:
                        text_deltas.append(text)
                        token_id = len(session.generated_tokens) + 1
                        tokens_this_step.append(token_id)
                        session.generated_tokens.append(token_id)
                        session.generated_text += text

                        now = time.perf_counter()
                        if session.first_token_time is None:
                            session.first_token_time = now
                            session.ttft_ms = (now - session.created_at) * 1000.0
                        session.last_token_time = now
                    if finish_reason:
                        session.done = True
                        session.stop_reason = finish_reason
                        break

        gen_count = len(session.generated_tokens)
        if session.first_token_time and session.last_token_time and gen_count > 1:
            duration = session.last_token_time - session.first_token_time
            decode_tok_s = (gen_count - 1) / duration if duration > 0.001 else 0.0
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

    def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        stream_data = self._streams.pop(session_id, None)
        if stream_data:
            resp, _ = stream_data
            try:
                resp.close()
            except Exception:
                pass
        client = self._clients.pop(session_id, None)
        if client:
            try:
                client.close()
            except Exception:
                pass

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def get_info(self) -> Dict[str, Any]:
        return {
            "status": "remote" if self._is_loaded else "unloaded",
            "base_url": self.base_url,
            "model_name": self.model_name,
            "active_sessions": len(self.sessions)
        }
