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
    """Local CPU execution engine powered by llama.cpp (with llama_cpp and native llama-cli support)."""

    def __init__(self) -> None:
        self.llm: Optional[Any] = None
        self._cli_binary: Optional[str] = None
        self.model_path: Optional[Path] = None
        self.n_ctx: int = 2048
        self.n_threads: int = 4
        self.metadata: Optional[GGUFMetadata] = None
        self.sessions: Dict[str, SessionState] = {}
        # Keep generator objects and subprocesses per session for smooth token streaming
        self._generators: Dict[str, Any] = {}
        self._processes: Dict[str, Any] = {}

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

        if HAS_LLAMA_CPP and llama_cpp is not None:
            self.llm = llama_cpp.Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=kwargs.get("n_batch", 512),
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                verbose=kwargs.get("verbose", False)
            )
        else:
            import os
            import shutil
            import sys
            cand = (
                shutil.which("llama-cli")
                or shutil.which("llama")
                or shutil.which("llama-run")
                or shutil.which("llama-completion")
            )
            if not cand and sys.platform != "win32":
                cand = shutil.which("main")
            if not cand:
                bin_name = "llama-cli.exe" if sys.platform == "win32" else "llama-cli"
                main_name = "main.exe" if sys.platform == "win32" else "main"
                for p in [
                    Path.home() / "llama.cpp" / "build" / "bin" / bin_name,
                    Path.home() / "llama.cpp" / "build" / "bin" / "Release" / bin_name,
                    Path.home() / "llama.cpp" / "build" / "bin" / main_name,
                    Path("/opt/homebrew/bin/llama-cli"),
                    Path("/usr/local/bin/llama-cli"),
                    Path("/data/data/com.termux/files/usr/bin/llama-cli"),
                    Path("/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli"),
                ]:
                    if p.is_file() and (sys.platform == "win32" or os.access(p, os.X_OK)):
                        cand = str(p)
                        break
            self._cli_binary = cand
            if not self._cli_binary:
                raise RuntimeError(
                    "Neither 'llama-cpp-python' nor native 'llama-cli' binary was found.\n"
                    "To enable model inference in Termux, install native llama.cpp with:\n"
                    "  pkg install llama-cpp"
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
        self._cli_binary = None
        self.model_path = None
        self.metadata = None

    def is_loaded(self) -> bool:
        return (self.llm is not None) or (self._cli_binary is not None)

    def _create_cli_generator(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Tuple[Any, Any]:
        """Spawn a streaming generator wrapping native llama-cli subprocess."""
        import subprocess
        import threading
        import uuid
        from llmhub.core.runner import get_cli_supported_flags, ends_with_stop_prefix

        flags = get_cli_supported_flags(self._cli_binary)

        prompt_file: Optional[Path] = None
        if len(prompt) > 4096:
            temp_dir = Path.home() / ".llmhub" / "run"
            temp_dir.mkdir(parents=True, exist_ok=True)
            prompt_file = temp_dir / f"prompt_{uuid.uuid4().hex[:8]}.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
            prompt_args = ["-f", str(prompt_file)]
        else:
            prompt_args = ["-p", prompt]

        cmd = [
            self._cli_binary,
            "-m", str(self.model_path),
        ] + prompt_args + [
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "-t", str(self.n_threads),
            "-c", str(self.n_ctx),
        ]
        if "-no-cnv" in flags:
            cmd.append("-no-cnv")
        elif "--no-conversation" in flags:
            cmd.append("--no-conversation")

        if "-st" in flags:
            cmd.append("-st")
        elif "--single-turn" in flags:
            cmd.append("--single-turn")

        no_display_prompt_active = False
        if "--no-display-prompt" in flags:
            cmd.append("--no-display-prompt")
            no_display_prompt_active = True
        if "--log-disable" in flags:
            cmd.append("--log-disable")
        if "--special" in flags:
            cmd.append("--special")

        has_reverse_prompt = ("-r" in flags) or ("--reverse-prompt" in flags)
        if has_reverse_prompt and stop:
            for s in stop:
                cmd.extend(["-r", s])

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        stderr_chunks: List[str] = []
        def _drain_stderr() -> None:
            try:
                if proc.stderr:
                    for line in proc.stderr:
                        stderr_chunks.append(line)
            except Exception:
                pass

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()

        def _gen():
            buffer = ""
            initial_buffer = ""
            prompt_stripped = not bool(prompt) or no_display_prompt_active
            clean_prompt = prompt.strip() if prompt else ""
            tokens_yielded = 0
            accumulated_output = ""
            hit_stop = False
            try:
                while not hit_stop:
                    char = proc.stdout.read(1)
                    if not char:
                        break

                    # Detect and strip echoed prompt
                    if not prompt_stripped:
                        initial_buffer += char
                        cand = initial_buffer.lstrip("\r\n ")
                        if clean_prompt and clean_prompt.startswith(cand):
                            if cand == clean_prompt:
                                prompt_stripped = True
                                initial_buffer = ""
                                accumulated_output = ""
                            continue
                        elif clean_prompt and clean_prompt in initial_buffer:
                            prompt_stripped = True
                            remainder = initial_buffer.split(clean_prompt, 1)[1]
                            buffer = remainder.lstrip("\r\n ")
                            accumulated_output = buffer
                            initial_buffer = ""
                        else:
                            # Diverged from prompt prefix: output is generated text, not prompt echo
                            prompt_stripped = True
                            buffer = initial_buffer
                            accumulated_output = initial_buffer
                            initial_buffer = ""
                    else:
                        buffer += char
                        accumulated_output += char

                    # Check stop sequences only on assistant generated text
                    if prompt_stripped and stop:
                        for s in stop:
                            if s in accumulated_output:
                                stop_idx = accumulated_output.rfind(s)
                                chars_already_yielded = len(accumulated_output) - len(buffer)
                                if stop_idx >= chars_already_yielded:
                                    buffer = buffer[:(stop_idx - chars_already_yielded)]
                                else:
                                    buffer = ""
                                hit_stop = True
                                break
                    if hit_stop:
                        break

                    is_stop_prefix = prompt_stripped and stop and ends_with_stop_prefix(buffer, stop)
                    if not is_stop_prefix:
                        if char in (" ", "\n", "\t") or len(buffer) >= 6:
                            tokens_yielded += 1
                            yield {"choices": [{"text": buffer}]}
                            buffer = ""

                # Flush any remaining unstripped buffer if EOF was reached prematurely (CRIT-2)
                if not prompt_stripped and initial_buffer:
                    buffer = initial_buffer + buffer
                    initial_buffer = ""
                    prompt_stripped = True

                if buffer:
                    tokens_yielded += 1
                    yield {"choices": [{"text": buffer}]}
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except (subprocess.TimeoutExpired, Exception):
                        proc.kill()
                        proc.wait()
                else:
                    try:
                        proc.wait(timeout=0.5)
                    except Exception:
                        pass
                t.join(timeout=0.5)
                for stream in (proc.stdout, proc.stderr):
                    if stream:
                        try:
                            stream.close()
                        except Exception:
                            pass
                if prompt_file and prompt_file.is_file():
                    try:
                        prompt_file.unlink()
                    except Exception:
                        pass

            # Check if process crashed with non-zero exit code
            rc = proc.returncode if isinstance(proc.returncode, int) else proc.poll()
            if isinstance(rc, int) and rc != 0 and not hit_stop:
                err_text = "".join(stderr_chunks).strip()
                extra_hint = ""
                if rc in (-9, 137, 9):
                    extra_hint = " (Process terminated by SIGKILL, likely Android Low Memory Killer OOM)"
                raise RuntimeError(
                    f"Native llama-cli failed (exit code {rc}{extra_hint}).\n"
                    f"Tokens yielded before crash: {tokens_yielded}\n"
                    f"Details:\n{err_text or 'No stderr'}"
                )

        return _gen(), proc

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
        if not self.is_loaded():
            raise RuntimeError("Backend model is not loaded.")

        if stop is None:
            stop = ["<|im_end|>", "<|eot_id|>", "</s>", "\nUser:", "<|end_of_text|>"]

        # Use pre-tokenized tokens if provided, else tokenize with llama.cpp or fallback
        if prompt_tokens is not None:
            raw_tokens = list(prompt_tokens)
        else:
            raw_tokens = self.tokenize(prompt)

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

        if self.llm is not None:
            gen = self.llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop=stop
            )
            self._generators[session_id] = gen
        else:
            gen, proc = self._create_cli_generator(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop
            )
            self._generators[session_id] = gen
            self._processes[session_id] = proc

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
        proc = self._processes.pop(session_id, None)
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait()
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
