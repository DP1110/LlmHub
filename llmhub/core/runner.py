"""Single-model execution engine using llama-cpp-python.

Features:
- Streaming token generator with sub-millisecond yield
- TTFT (Time To First Token) and decode tok/s tracking
- Memory estimation and physical core thread tuning
- Interactive REPL and one-shot prompt support
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    llama_cpp = None
    HAS_LLAMA_CPP = False

from llmhub.config import LLMHubConfig
from llmhub.core.gguf_parser import GGUFParser, GGUFMetadata
from llmhub.core.registry import ModelRegistry, ModelRecord
from llmhub.core.template import ChatTemplateEngine
from llmhub.utils.math_calc import compute_memory_breakdown, estimate_decode_throughput
from llmhub.utils.platform import get_hardware_profile

_CLI_FLAGS_CACHE: Dict[str, Any] = {}
_CLI_FLAGS_LOCK = threading.Lock()

def get_cli_supported_flags(cli_binary: str) -> Any:
    """Inspect and cache supported command-line flags for the llama-cli executable."""
    with _CLI_FLAGS_LOCK:
        if cli_binary in _CLI_FLAGS_CACHE:
            return _CLI_FLAGS_CACHE[cli_binary]

        supported = set()
        probed_ok = False
        try:
            import subprocess
            import re
            res = subprocess.run(
                [cli_binary, "-h"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15.0,
                check=False
            )
            help_text = (res.stdout or "") + "\n" + (res.stderr or "")
            if len(help_text.strip()) < 100:
                res = subprocess.run(
                    [cli_binary, "--help"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15.0,
                    check=False
                )
                help_text += "\n" + (res.stdout or "") + "\n" + (res.stderr or "")

            if help_text.strip():
                matched = set(re.findall(r"(--?[\w-]+)", help_text))
                if len(matched) > 5:
                    supported = matched
                    probed_ok = True
        except Exception:
            pass

        # Universal default fallback if flags couldn't be discovered (e.g. timeout or minimal mock)
        is_cli = "llama-cli" in cli_binary.lower() or "llama-run" in cli_binary.lower()
        default_flags = {
            "-m", "-p", "-f", "-c", "-t", "-n", "--temp",
            "--log-disable", "--special"
        }
        if is_cli:
            default_flags.update({"-st", "--single-turn", "-no-cnv", "--no-conversation", "--no-display-prompt"})
        if not supported:
            supported.update(default_flags)

        # Always cache result to prevent repeated 15s probe timeouts on slow mobile devices
        _CLI_FLAGS_CACHE[cli_binary] = supported
        return supported

ASSISTANT_HEADERS = (
    "<|im_start|>assistant\n",
    "<|im_start|>assistant",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>",
    "### Response:\n",
    "### Response:",
    "[/INST] ",
    "[/INST]",
    "\nAssistant:\n",
    "\nAssistant: ",
    "\nAssistant:",
    "Assistant: ",
    "Assistant:",
)

def strip_assistant_header(text: str) -> Optional[str]:
    """Find known assistant turn headers and return text after the last header."""
    best_pos = -1
    best_len = 0
    for h in ASSISTANT_HEADERS:
        pos = text.rfind(h)
        if pos != -1:
            if pos > best_pos or (pos == best_pos and len(h) > best_len):
                best_pos = pos
                best_len = len(h)
    if best_pos != -1:
        return text[best_pos + best_len:].lstrip("\r\n")
    return None

def ends_with_stop_prefix(text: str, stop_seqs: Optional[List[str]]) -> bool:
    """Return True if text ends with a non-empty prefix of any stop sequence."""
    if not text or not stop_seqs:
        return False
    for s in stop_seqs:
        for i in range(1, len(s) + 1):
            prefix = s[:i]
            # Don't treat a lone whitespace character as a stop prefix
            if prefix in ("\n", " ", "\t") and len(s) > 1:
                continue
            if text.endswith(prefix):
                return True
    return False

class ModelRunner:
    """Manages loading, inference, and streaming for a single GGUF model."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        use_mmap: bool = True,
        use_mlock: bool = False,
        verbose: bool = False
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Hardware profiling
        hw = get_hardware_profile()
        self.n_threads = n_threads or hw.suggested_threads
        self.n_ctx = n_ctx
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.verbose = verbose
        self._cli_binary: Optional[str] = None

        # Parse GGUF metadata for chat template and layer info
        self.metadata: GGUFMetadata = GGUFParser.parse_file(self.model_path)
        self.template_engine = ChatTemplateEngine(
            template_str=self.metadata.chat_template,
            architecture=self.metadata.architecture
        )

        # Initialize llama.cpp Llama instance
        self.llm: Optional[Any] = None
        self._load_model()

    def _load_model(self) -> None:
        """Instantiate llama.cpp context with mmap and thread settings, with llama-cli fallback."""
        if HAS_LLAMA_CPP and llama_cpp is not None:
            self.llm = llama_cpp.Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=512,
                use_mmap=self.use_mmap,
                use_mlock=self.use_mlock,
                verbose=self.verbose
            )
        else:
            import os
            import shutil
            import subprocess

            candidates = []
            for name in ["llama-completion", "llama-cli", "llama", "llama-run"]:
                w = shutil.which(name)
                if w and w not in candidates:
                    candidates.append(w)
            if sys.platform != "win32":
                w = shutil.which("main")
                if w and w not in candidates:
                    candidates.append(w)

            bin_name = "llama-cli.exe" if sys.platform == "win32" else "llama-cli"
            comp_name = "llama-completion.exe" if sys.platform == "win32" else "llama-completion"
            main_name = "main.exe" if sys.platform == "win32" else "main"
            for p in [
                Path.home() / "llama.cpp" / "build" / "bin" / comp_name,
                Path.home() / "llama.cpp" / "build" / "bin" / bin_name,
                Path.home() / "llama.cpp" / "build" / "bin" / "Release" / comp_name,
                Path.home() / "llama.cpp" / "build" / "bin" / "Release" / bin_name,
                Path.home() / "llama.cpp" / "build" / "bin" / main_name,
                Path("/data/data/com.termux/files/home/llama.cpp/build/bin/llama-completion"),
                Path("/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli"),
                Path("/data/data/com.termux/files/usr/bin/llama-completion"),
                Path("/data/data/com.termux/files/usr/bin/llama-cli"),
                Path("/opt/homebrew/bin/llama-completion"),
                Path("/opt/homebrew/bin/llama-cli"),
                Path("/usr/local/bin/llama-completion"),
                Path("/usr/local/bin/llama-cli"),
            ]:
                if p.is_file() and (sys.platform == "win32" or os.access(p, os.X_OK)):
                    sp = str(p)
                    if sp not in candidates:
                        candidates.append(sp)

            def is_runnable(path: str) -> bool:
                try:
                    res = subprocess.run(
                        [path, "-h"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=3.0
                    )
                    out = (res.stdout or b"") + (res.stderr or b"")
                    err = out.decode("utf-8", errors="replace")
                    if "CANNOT LINK EXECUTABLE" in err or "cannot locate symbol" in err:
                        return False
                    return res.returncode == 0 or "usage" in err.lower() or "options" in err.lower()
                except Exception:
                    return False

            # Prioritize a candidate that is actively runnable without dynamic linker errors
            chosen = None
            for c in candidates:
                if is_runnable(c):
                    chosen = c
                    break

            # Fallback to first available candidate if none actively runnable (e.g. mocked in tests)
            self._cli_binary = chosen or (candidates[0] if candidates else None)
            if not self._cli_binary:
                raise RuntimeError(
                    "Neither 'llama-cpp-python' nor native 'llama-cli' binary was found.\n"
                    "To enable model inference in Termux, install native llama.cpp with:\n"
                    "  pkg install libc++ llama-cpp\n"
                    "Or compile it directly with:\n"
                    "  bash install-termux.sh"
                )

    def close(self) -> None:
        """Explicitly free native model resources."""
        if self.llm is not None:
            if hasattr(self.llm, "close"):
                try:
                    self.llm.close()
                except Exception:
                    pass
            self.llm = None

    def __enter__(self) -> "ModelRunner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_memory_info(self) -> Dict[str, Any]:
        """Compute memory profile and theoretical decode speed."""
        file_size = self.model_path.stat().st_size
        breakdown = compute_memory_breakdown(
            weights_bytes=file_size,
            n_layers=self.metadata.block_count,
            n_kv_heads=self.metadata.head_count_kv,
            head_dim=self.metadata.head_dim,
            n_ctx=self.n_ctx,
            bytes_per_element=2  # default FP16 KV
        )
        bandwidth = estimate_decode_throughput(file_size)
        return {
            "weights_mb": breakdown.weights_mb,
            "kv_cache_mb": breakdown.kv_cache_mb,
            "overhead_mb": breakdown.overhead_mb,
            "total_mb": breakdown.total_mb,
            "theoretical_max_tok_s": bandwidth.theoretical_max_tok_s,
            "bottleneck": bandwidth.bottleneck_type,
            "explanation": bandwidth.explanation
        }

    def stream_completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream generated tokens while recording TTFT and tok/s metrics."""
        if not self.llm and not getattr(self, "_cli_binary", None):
            raise RuntimeError("Model is not loaded.")

        if stop is None:
            stop = ["<|im_end|>", "<|eot_id|>", "</s>", "\nUser:", "<|end_of_text|>"]

        if self.llm is None and getattr(self, "_cli_binary", None):
            import subprocess
            import threading
            import uuid
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
            # Conversation-mode flags are ONLY for llama-cli / llama-run.
            # llama-completion is already a pure completion binary and these flags
            # cause it to silently produce 0 tokens on Termux ARM64.
            is_completion_bin = "llama-completion" in self._cli_binary.lower()

            if not is_completion_bin:
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
            elif "-no-display-prompt" in flags:
                cmd.append("-no-display-prompt")
                no_display_prompt_active = True

            if "--log-disable" in flags:
                cmd.append("--log-disable")
            elif "-ld" in flags:
                cmd.append("-ld")

            if "--special" in flags:
                cmd.append("--special")

            # Note: Do NOT pass -r/--reverse-prompt to native llama-cli. Modern instruct models
            # have stop tokens (like <|im_end|>, <|eot_id|>) embedded in their system/user prompt.
            # Passing -r triggers llama-cli to treat the prompt as an antiprompt, immediately halting
            # generation and yielding to stdin (which hits DEVNULL EOF and produces 0 tokens).
            # Python streaming handles stop sequences reliably on generated text.

            env = dict(os.environ)
            env["TERM"] = "dumb"
            env["NO_COLOR"] = "1"

            popen_kwargs: Dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
                "env": env,
            }
            if sys.platform != "win32":
                popen_kwargs["start_new_session"] = True

            proc = subprocess.Popen(cmd, **popen_kwargs)
            stderr_chunks: List[str] = []
            def _drain_stderr() -> None:
                try:
                    if proc.stderr:
                        for line in proc.stderr:
                            stderr_chunks.append(line)
                except Exception:
                    pass
            stderr_t = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_t.start()

            start_time = time.perf_counter()
            first_token_time = None
            ttft_ms = 0.0
            token_count = 0
            accumulated_output = ""
            hit_stop = False
            try:
                # Fluid token/character-based chunk streaming with prompt echo filtering
                buffer = ""
                initial_buffer = ""
                prompt_stripped = not bool(prompt)
                clean_prompt = prompt.strip() if prompt else ""
                while not hit_stop:
                    char = proc.stdout.read(1)
                    if not char:
                        break

                    # Detect and strip prompt if llama-cli echoed it
                    if not prompt_stripped:
                        initial_buffer += char

                        # 1. Clean match: clean_prompt appeared in initial_buffer
                        if clean_prompt and clean_prompt in initial_buffer:
                            prompt_stripped = True
                            remainder = initial_buffer.split(clean_prompt, 1)[1]
                            hdr_rem = strip_assistant_header(remainder)
                            if hdr_rem is not None:
                                buffer = hdr_rem
                            else:
                                buffer = remainder.lstrip("\r\n >\t")
                            accumulated_output = buffer
                            initial_buffer = ""
                            continue

                        # 2. In-progress prefix match (strip prompt leaders like "> ")
                        cand = initial_buffer.lstrip("\r\n >\t")
                        if not cand:
                            continue

                        # Prioritize in-progress prefix match while binary is still echoing prompt
                        if clean_prompt and clean_prompt.startswith(cand):
                            continue

                        # 4. Check if clean_prompt start appears after some banner or menu
                        prompt_head = clean_prompt[:min(30, len(clean_prompt))] if clean_prompt else ""
                        if prompt_head and prompt_head in initial_buffer:
                            initial_buffer = initial_buffer[initial_buffer.find(prompt_head):]
                            continue

                        # 5. If candidate matches a known banner, continue buffering up to 256 bytes
                        known_banners = ("/exit", "/quit", "/regen", "/clear", "/reset", "/read", "/glob")
                        is_banner = any(b.startswith(cand) or cand.startswith(b) for b in known_banners)
                        if is_banner and len(initial_buffer) < 256:
                            continue

                        # 6. Diverged from prompt prefix and not a banner: output is genuine generated text
                        prompt_stripped = True
                        hdr_div = strip_assistant_header(initial_buffer)
                        if hdr_div is not None:
                            buffer = hdr_div
                            accumulated_output = hdr_div
                        else:
                            buffer = initial_buffer.lstrip("\r\n >\t")
                            accumulated_output = buffer
                        initial_buffer = ""
                    else:
                        buffer += char
                        accumulated_output += char

                    # Check stop sequences only on assistant generated text (never inside prompt)
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
                        if char in (" ", "\n", "\t") or len(buffer) >= 8:
                            token_count += 1
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                                ttft_ms = (first_token_time - start_time) * 1000.0
                            decode_duration = (now - first_token_time) if first_token_time else 0.0
                            decode_tok_s = (token_count / decode_duration) if decode_duration > 0.001 else 0.0
                            yield {
                                "token": buffer,
                                "token_count": token_count,
                                "ttft_ms": ttft_ms,
                                "decode_tok_s": decode_tok_s,
                                "total_elapsed_s": now - start_time,
                                "done": False
                            }
                            buffer = ""

                # Flush any remaining unstripped buffer if EOF was reached prematurely (CRIT-2)
                if not prompt_stripped and initial_buffer:
                    if clean_prompt and clean_prompt in initial_buffer:
                        buffer = initial_buffer.split(clean_prompt, 1)[1].lstrip("\r\n >\t") + buffer
                    else:
                        hdr = strip_assistant_header(initial_buffer)
                        if hdr is not None:
                            buffer = hdr + buffer
                        else:
                            buffer = initial_buffer + buffer
                    initial_buffer = ""
                    prompt_stripped = True

                # Strip any stop sequences present at premature EOF flush (CRIT-4)
                if buffer and stop:
                    for s in stop:
                        if s in buffer:
                            buffer = buffer[:buffer.rfind(s)]

                # Yield remaining buffer if any
                if buffer:
                    token_count += 1
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                        ttft_ms = (first_token_time - start_time) * 1000.0
                    decode_duration = (now - first_token_time) if first_token_time else 0.0
                    decode_tok_s = (token_count / decode_duration) if decode_duration > 0.001 else 0.0
                    yield {
                        "token": buffer,
                        "token_count": token_count,
                        "ttft_ms": ttft_ms,
                        "decode_tok_s": decode_tok_s,
                        "total_elapsed_s": now - start_time,
                        "done": False
                    }
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except (subprocess.TimeoutExpired, Exception):
                        proc.kill()
                        try:
                            proc.wait(timeout=2.0)
                        except (subprocess.TimeoutExpired, Exception):
                            pass
                else:
                    try:
                        proc.wait(timeout=1.0)
                    except (subprocess.TimeoutExpired, Exception):
                        pass
                stderr_t.join(timeout=1.0)
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

            total_elapsed = time.perf_counter() - start_time
            decode_duration = (total_elapsed - (ttft_ms / 1000.0)) if first_token_time else total_elapsed
            final_tok_s = (token_count / decode_duration) if decode_duration > 0.001 else 0.0

            rc = proc.returncode if isinstance(proc.returncode, int) else proc.poll()
            if isinstance(rc, int) and rc != 0 and not hit_stop:
                err_text = "".join(stderr_chunks).strip()
                extra_hint = ""
                if rc in (-9, 137, 9):
                    extra_hint = " (Process terminated by SIGKILL, likely Android Low Memory Killer OOM)"
                elif "cannot locate symbol" in err_text or "CANNOT LINK EXECUTABLE" in err_text:
                    extra_hint = (
                        " (Android dynamic linker ABI mismatch: outdated Termux libc++)\n"
                        "To fix this immediately, run in Termux:\n"
                        "  pkg install -y libc++\n"
                        "Or re-run the installer to compile from source:\n"
                        "  bash install-termux.sh"
                    )
                raise RuntimeError(
                    f"Native llama-cli failed (exit code {rc}{extra_hint}).\n"
                    f"Tokens yielded before crash: {token_count}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Details:\n{err_text or 'No stderr output returned.'}"
                )
            elif isinstance(rc, int) and rc == 0 and token_count == 0 and not hit_stop:
                err_text = "".join(stderr_chunks).strip()
                import logging
                logging.getLogger("llmhub.core.runner").warning(
                    "Native llama-cli finished with exit code 0 but yielded 0 tokens.\nCommand: %s\nStderr: %s",
                    " ".join(cmd),
                    err_text or "No stderr"
                )

            yield {
                "token": "",
                "token_count": token_count,
                "ttft_ms": ttft_ms,
                "decode_tok_s": final_tok_s,
                "total_elapsed_s": total_elapsed,
                "done": True
            }
            return

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        ttft_ms: float = 0.0
        token_count = 0

        # Generate stream from llama-cpp-python
        generator = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop
        )

        for chunk in generator:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
                ttft_ms = (first_token_time - start_time) * 1000.0

            text_delta = chunk["choices"][0]["text"]
            token_count += 1
            
            # Decode time excludes TTFT prefill latency
            decode_duration = (now - first_token_time) if first_token_time else 0.0
            if token_count > 1 and decode_duration > 0.001:
                decode_tok_s = (token_count - 1) / decode_duration
            else:
                decode_tok_s = 0.0

            yield {
                "token": text_delta,
                "token_count": token_count,
                "ttft_ms": ttft_ms,
                "decode_tok_s": decode_tok_s,
                "total_elapsed_s": now - start_time,
                "done": False
            }

        total_time = time.perf_counter() - start_time
        final_decode_time = (total_time - (ttft_ms / 1000.0)) if first_token_time else total_time
        if token_count > 1 and final_decode_time > 0.001:
            final_tok_s = (token_count - 1) / final_decode_time
        elif token_count == 1 and total_time > 0:
            final_tok_s = 1.0 / total_time
        else:
            final_tok_s = 0.0

        yield {
            "token": "",
            "token_count": token_count,
            "ttft_ms": ttft_ms,
            "decode_tok_s": final_tok_s,
            "total_elapsed_s": total_time,
            "done": True
        }

    def chat_repl(
        self,
        system_prompt: Optional[str] = "You are a helpful and concise offline AI assistant.",
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> None:
        """Run an interactive multi-turn terminal chat session."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        mem = self.get_memory_info()
        sys.stdout.write(f"\n=================================================================\n")
        sys.stdout.write(f" LLMHub Interactive Runner: {self.metadata.model_name}\n")
        sys.stdout.write(f" Threads: {self.n_threads} | Context: {self.n_ctx} | Weights: {mem['weights_mb']:.1f} MB | KV: {mem['kv_cache_mb']:.1f} MB\n")
        sys.stdout.write(f" Theoretical Peak Decode: {mem['theoretical_max_tok_s']:.1f} tok/s ({mem['bottleneck']})\n")
        sys.stdout.write(f" Type '/exit' or '/quit' to leave. '/reset' to clear chat.\n")
        sys.stdout.write(f"=================================================================\n\n")

        while True:
            try:
                user_input = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                sys.stdout.write("\nExiting chat session.\n")
                break

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                break
            elif user_input.lower() == "/reset":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                sys.stdout.write("Chat history reset.\n\n")
                continue

            messages.append({"role": "user", "content": user_input})
            formatted_prompt = self.template_engine.render(messages, add_generation_prompt=True)

            sys.stdout.write("\n")
            assistant_response = ""
            last_stats = {}
            first_token = True
            is_tty = sys.stdout.isatty()
            if is_tty:
                sys.stderr.write("[LLMHub] Thinking...")
                sys.stderr.flush()

            for chunk in self.stream_completion(
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                if not chunk["done"]:
                    if first_token:
                        if is_tty:
                            sys.stderr.write("\r" + " " * 40 + "\r")
                            sys.stderr.flush()
                        first_token = False
                    sys.stdout.write(chunk["token"])
                    sys.stdout.flush()
                    assistant_response += chunk["token"]
                last_stats = chunk

            if first_token and is_tty:
                sys.stderr.write("\r" + " " * 40 + "\r")
                sys.stderr.flush()

            sys.stdout.write("\n\n")
            if last_stats:
                sys.stdout.write(
                    f"[{last_stats.get('token_count', 0)} tokens | "
                    f"TTFT: {last_stats.get('ttft_ms', 0):.1f}ms | "
                    f"Speed: {last_stats.get('decode_tok_s', 0):.1f} tok/s | "
                    f"Time: {last_stats.get('total_elapsed_s', 0):.2f}s]\n\n"
                )

            messages.append({"role": "assistant", "content": assistant_response.strip()})
