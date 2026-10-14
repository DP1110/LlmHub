"""Single-model execution engine using llama-cpp-python.

Features:
- Streaming token generator with sub-millisecond yield
- TTFT (Time To First Token) and decode tok/s tracking
- Memory estimation and physical core thread tuning
- Interactive REPL and one-shot prompt support
"""

import sys
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
            import shutil
            self._cli_binary = shutil.which("llama-cli") or shutil.which("llama") or shutil.which("main")
            if not self._cli_binary:
                raise RuntimeError(
                    "Neither 'llama-cpp-python' nor native 'llama-cli' binary was found.\n"
                    "To enable model inference in Termux, install native llama.cpp with:\n"
                    "  pkg install llama-cpp\n"
                    "Or compile it directly with:\n"
                    "  pkg install git cmake clang make -y\n"
                    "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j4"
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

        if self.llm is None and getattr(self, "_cli_binary", None):
            import subprocess
            cmd = [
                self._cli_binary,
                "-m", str(self.model_path),
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "--top-p", str(top_p),
                "-t", str(self.n_threads),
                "-c", str(self.n_ctx),
                "--no-display-prompt"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            start_time = time.perf_counter()
            first_token_time = None
            ttft_ms = 0.0
            token_count = 0
            for line in proc.stdout:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    ttft_ms = (first_token_time - start_time) * 1000.0
                token_count += len(line.split())
                decode_duration = (now - first_token_time) if first_token_time else 0.0
                decode_tok_s = (token_count / decode_duration) if decode_duration > 0.001 else 0.0
                yield {
                    "token": line,
                    "token_count": token_count,
                    "ttft_ms": ttft_ms,
                    "decode_tok_s": decode_tok_s,
                    "total_elapsed_s": now - start_time,
                    "done": False
                }
            proc.wait()
            yield {
                "token": "",
                "token_count": token_count,
                "ttft_ms": ttft_ms,
                "decode_tok_s": decode_tok_s,
                "total_elapsed_s": time.perf_counter() - start_time,
                "done": True
            }
            return

        if stop is None:
            stop = ["<|im_end|>", "<|eot_id|>", "</s>", "\nUser:", "<|end_of_text|>"]

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

            for chunk in self.stream_completion(
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                if not chunk["done"]:
                    sys.stdout.write(chunk["token"])
                    sys.stdout.flush()
                    assistant_response += chunk["token"]
                last_stats = chunk

            sys.stdout.write("\n\n")
            if last_stats:
                sys.stdout.write(
                    f"[{last_stats.get('token_count', 0)} tokens | "
                    f"TTFT: {last_stats.get('ttft_ms', 0):.1f}ms | "
                    f"Speed: {last_stats.get('decode_tok_s', 0):.1f} tok/s | "
                    f"Time: {last_stats.get('total_elapsed_s', 0):.2f}s]\n\n"
                )

            messages.append({"role": "assistant", "content": assistant_response.strip()})
