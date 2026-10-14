"""Unified CLI for LLMHub.

Commands:
- pull: Download GGUF models from Hugging Face with resume and SHA256 verification
- run:  Execute model in interactive REPL or one-shot prompt with streaming
- list: List locally downloaded models and their specifications
- show: Display deep GGUF architecture metadata, memory physics, and bandwidth limits
- rm:   Remove model and cleanup registry
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from llmhub.config import LLMHubConfig
from llmhub.core.downloader import HFDownloader
from llmhub.core.gguf_parser import GGUFParser
from llmhub.core.registry import ModelRegistry
from llmhub.core.runner import ModelRunner
from llmhub.utils.math_calc import compute_memory_breakdown, estimate_decode_throughput
from llmhub.utils.platform import get_hardware_profile

def cmd_pull(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    downloader = HFDownloader(config)
    try:
        downloader.download(args.model)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to pull model {args.model}: {e}\n")
        sys.exit(1)

def cmd_list(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    registry = ModelRegistry(config.db_path)
    records = registry.list_models()

    if not records:
        print("No models found in local storage.")
        print("Pull a model with: llmhub pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m")
        return

    print(f"\n{'TAG':<45} {'ARCH':<10} {'QUANT':<10} {'SIZE':<10} {'MAX CTX':<10}")
    print("-" * 90)
    for r in records:
        size_str = f"{r.size_bytes / (1024*1024):.1f} MB" if r.size_bytes < 1024*1024*1024 else f"{r.size_bytes / (1024*1024*1024):.2f} GB"
        print(f"{r.tag:<45} {r.architecture:<10} {r.quant:<10} {size_str:<10} {r.ctx_max:<10}")
    print()

def cmd_show(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    registry = ModelRegistry(config.db_path)
    record = registry.find_model(args.model)

    if not record:
        sys.stderr.write(f"[ERROR] Model '{args.model}' not found in registry. Run 'llmhub list' to see available models.\n")
        sys.exit(1)

    model_path = Path(record.file_path)
    if not model_path.exists():
        sys.stderr.write(f"[ERROR] Model binary file missing from disk: {model_path}\n")
        sys.exit(1)

    metadata = GGUFParser.parse_file(model_path)
    file_size = model_path.stat().st_size
    hw = get_hardware_profile()

    print("=" * 75)
    print(f" LLMHUB MODEL INSPECTOR: {record.tag}")
    print("=" * 75)
    print(f" File Path:          {record.file_path}")
    print(f" SHA-256 Digest:     {record.sha256}")
    print(f" Architecture:       {record.architecture}")
    print(f" Layers (Blocks):    {record.n_layers}")
    print(f" Attention Heads:    {metadata.head_count} (KV Heads: {record.n_kv_heads})")
    print(f" Head Dimension:     {record.head_dim}")
    print(f" Context Window:     {record.ctx_max} tokens")
    print(f" Weights Size:       {file_size / (1024*1024):.1f} MB ({file_size / (1024*1024*1024):.2f} GB)")
    print(f" Embedded Template:  {'Yes' if metadata.chat_template else 'No (using fallback)'}")

    print("\n--- Memory Footprint Breakdown ---")
    for ctx in [2048, 4096, 8192]:
        if ctx <= record.ctx_max:
            mem = compute_memory_breakdown(
                weights_bytes=file_size,
                n_layers=record.n_layers,
                n_kv_heads=record.n_kv_heads,
                head_dim=record.head_dim,
                n_ctx=ctx
            )
            print(f" Context {ctx:<5}: Weights: {mem.weights_mb:7.1f} MB | KV Cache: {mem.kv_cache_mb:6.1f} MB | Total RAM: {mem.total_mb:7.1f} MB")

    print("\n--- Theoretical Bandwidth Physics ---")
    # Estimate dual channel DDR4/DDR5
    bandwidth = estimate_decode_throughput(file_size, estimated_bandwidth_gb_s=40.0)
    print(f" Memory Bandwidth Est: {bandwidth.memory_bandwidth_gb_s:.1f} GB/s (Dual-Channel DDR4 standard)")
    print(f" Peak Theoretical:    {bandwidth.theoretical_max_tok_s:.1f} tokens/sec")
    print(f" Limiting Factor:     {bandwidth.bottleneck_type}")
    print(f" Analysis:            {bandwidth.explanation}")
    print("=" * 75 + "\n")

def cmd_run(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    registry = ModelRegistry(config.db_path)
    record = registry.find_model(args.model)

    if not record:
        # Check if argument is a direct GGUF file path
        candidate_path = Path(args.model).resolve()
        if candidate_path.is_file() and candidate_path.suffix.lower() == ".gguf":
            model_path = candidate_path
        else:
            sys.stderr.write(f"[ERROR] Model '{args.model}' not found in registry. Pull it first with:\n")
            sys.stderr.write(f"  llmhub pull {args.model}\n")
            sys.exit(1)
    else:
        model_path = Path(record.file_path)

    ctx_size = args.ctx_size or config.settings["runtime"]["default_context_size"]
    threads = args.threads or config.settings["runtime"]["default_threads"]

    runner = ModelRunner(
        model_path=model_path,
        n_ctx=ctx_size,
        n_threads=threads,
        use_mmap=config.settings["runtime"]["use_mmap"],
        use_mlock=config.settings["runtime"]["use_mlock"],
    )

    if args.prompt:
        # One-shot mode
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        formatted = runner.template_engine.render(messages, add_generation_prompt=True)

        last_chunk = {}
        for chunk in runner.stream_completion(
            prompt=formatted,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        ):
            if not chunk["done"]:
                sys.stdout.write(chunk["token"])
                sys.stdout.flush()
            last_chunk = chunk

        sys.stdout.write("\n\n")
        if last_chunk:
            sys.stdout.write(
                f"[{last_chunk.get('token_count', 0)} tokens | "
                f"TTFT: {last_chunk.get('ttft_ms', 0):.1f}ms | "
                f"Speed: {last_chunk.get('decode_tok_s', 0):.1f} tok/s | "
                f"Time: {last_chunk.get('total_elapsed_s', 0):.2f}s]\n"
            )
        runner.close()
    else:
        # Interactive REPL mode
        try:
            runner.chat_repl(
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
        finally:
            runner.close()

def cmd_rm(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    registry = ModelRegistry(config.db_path)
    record = registry.find_model(args.model)

    if not record:
        sys.stderr.write(f"[ERROR] Model '{args.model}' not found in registry.\n")
        sys.exit(1)

    model_path = Path(record.file_path)
    manifest_path = Path(record.manifest_path)
    model_dir = model_path.parent

    try:
        models_root = config.models_dir.resolve()
        resolved_dir = model_dir.resolve()
        # Strictly ensure we only rmtree dedicated model subdirectories inside ~/.llmhub/models/
        if resolved_dir.is_relative_to(models_root) and resolved_dir != models_root:
            shutil.rmtree(resolved_dir)
        else:
            if model_path.exists():
                model_path.unlink()
            if manifest_path.exists():
                manifest_path.unlink()
        registry.remove_model(record.tag)
        print(f"Removed model '{record.tag}' and cleaned files from {model_dir}")
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to delete {model_dir}: {e}\n")
        sys.exit(1)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmhub",
        description="LLMHub: Offline-First High-Performance GGUF Model Runtime Platform"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # pull
    p_pull = subparsers.add_parser("pull", help="Download a model from Hugging Face")
    p_pull.add_argument("model", help="Model specification (e.g. Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m)")
    p_pull.set_defaults(func=cmd_pull)

    # run
    p_run = subparsers.add_parser("run", help="Run model in interactive REPL or execute prompt")
    p_run.add_argument("model", help="Model name or spec (e.g. Qwen2.5-0.5B-Instruct-GGUF:q4_k_m)")
    p_run.add_argument("-p", "--prompt", default=None, help="One-shot prompt to execute")
    p_run.add_argument("-s", "--system", default="You are a helpful and concise offline AI assistant.", help="System prompt")
    p_run.add_argument("-c", "--ctx-size", type=int, default=None, help="Context size in tokens (defaults to config.toml)")
    p_run.add_argument("-t", "--threads", type=int, default=None, help="Number of CPU worker threads")
    p_run.add_argument("-m", "--max-tokens", type=int, default=512, help="Maximum generated tokens")
    p_run.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p_run.set_defaults(func=cmd_run)

    # list
    p_list = subparsers.add_parser("list", help="List all locally installed models")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = subparsers.add_parser("show", help="Show model architecture, metadata, and memory breakdown")
    p_show.add_argument("model", help="Model name or spec")
    p_show.set_defaults(func=cmd_show)

    # rm
    p_rm = subparsers.add_parser("rm", help="Remove model from local storage and registry")
    p_rm.add_argument("model", help="Model name or spec")
    p_rm.set_defaults(func=cmd_rm)

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
