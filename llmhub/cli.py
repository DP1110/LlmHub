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
import time
from pathlib import Path
from typing import Optional

from llmhub.config import LLMHubConfig
from llmhub.core.downloader import HFDownloader
from llmhub.core.gguf_parser import GGUFParser
from llmhub.core.registry import ModelRegistry
from llmhub.core.runner import ModelRunner
from llmhub.utils.math_calc import compute_memory_breakdown, estimate_decode_throughput
from llmhub.utils.platform import get_hardware_profile
from llmhub.hardware.detector import HardwareDetector
from llmhub.hardware.tuner import AutoTuner
from llmhub.server.client import LLMHubClient, ServerUnavailableError
from llmhub.server.server import run_server

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
    server_cfg = config.settings.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 11434)
    client = LLMHubClient(base_url=f"http://{host}:{port}")

    if client.is_server_running():
        models = client.list_models()
        if not models:
            print("No models found in local storage.")
            print("Pull a model with: llmhub pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m")
            return

        print(f"\n{'TAG':<45} {'ARCH':<10} {'QUANT':<10} {'SIZE':<10} {'MAX CTX':<10}")
        print("-" * 90)
        for r in models:
            size_b = r.get("size_bytes", 0)
            size_str = f"{size_b / (1024*1024):.1f} MB" if size_b < 1024**3 else f"{size_b / (1024**3):.2f} GB"
            print(f"{r.get('tag', ''):<45} {r.get('architecture', ''):<10} {r.get('quant', ''):<10} {size_str:<10} {r.get('ctx_max', ''):<10}")
        print()
    else:
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

def _run_via_http(client: LLMHubClient, args: argparse.Namespace) -> None:
    if args.prompt:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})

        last_usage = {}
        try:
            for chunk in client.stream_chat(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            ):
                token_text = chunk.get("token", "")
                if token_text:
                    sys.stdout.write(token_text)
                    sys.stdout.flush()
                if chunk.get("usage"):
                    last_usage = chunk["usage"]
        except KeyboardInterrupt:
            sys.stdout.write("\n[Interrupted]")

        sys.stdout.write("\n\n")
        if last_usage:
            sys.stdout.write(
                f"[{last_usage.get('completion_tokens', 0)} tokens | "
                f"TTFT: {last_usage.get('ttft_ms', 0):.1f}ms | "
                f"Speed: {last_usage.get('decode_tok_s', 0):.1f} tok/s]\n"
            )
    else:
        print("=" * 65)
        print(f" LLMHub Interactive Chat Client (Connected to HTTP Daemon)")
        print(f" Model: {args.model}")
        print(" Type '/bye', 'exit', or 'quit' to end session.")
        print("=" * 65 + "\n")

        history: list[dict[str, str]] = []
        if args.system:
            history.append({"role": "system", "content": args.system})

        while True:
            try:
                user_input = input("\n>>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting session.")
                break

            if not user_input:
                continue

            if user_input.lower() in ("/bye", "exit", "quit"):
                print("Session terminated.")
                break

            history.append({"role": "user", "content": user_input})
            sys.stdout.write("\n")
            sys.stdout.flush()

            bot_reply = ""
            last_usage = {}
            try:
                for chunk in client.stream_chat(
                    model=args.model,
                    messages=history,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature
                ):
                    token_text = chunk.get("token", "")
                    if token_text:
                        sys.stdout.write(token_text)
                        sys.stdout.flush()
                        bot_reply += token_text
                    if chunk.get("usage"):
                        last_usage = chunk["usage"]
            except KeyboardInterrupt:
                sys.stdout.write("\n[Interrupted]")

            sys.stdout.write("\n")
            if last_usage:
                sys.stdout.write(
                    f"[{last_usage.get('completion_tokens', 0)} tokens | "
                    f"TTFT: {last_usage.get('ttft_ms', 0):.1f}ms | "
                    f"Speed: {last_usage.get('decode_tok_s', 0):.1f} tok/s]\n"
                )
            history.append({"role": "assistant", "content": bot_reply})

def _run_standalone(config: LLMHubConfig, args: argparse.Namespace) -> None:
    registry = ModelRegistry(config.db_path)
    record = registry.find_model(args.model)

    if not record:
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
        try:
            runner.chat_repl(
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
        finally:
            runner.close()

def cmd_run(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    if getattr(args, "low_mem", False):
        config.set_low_memory_mode(True)
    server_cfg = config.settings.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 11434)
    client = LLMHubClient(base_url=f"http://{host}:{port}", api_key=getattr(args, "api_key", None))

    use_http = not getattr(args, "local", False) and client.is_server_running()

    if not getattr(args, "local", False) and not use_http:
        sys.stdout.write(f"[INFO] Server not detected at http://{host}:{port}. Running in standalone mode.\n")
        sys.stdout.write(f"       (Tip: run 'llmhub serve' to enable HTTP/OpenAI API daemon)\n\n")

    if use_http:
        _run_via_http(client, args)
    else:
        _run_standalone(config, args)

def cmd_serve(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    if getattr(args, "low_mem", False):
        config.set_low_memory_mode(True)
    server_cfg = config.settings.get("server", {})
    host = args.host or server_cfg.get("host", "127.0.0.1")
    port = args.port or server_cfg.get("port", 11434)
    run_server(host=host, port=port, config=config)

def cmd_ps(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    server_cfg = config.settings.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 11434)
    client = LLMHubClient(base_url=f"http://{host}:{port}")

    if not client.is_server_running():
        sys.stderr.write(f"[ERROR] LLMHub server is not running on http://{host}:{port}.\n")
        sys.stderr.write(f"Start the server using:\n  llmhub serve\n")
        sys.exit(1)

    try:
        ps_data = client.get_ps()
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to query /api/ps: {e}\n")
        sys.exit(1)

    models_list = ps_data.get("models", [])
    count = ps_data.get("resident_models_count", len(models_list))
    total_resident_mb = ps_data.get("total_resident_mb", 0.0)
    free_ram = ps_data.get("system_ram_free_mb", 0.0)
    safety_margin = ps_data.get("safety_margin", 0.85)

    print("=" * 95)
    print(" LLMHUB MULTI-MODEL RESIDENT MATRIX (/api/ps)")
    print("=" * 95)
    if not models_list:
        print(" No models currently resident in memory.")
        print(f" System Free RAM: {free_ram:.1f} MB | Safety Margin: {safety_margin * 100:.0f}%")
    else:
        print(f"{'MODEL':<36} {'PINNED':<8} {'ACTIVE':<8} {'IDLE':<10} {'RAM (TOTAL / W+KV)':<22} {'PEAK SPEED':<12}")
        print("-" * 95)
        for m in models_list:
            tag = m.get("tag", "")
            pinned_str = "YES" if m.get("pinned") else "NO"
            active_str = str(m.get("active_sessions", 0))
            idle_str = f"{m.get('idle_seconds', 0.0):.1f}s"
            mem = m.get("memory", {})
            tot = mem.get("total_mb", 0)
            w = mem.get("weights_mb", 0)
            kv = mem.get("kv_cache_mb", 0)
            ram_str = f"{tot:.1f}M ({w:.0f}+{kv:.0f})"
            bw = m.get("bandwidth", {})
            bw_str = f"{bw.get('theoretical_max_tok_s', 0):.1f} tok/s"
            print(f"{tag:<36} {pinned_str:<8} {active_str:<8} {idle_str:<10} {ram_str:<22} {bw_str:<12}")
        print("-" * 95)
        print(f" Total Resident: {count} models ({total_resident_mb:.1f} MB) | System Free RAM: {free_ram:.1f} MB | Safety Margin: {safety_margin * 100:.0f}%")
    print("=" * 95 + "\n")

def cmd_pin(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    server_cfg = config.settings.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 11434)
    client = LLMHubClient(base_url=f"http://{host}:{port}")

    if not client.is_server_running():
        sys.stderr.write(f"[ERROR] LLMHub server is not running on http://{host}:{port}.\n")
        sys.stderr.write(f"Start the server with 'llmhub serve' before pinning models.\n")
        sys.exit(1)

    try:
        client.pin_model(args.model)
        print(f"Pinned model '{args.model}' in memory (immune to LRU eviction).")
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to pin model: {e}\n")
        sys.exit(1)

def cmd_unpin(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    server_cfg = config.settings.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 11434)
    client = LLMHubClient(base_url=f"http://{host}:{port}")

    if not client.is_server_running():
        sys.stderr.write(f"[ERROR] LLMHub server is not running on http://{host}:{port}.\n")
        sys.stderr.write(f"Start the server with 'llmhub serve' before unpinning models.\n")
        sys.exit(1)

    try:
        client.unpin_model(args.model)
        print(f"Unpinned model '{args.model}'.")
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to unpin model: {e}\n")
        sys.exit(1)

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

    # If the daemon server is running, attempt to unload the model first so Windows mmap lock is released
    server_cfg = config.settings.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 11434)
    client = LLMHubClient(base_url=f"http://{host}:{port}")
    if client.is_server_running():
        try:
            client.unload_model(record.tag)
        except Exception:
            pass

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

def cmd_hardware(args: argparse.Namespace) -> None:
    prof = HardwareDetector.get_profile()
    cpu = prof.cpu
    gpu = prof.gpu

    print("=" * 75)
    print(" LLMHUB SYSTEM HARDWARE PROFILER")
    print("=" * 75)
    print(f" Operating System:      {prof.os_name} ({prof.arch})")
    print(f" Environment Mode:      {'Android Termux' if prof.is_termux else 'Desktop / Server'}")
    print(f" User Privileges:       {'Root / Administrator' if not prof.is_non_root else 'Non-Root (Restricted)'}")
    print("-" * 75)
    print(f" CPU Architecture:      {cpu.arch}")
    print(f" Physical CPU Cores:    {cpu.physical_cores}")
    print(f" Logical Threads (SMT): {cpu.logical_cores}")
    if cpu.is_hybrid_big_little:
        print(f" Core Architecture:     Hybrid Big.LITTLE ({cpu.performance_cores} Performance + {cpu.efficiency_cores} Efficiency)")
    else:
        print(f" Core Architecture:     Homogeneous Symmetric Multi-Processing")
    print(f" SIMD Vector Sets:      {', '.join(cpu.simd_extensions) if cpu.simd_extensions else 'None / Standard'}")
    print("-" * 75)
    print(f" Total System RAM:      {prof.total_ram_bytes / (1024**3):.2f} GB ({prof.total_ram_bytes / (1024**2):.0f} MB)")
    print(f" Available Free RAM:    {prof.free_ram_bytes / (1024**3):.2f} GB ({prof.free_ram_bytes / (1024**2):.0f} MB)")
    print("-" * 75)
    print(f" Accelerator / GPU:     {gpu.name} (Backend: {gpu.backend.upper()})")
    if gpu.is_available:
        print(f" VRAM Available:        {gpu.vram_bytes / (1024**2):.0f} MB ({gpu.device_count} devices)")
    print("-" * 75)
    print(f" Auto-Tuned Threads:    {prof.suggested_threads} worker threads")
    threads, rationale = AutoTuner.select_threads(prof)
    print(f" Threading Rationale:   {rationale}")

    if prof.warnings:
        print("-" * 75)
        print(" Platform Flags & Heuristics:")
        for w in prof.warnings:
            print(f"   [!] {w}")
    print("=" * 75 + "\n")

def cmd_plan(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    registry = ModelRegistry(config.db_path)
    record = registry.find_model(args.model)

    if not record:
        candidate_path = Path(args.model).resolve()
        if candidate_path.is_file() and candidate_path.suffix.lower() == ".gguf":
            metadata = GGUFParser.parse_file(candidate_path)
            weights_bytes = candidate_path.stat().st_size
            tag = candidate_path.stem
            layers = metadata.block_count
            kv_heads = metadata.head_count_kv
            head_dim = metadata.head_dim
            ctx_max = metadata.context_length
        else:
            sys.stderr.write(f"[ERROR] Model '{args.model}' not found in registry. Pull it first with:\n")
            sys.stderr.write(f"  llmhub pull {args.model}\n")
            sys.exit(1)
    else:
        model_path = Path(record.file_path)
        weights_bytes = record.size_bytes
        tag = record.tag
        layers = record.n_layers
        kv_heads = record.n_kv_heads
        head_dim = record.head_dim
        ctx_max = record.ctx_max

    prof = HardwareDetector.get_profile()
    plan = AutoTuner.generate_tuning_plan(
        model_tag=tag,
        weights_bytes=weights_bytes,
        n_layers=layers,
        n_kv_heads=kv_heads,
        head_dim=head_dim,
        ctx_max=ctx_max,
        profile=prof
    )

    print("=" * 75)
    print(f" LLMHUB HARDWARE AUTO-TUNING PLAN: {plan.model_tag}")
    print("=" * 75)
    print(f" Status:                {'SAFE TO EXECUTE' if plan.is_safe else 'MEMORY PRESSURE WARNING'}")
    print(f" Optimal Threads:       {plan.recommended_threads} ({plan.thread_rationale})")
    print(f" Recommended Context:   {plan.recommended_ctx} tokens")
    print(f" Max Safe Context:      {plan.max_safe_ctx} tokens (Model Max: {ctx_max})")
    print(f" Weights Footprint:     {plan.memory_breakdown_mb['weights_mb']:.1f} MB")
    print(f" KV Cache ({plan.recommended_ctx} ctx):   {plan.memory_breakdown_mb['kv_cache_mb']:.1f} MB")
    print(f" Total Allocation:      {plan.memory_breakdown_mb['total_mb']:.1f} MB")
    print(f" Free RAM After Load:   {plan.free_ram_after_load_mb:.1f} MB")
    
    if plan.warnings:
        print("-" * 75)
        print(" Notices & Warnings:")
        for w in plan.warnings:
            print(f"   [!] {w}")

    # Estimate parameter scale from weights size (Q4 ~0.55 GB per 1B params)
    est_params_b = weights_bytes / (0.55 * 1024 * 1024 * 1024)
    print("\n--- Quantization Suggestions for ~" + f"{est_params_b:.1f}B Scale ---")
    suggestions = AutoTuner.evaluate_quant_suggestions(est_params_b, prof)
    print(f"{'QUANT':<8} {'BITS/W':<8} {'WEIGHTS':<10} {'TOTAL RAM':<12} {'COMPATIBILITY':<30}")
    print("-" * 75)
    for s in suggestions:
        print(f"{s.quant:<8} {s.bits_per_weight:<8.1f} {s.weights_size_gb:<10.2f} GB {s.total_ram_needed_gb:<10.2f} GB {s.status:<30}")
    print("=" * 75 + "\n")

def cmd_termux(args: argparse.Namespace) -> None:
    from llmhub.termux.controller import TermuxController
    config = LLMHubConfig()
    controller = TermuxController(config=config)

    if getattr(args, "wake_lock", False):
        success = controller.wakelock.acquire()
        print(f"Acquire wake-lock: {'SUCCESS' if success else 'FAILED (Unsupported or error)'}")
    elif getattr(args, "release_lock", False):
        success = controller.wakelock.release()
        print(f"Release wake-lock: {'SUCCESS' if success else 'FAILED'}")

    status = controller.get_status()
    print("=" * 75)
    print(" LLMHUB TERMUX & MOBILE PLATFORM DIAGNOSTICS")
    print("=" * 75)
    print(f" Termux Environment: {'YES (Android ARM64)' if status['is_termux'] else 'NO (Standard OS)'}")
    print(f" Non-Root Sandbox:   {'YES (Unprivileged)' if status['is_non_root'] else 'NO (Root/Admin)'}")
    print(f" Low-Memory Mode:    {'ENABLED' if status['low_memory_mode'] else 'DISABLED'}")
    print("-" * 75)
    print(" CPU WAKELOCK")
    wl = status['wakelock']
    print(f"   Supported:        {wl['is_supported']}")
    print(f"   Currently Held:   {wl['is_held']}")
    print("-" * 75)
    print(" THERMAL MONITOR")
    th = status['thermal']
    print(f"   Thermal State:    {th['state'].upper()}")
    print(f"   Max Temperature:  {th['max_temp_c']}°C (Avg: {th['avg_temp_c']}°C)")
    print(f"   Sensor Source:    {th['source']}")
    print(f"   Effective Threads:{th['effective_threads']} (Base: {th['base_threads']})")
    if th['zones']:
        print("   Detected Zones:")
        for z in th['zones'][:8]:
            print(f"     - {z['id']} ({z['type']}): {z['temp_c']}°C")
    print("-" * 75)
    print(" MEMORY PRESSURE & ANDROID LMK")
    mp = status['memory_pressure']
    print(f"   Pressure Level:   {mp['level'].upper()}")
    print(f"   Available RAM:    {mp['available_ram_mb']:.1f} MB ({mp['available_percent']:.1f}%)")
    print(f"   Total RAM:        {mp['total_ram_mb']:.1f} MB")
    print(f"   LMK oom_score_adj:{mp['oom_score_adj'] if mp['oom_score_adj'] is not None else 'N/A'}")
    if mp.get('psi_full_avg10') is not None:
        print(f"   PSI Stall (full): {mp['psi_full_avg10']:.2f}%")
    print("=" * 75 + "\n")

def cmd_status(args: argparse.Namespace) -> None:
    from llmhub.core.lifecycle import PIDManager, is_process_alive
    config = LLMHubConfig()
    pid_mgr = PIDManager(config.run_dir)
    pid_info = pid_mgr.read_pid_info()

    server_cfg = config.settings.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 11434)
    client = LLMHubClient(base_url=f"http://{host}:{port}", api_key=getattr(args, "api_key", None))

    print("=" * 75)
    print(" LLMHUB DAEMON & OBSERVABILITY STATUS")
    print("=" * 75)

    is_running = client.is_server_running()
    if pid_info:
        pid = pid_info.get("pid")
        alive = is_process_alive(pid) if pid else False
        state_str = "RUNNING (ACTIVE)" if alive and is_running else ("STALE LOCK (DEAD)" if not alive else "STARTING")
        print(f" Daemon State:       {state_str}")
        print(f" Process ID (PID):   {pid} (Alive: {alive})")
        print(f" Configured Address: http://{pid_info.get('host')}:{pid_info.get('port')}")
        print(f" Started At:         {pid_info.get('started_at')}")
        print(f" Runtime Version:    {pid_info.get('version')}")
    else:
        print(f" Daemon State:       STOPPED (No active PID file at {pid_mgr.pid_file})")

    print("-" * 75)
    if is_running:
        try:
            import httpx
            with httpx.Client(timeout=3.0) as http_c:
                hz = http_c.get(f"http://{host}:{port}/healthz").json()
                print(f" HTTP Health:        {hz.get('status', 'unknown').upper()}")
                print(f" Uptime:             {hz.get('uptime_seconds', 0.0):.1f}s")
                print(f" Resident Models:    {hz.get('resident_models_count', 0)} loaded")
                if hz.get('resident_models'):
                    for m in hz['resident_models']:
                        print(f"   - {m}")
                termux_info = hz.get("termux", {})
                if termux_info:
                    print(f" Thermal State:      {termux_info.get('thermal_state')}")
                    print(f" Memory Pressure:    {termux_info.get('memory_pressure')}")
                    print(f" Low-Memory Mode:    {termux_info.get('low_memory_mode')}")
        except Exception as e:
            print(f" [!] Error fetching healthz: {e}")
    else:
        print(" HTTP Health:        OFFLINE")
    print("=" * 75 + "\n")

def cmd_cluster(args: argparse.Namespace) -> None:
    config = LLMHubConfig()
    server_cfg = config.settings.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 11434)
    client = LLMHubClient(base_url=f"http://{host}:{port}", api_key=getattr(args, "api_key", None))

    action = getattr(args, "cluster_action", "status") or "status"

    if action == "join":
        peer = getattr(args, "peer", None)
        if not peer:
            print("[!] Error: Specify peer address to join, e.g. 'llmhub cluster join 192.168.1.15:11434'")
            return
        try:
            res = client.join_cluster(peer)
            print(f"[+] Cluster Join: {res.get('message', 'Success')}")
        except Exception as e:
            print(f"[!] Failed to join cluster via {peer}: {e}")
        return

    elif action == "ping":
        target = getattr(args, "target", None)
        if not target:
            print("[!] Error: Specify target node address to ping, e.g. 'llmhub cluster ping 192.168.1.15:11434'")
            return
        if not target.startswith("http://") and not target.startswith("https://"):
            target = f"http://{target}"
        try:
            import httpx
            start_t = time.perf_counter()
            with httpx.Client(timeout=4.0) as http_c:
                resp = http_c.get(f"{target.rstrip('/')}/healthz")
            rtt_ms = (time.perf_counter() - start_t) * 1000.0
            if resp.status_code == 200:
                hz = resp.json()
                print(f"[+] Ping {target}: SUCCESS ({rtt_ms:.1f} ms)")
                print(f"    Status:   {hz.get('status')}")
                print(f"    Uptime:   {hz.get('uptime_seconds', 0.0):.1f}s")
                print(f"    Resident: {len(hz.get('resident_models', []))} models")
            else:
                print(f"[!] Ping {target}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[!] Ping {target} FAILED: {e}")
        return

    elif action == "plan":
        model = getattr(args, "model", None)
        if not model:
            print("[!] Error: Specify model tag for pipeline plan, e.g. 'llmhub cluster plan llama3:70b'")
            return
        try:
            plan = client.get_cluster_pipeline_plan(
                model=model,
                total_layers=getattr(args, "layers", 32),
                hidden_dim=getattr(args, "hidden_dim", 4096),
            )
            print("=" * 80)
            print(f" LLMHUB DISTRIBUTED PIPELINE PARALLEL PLAN: {model}")
            print("=" * 80)
            print(f" Feasible:              {'YES' if plan.get('is_feasible') else 'NO'}")
            print(f" Total Layers:          {plan.get('total_layers')}")
            print(f" Hidden Dimension:      {plan.get('hidden_dim')}")
            print(f" Transfer Bytes/Step:   {plan.get('activation_bytes_per_token', 0) / 1024.0:.1f} KB")
            print(f" Est. Network Latency:  {plan.get('estimated_transfer_ms_per_step', 0.0):.2f} ms / step")
            print(f" Analysis:              {plan.get('feasibility_notes')}")
            print("-" * 80)
            stages = plan.get("stages", [])
            if stages:
                print(f" {'Stage':<6} | {'Node':<18} | {'Layers':<12} | {'Est RAM':<10} | {'Roles'}")
                print("-" * 80)
                for s in stages:
                    roles = []
                    if s.get("has_embeddings"): roles.append("Embeddings")
                    if s.get("has_lm_head"): roles.append("LM Head")
                    roles_str = ", ".join(roles) if roles else "Hidden Layers"
                    lyr_str = f"{s.get('layer_start')}-{s.get('layer_end')-1} ({s.get('layer_count')})"
                    print(f" {s.get('stage_id'):<6} | {s.get('node_name')[:18]:<18} | {lyr_str:<12} | {s.get('estimated_memory_mb', 0):.0f} MB{'':<4} | {roles_str}")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"[!] Failed to compute cluster pipeline plan: {e}")
        return

    # Default action: status
    try:
        st = client.get_cluster_status()
    except Exception as e:
        print(f"[!] Error querying cluster status: {e}")
        print("    Ensure LLMHub daemon is running with cluster enabled in config.toml.")
        return

    print("=" * 85)
    print(" LLMHUB DISTRIBUTED CLUSTER TOPOLOGY & CAPACITIES")
    print("=" * 85)
    print(f" Cluster Enabled:    {'YES' if st.get('cluster_enabled') else 'NO'}")
    print(f" Local Node ID:      {st.get('local_node_id')}")
    print(f" Nodes (Healthy/Tot):{st.get('healthy_nodes_count')} / {st.get('total_nodes')}")
    print(f" Total Cluster RAM:  {st.get('total_cluster_ram_mb', 0.0):.1f} MB (Free: {st.get('free_cluster_ram_mb', 0.0):.1f} MB)")
    print("-" * 85)

    nodes = st.get("nodes", [])
    if nodes:
        print(f" {'Node ID':<16} | {'Address':<22} | {'Status':<11} | {'Free RAM':<10} | {'Act':<4} | {'Resident Models'}")
        print("-" * 85)
        for n in nodes:
            nid = n.get("node_id", "")[:16]
            addr = f"{n.get('host')}:{n.get('port')}"
            status = n.get("status", "").upper()
            free_ram = f"{n.get('free_ram_mb', 0.0):.0f} MB"
            acts = n.get("active_sessions", 0)
            res_models = list(n.get("resident_models", {}).keys())
            res_str = ", ".join(res_models) if res_models else "(none)"
            print(f" {nid:<16} | {addr:<22} | {status:<11} | {free_ram:<10} | {acts:<4} | {res_str}")
    else:
        print(" (No cluster nodes detected)")

    res_cluster = st.get("resident_models", {})
    if res_cluster:
        print("-" * 85)
        print(" Cluster Distributed Models:")
        for tag, m_info in res_cluster.items():
            nodes_str = ", ".join(m_info.get("resident_nodes", []))
            print(f"   * {tag:<25} -> Hosted on: {nodes_str} (Active sessions: {m_info.get('total_active_sessions', 0)})")
    print("=" * 85 + "\n")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmhub",
        description="LLMHub: Offline-First High-Performance GGUF Model Runtime Platform"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # cluster
    p_cluster = subparsers.add_parser("cluster", help="Manage distributed cluster nodes, status, and pipeline sharding")
    cl_subparsers = p_cluster.add_subparsers(dest="cluster_action", required=False)

    p_cl_status = cl_subparsers.add_parser("status", help="Show cluster nodes, health, and memory capacities")
    p_cl_status.add_argument("--api-key", default=None, help="Optional API key for authenticated endpoints")

    p_cl_join = cl_subparsers.add_parser("join", help="Join an existing cluster peer node")
    p_cl_join.add_argument("peer", help="Peer node host:port (e.g. 192.168.1.15:11434)")
    p_cl_join.add_argument("--api-key", default=None, help="Optional API key")

    p_cl_ping = cl_subparsers.add_parser("ping", help="Ping a cluster peer node and measure RTT latency")
    p_cl_ping.add_argument("target", help="Peer host:port or URL")

    p_cl_plan = cl_subparsers.add_parser("plan", help="Compute pipeline layer partition plan across active cluster nodes")
    p_cl_plan.add_argument("model", help="Model specification (e.g. Llama-3-70B)")
    p_cl_plan.add_argument("--layers", type=int, default=32, help="Total transformer layers")
    p_cl_plan.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension size")
    p_cl_plan.add_argument("--api-key", default=None, help="Optional API key")
    p_cluster.set_defaults(func=cmd_cluster)

    # status
    p_status = subparsers.add_parser("status", help="Inspect server daemon process, health, and observability metrics")
    p_status.add_argument("--api-key", default=None, help="Optional API key for authenticated endpoints")
    p_status.set_defaults(func=cmd_status)

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
    p_run.add_argument("--low-mem", action="store_true", help="Enable low-memory mode with context downshifting")
    p_run.add_argument("--local", action="store_true", help="Force local direct execution without HTTP daemon")
    p_run.add_argument("--api-key", default=None, help="API key for HTTP server authentication")
    p_run.set_defaults(func=cmd_run)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the LLMHub HTTP API server daemon")
    p_serve.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=None, help="Bind port (default: 11434)")
    p_serve.add_argument("--low-mem", action="store_true", help="Start server in low-memory mode")
    p_serve.add_argument("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
    p_serve.set_defaults(func=cmd_serve)

    # termux
    p_termux = subparsers.add_parser("termux", help="Inspect Termux/mobile environment, thermals, and memory pressure")
    p_termux.add_argument("--wake-lock", action="store_true", help="Acquire Android CPU wake-lock")
    p_termux.add_argument("--release-lock", action="store_true", help="Release Android CPU wake-lock")
    p_termux.set_defaults(func=cmd_termux)

    # ps
    p_ps = subparsers.add_parser("ps", help="Show active running models, sessions, and memory (/api/ps)")
    p_ps.set_defaults(func=cmd_ps)

    # pin
    p_pin = subparsers.add_parser("pin", help="Pin a model in RAM against LRU eviction")
    p_pin.add_argument("model", help="Model name or spec")
    p_pin.set_defaults(func=cmd_pin)

    # unpin
    p_unpin = subparsers.add_parser("unpin", help="Unpin a model from RAM")
    p_unpin.add_argument("model", help="Model name or spec")
    p_unpin.set_defaults(func=cmd_unpin)

    # list
    p_list = subparsers.add_parser("list", help="List all locally installed models")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = subparsers.add_parser("show", help="Show model architecture, metadata, and memory breakdown")
    p_show.add_argument("model", help="Model name or spec")
    p_show.set_defaults(func=cmd_show)

    # hardware
    p_hw = subparsers.add_parser("hardware", help="Inspect CPU topology, SIMD extensions, RAM, and GPU")
    p_hw.set_defaults(func=cmd_hardware)

    # plan
    p_plan = subparsers.add_parser("plan", help="Auto-tune threads, context, and evaluate quantization for a model")
    p_plan.add_argument("model", help="Model name or spec")
    p_plan.set_defaults(func=cmd_plan)

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
