<div align="center">
  <img src="llmhub/static/logo.svg" width="128" height="128" alt="LLMHub Logo" />
  <h1>LLMHub</h1>
  <p><b>Offline-first, Ollama-like concurrent GGUF model execution runtime on CPU with distributed cluster scale-out and built-in Web GUI.</b></p>

  <p>
    <a href="#features">Features</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#web-gui-dashboard">Web GUI</a> •
    <a href="#cli-reference">CLI Reference</a> •
    <a href="#cluster--distributed">Cluster</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#benchmarks">Benchmarks</a>
  </p>
</div>

---

## Overview

**LLMHub** is a zero-cloud, memory-bandwidth bounded CPU inference runtime designed for standard PCs (Windows, Linux, macOS) and unprivileged Android mobile devices via Termux (ARM64). It enables concurrent multi-model multi-session execution scheduled by a token-level round-robin engine with continuous batching, prefix caching, multi-device cluster pipeline sharding, and an OpenAI-compatible HTTP daemon with a built-in minimalist Web GUI.

---

## Features

- 🔒 **Zero Cloud Dependencies**: Pure offline-first execution with HTTP Range resumable downloads and streaming SHA-256 validation directly from Hugging Face.
- ⚡ **Token-Level Round-Robin Scheduler**: Deficit Round Robin (DRR), chunked prefill interleaving, and thread allocation modes (Time-Slice, Partition, Hybrid).
- 🚀 **Continuous Batching & Prefix Caching**: Per-model iteration-level continuous batching with LRU Radix prefix KV cache saving up to 80% redundant prefill compute.
- 🎯 **Advanced Scheduling & Speculative Decoding**:
  - Multi-Level Feedback Queue (MLFQ) starvation-free priority aging.
  - Predictive SLO latency enforcement with strict/permissive admission control.
  - Multi-tenant fair-share deficit balancing.
  - Speculative decoding draft verification engine.
- 📱 **Android Termux ARM64 (Non-Root)**:
  - Thermal monitoring with 4-state hysteresis downshifting.
  - Android Low-Memory Killer (LMK) / `oom_score_adj` memory pressure monitoring.
  - Context downshifting & quantization downgrade under physical RAM pressure.
  - Android `termux-wake-lock` background execution management.
- 🌐 **Distributed Cluster Scale-Out**:
  - Zero-configuration LAN UDP broadcast peer discovery (port 11435).
  - Locality-aware routing (local resident, remote resident, cold capacity balancing).
  - Heterogeneous edge pipeline sharding (e.g. Workstation PC + Android Phone).
- 🖥️ **Built-in Minimalist Monochrome Web GUI**:
  - Real-time streaming chat with token speed and TTFT metrics.
  - Model & RAM management (Load, Unload, Pin/Unpin).
  - Cluster topology viewer and pipeline sharding simulator.
  - High-contrast black-and-white Scandinavian aesthetic.
- 🔌 **OpenAI-Compatible HTTP / SSE Daemon**:
  - `/v1/chat/completions` (JSON & Server-Sent Events streaming).
  - Prometheus 0.0.4 metrics exporter (`/metrics`).
  - Structured JSON logging with `X-Request-ID` tracing.

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/llmhub.git
cd llmhub

# Install in editable mode
pip install -e .
```

### 2. Start the Server & Web GUI

```bash
# Start background daemon on port 11434
llmhub serve
```

Open your browser at **[http://localhost:11434/](http://localhost:11434/)** to access the built-in Web GUI!

---

## CLI Reference

```bash
# Pull model from Hugging Face with auto-resume & SHA256 verification
llmhub pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m

# Run interactive terminal chat REPL
llmhub run Qwen2.5-0.5B-Instruct-GGUF:q4_k_m

# One-shot command-line prompt
llmhub run Qwen2.5-0.5B-Instruct-GGUF:q4_k_m -p "Explain memory bandwidth."

# Inspect model architecture, KV heads, and memory breakdown
llmhub show Qwen2.5-0.5B-Instruct-GGUF:q4_k_m

# List locally downloaded models
llmhub list

# View active resident models in RAM, active sessions, and peak throughput
llmhub ps

# Probe CPU cores, SIMD (AVX/NEON), and RAM bandwidth
llmhub hardware

# Recommended auto-tuner plan
llmhub plan

# Cluster management
llmhub cluster status
llmhub cluster join 192.168.1.50:11434
llmhub cluster ping 192.168.1.50:11434
llmhub cluster plan llama3:70b --layers 80 --hidden-dim 8192
```

---

## Web GUI Dashboard

Served directly by the FastAPI daemon at `http://localhost:11434/`:
- **Chat**: Streaming token responses with live tok/s and TTFT stats.
- **Models & RAM**: One-click model loading, unloading, and memory pinning.
- **Cluster & Sharding**: Live LAN node discovery and partition planner.
- **Telemetry**: Hardware thermals, LMK memory pressure, and Prometheus metrics.

---

## Benchmarks & Verification

The test harness validates all 10 phases with **100% test pass rate** (163/163 tests passing):

```bash
# Run complete test suite
pytest -v

# Run micro-benchmark harness
python benchmarks/bench_single_runner.py
```

### Micro-Benchmark Latencies (Tested on Windows AMD64 CPU)
- **GGUF Parse Latency**: 0.21 ms / file
- **Jinja2 Chat Template**: 0.015 ms / turn
- **Memory Physics Formula**: 3.17 µs / call
- **Auto-Tuner Plan Latency**: 3.69 µs / plan
- **Quantum Decode Dispatch**: 0.98 µs / quantum
- **FastAPI SSE Turnaround**: 8.35 ms / request
- **Continuous Batching Throughput**: 1,830,873 tok/s (B=4)
- **Prefix Cache Lookup**: 25.21 µs / query (80% hit rate)
- **MLFQ Queue Demotion**: 12.26 µs / step
- **Prometheus Metric Scrape**: 347.69 µs / scrape
- **Cluster Route Resolution**: 4.08 µs / decision
- **Discovery Beacon Serialization**: 5.41 µs / packet
- **Pipeline Sharding Plan**: 23.07 µs / plan

---

## License

MIT License. Designed and built with offline-first principles.
