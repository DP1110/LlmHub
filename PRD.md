# LLMHub: Offline-First High-Performance GGUF Model Runtime Platform
## System Requirements & Product Requirements Document (PRD)

### Executive Summary
LLMHub is an offline-first, Ollama-compatible platform designed to run GGUF quantized models (from 512M up to 70B+ parameters) across heterogeneous CPU architectures (x86_64 and ARM64), including standard PC/Server environments (Linux, Windows, macOS) and resource-constrained Android devices running non-root Termux. 

The core architecture separates frontend orchestration, memory admission control, a preemptive token-level round-robin scheduler, continuous slot-based batching, and low-level hardware-optimized C execution contexts.

---

## 1. Storage Architecture & Directory Layout

LLMHub strictly adheres to the following isolated directory hierarchy rooted at `~/.llmhub` (overridable via `LLMHUB_HOME`):

```text
~/.llmhub/
├── config.toml                              # Runtime configuration (memory ceilings, timeouts, ports)
├── registry.db                              # SQLite 3 ACID registry of models, manifests, sessions
├── models/
│   └── <org>/
│       └── <name>/
│           └── <quant>/
│               ├── model.gguf               # Raw GGUF binary weights
│               └── manifest.json            # Architecture, quant, hashes, and profile metadata
├── logs/
│   ├── llmhub.log                           # Rotating system operational logs
│   └── audit.log                            # Request and token generation metrics
└── run/
    ├── llmhub.pid                           # Daemon process ID
    └── llmhub.sock                          # UNIX domain socket (Linux/macOS/Termux) or Named Pipe (Windows)
```

---

## 2. Multi-Phase Roadmap & Milestones

| Phase | Milestone Name | Key Deliverables | Status |
|:-----:|:---------------|:-----------------|:------:|
| **1** | **Single-Model Runner** | CLI (`pull`, `run`, `list`, `show`, `rm`), resumable HF downloader, streaming SHA-256, binary GGUF header parser, Jinja2 chat template rendering, stdout token streaming. | **In Progress** |
| **2** | **Registry & Hardware Detection** | Physical core detection (excluding SMT/Hyperthreading), RAM and free page detection, AVX2/AVX512/NEON/AMX SIMD probing, GGUF manifest extraction, thread allocation heuristics. | Planned |
| **3** | **Backend Interface & Server** | Async backend ABC (`load`, `unload`, `new_session`, `decode_step`, `prefill_chunk`), `LlamaCppBackend`, `RemoteBackend`, OpenAI-compatible `/v1/chat/completions` (SSE streaming), `/v1/models`, `/api/ps`. | Planned |
| **4** | **Multi-Model Manager** | Multiple concurrent resident models, memory admission control formula (`weights + kv_cache + overhead <= free_ram * safety_margin`), LRU eviction with keep-alive timers, model pinning, `mmap` page cache sharing. | Planned |
| **5** | **Token-Level Round-Robin Scheduler** | Runnable session queue across models, quantum $Q$ tokens per turn, chunked prefill $P$ tokens interleaved with decode, time-slice vs core-partition vs hybrid threading modes, Jain's fairness index harness. | Planned |
| **6** | **Continuous Batching per Model** | llama.cpp slot-based parallel batch API with shared weights context and isolated per-slot KV caches, prefix caching, scheduler batch selection. | Planned |
| **7** | **Advanced Scheduling & Speculative Decoding** | Multilevel feedback queue (MLFQ), latency SLOs (TTFT < 2s), speculative decoding with 512M draft models accelerating 7B/70B targets. | Planned |
| **8** | **Termux/ARM64 & Low-Memory Constraints** | Non-root compliance, `termux-wake-lock`, thermal throttling mitigation, dynamic context downshifting, Android Low Memory Killer (LMK) protection, zero-dependency installer. | Planned |
| **9** | **Observability & Hardening** | Prometheus `/metrics`, structured JSON logging, crash recovery, graceful teardown, token rate-limiting, input sanitization. | Planned |
| **10** | **Distributed Scale-Out (Architecture Only)** | Thin-client topology, mDNS/LAN discovery, layer-split pipeline parallelism across edge devices, evolution path toward vLLM/SGLang. | Planned |

---

## 3. Mathematical Foundations: Memory Physics & Bandwidth Limits

### 3.1 Memory Physics: Weights and KV Cache

For any transformer model with:
- $L$ = Number of layers (`llama.block_count`)
- $H_{kv}$ = Number of Key-Value attention heads (`llama.attention.head_count_kv`)
- $D_{head}$ = Head dimension ($d_{\text{model}} / H_{q}$)
- $C$ = Context window length ($n_{\text{ctx}}$ tokens)
- $B_{kv}$ = Bytes per KV cache element (typically 2 for FP16, 1 for FP8 / Q8_0)
- $S_{weights}$ = Total size of quantized model weights in bytes

$$\text{Memory}_{\text{total}} = S_{\text{weights}} + \text{Memory}_{\text{KV}}(C) + \text{Overhead}_{\text{compute}}$$

$$\text{Memory}_{\text{KV}}(C) = 2 \times L \times H_{kv} \times D_{head} \times C \times B_{kv}$$

#### Example 1: Qwen2.5-0.5B (Q4_K_M)
- $L = 24$, $H_{kv} = 2$, $D_{head} = 64$, $B_{kv} = 2$ (FP16 KV)
- $S_{\text{weights}} \approx 392 \text{ MB}$
- At $C = 2048$:
  $$\text{Memory}_{\text{KV}} = 2 \times 24 \times 2 \times 64 \times 2048 \times 2 = 25,165,824 \text{ bytes} \approx 24.0 \text{ MB}$$
- Total RAM required: $\approx 392 \text{ MB} + 24 \text{ MB} + 50 \text{ MB} \approx 466 \text{ MB}$.

#### Example 2: Llama-3.1-8B (Q4_K_M)
- $L = 32$, $H_{kv} = 8$, $D_{head} = 128$, $B_{kv} = 2$ (FP16 KV)
- $S_{\text{weights}} \approx 4.92 \text{ GB}$
- At $C = 8192$:
  $$\text{Memory}_{\text{KV}} = 2 \times 32 \times 8 \times 128 \times 8192 \times 2 = 1,073,741,824 \text{ bytes} = 1.00 \text{ GB}$$
- Total RAM required: $\approx 4.92 \text{ GB} + 1.00 \text{ GB} + 250 \text{ MB} \approx 6.17 \text{ GB}$.

### 3.2 Token Generation Limits: Memory Bandwidth Bottleneck

Autoregressive token generation (decoding with batch size $N=1$) has an arithmetic intensity of $\approx 1 \text{ FLOP} / \text{Byte}$. Every newly generated token requires loading **all active model weights from RAM into CPU L1/L2/L3 cache**.

$$\text{Maximum Theoretical Decode Speed (tokens/sec)} \le \frac{\text{Sustained Memory Bandwidth (GB/s)}}{S_{\text{weights}} \text{ (GB)}}$$

| System / Channel Configuration | Typical Memory Bandwidth | 0.5B Model (0.4 GB) | 7B/8B Model (4.9 GB) | 70B Model (Q4: 40 GB) |
|:---|:---:|:---:|:---:|:---:|
| **Single-Channel DDR4-2400** | ~17 GB/s | ~42.5 tok/s | ~3.4 tok/s | ~0.42 tok/s |
| **Dual-Channel DDR4-3200** (Current Host) | ~40 GB/s | ~100 tok/s | ~8.1 tok/s | ~1.0 tok/s |
| **Dual-Channel DDR5-5600** | ~75 GB/s | ~187 tok/s | ~15.3 tok/s | ~1.87 tok/s |
| **Apple M-Series Pro (150-200 GB/s)** | ~180 GB/s | ~450 tok/s | ~36.7 tok/s | ~4.5 tok/s |
| **Apple M-Series Max/Ultra (400-800 GB/s)**| ~600 GB/s | ~1500 tok/s | ~122 tok/s | ~15.0 tok/s |
| **Smartphone LPDDR5 (Snapdragon 8 Gen 2/3)**| ~50 GB/s | ~125 tok/s | ~10.2 tok/s | OOM (exceeds physical RAM) |

**Key Takeaway**: CPU decode performance on 7B+ models is **strictly memory-bandwidth bound**, not CPU clock or core count bound. Adding more CPU threads beyond memory channel saturation yields diminishing returns or degrades throughput due to cache contention and thread synchronization overhead.

---

## 4. Cross-Platform & Non-Root Android Termux Constraints

1. **Filesystem Hierarchy**: Non-root Android apps reside in `/data/data/com.termux/files/home`. Hardcoded `/home/...` or `/tmp/...` paths will fail. LLMHub dynamically queries `Path.home()` or `os.environ.get("HOME")`.
2. **Virtual Memory & Page Locking**: `mlock()` and `RLIMIT_MEMLOCK` require `CAP_SYS_RESOURCE` (root). LLMHub defaults to standard `mmap` with fallback to plain buffer reads if `mmap` flags are rejected by Android bionic libc.
3. **Process Priority & Core Affinity**: Setting thread affinity on Linux/Android (`sched_setaffinity`) may fail with `EPERM` when non-root processes try to pin to reserved cores. LLMHub wraps affinity calls in safe error-handling blocks.
4. **Android Low Memory Killer (LMK)**: The Android kernel aggressively terminates background processes with high RSS. LLMHub must track RSS and provide graceful proactive context truncation or model unloading before LMK issues SIGKILL.
