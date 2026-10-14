# LLMHub

Offline-first, Ollama-like concurrent GGUF inference platform designed for standard PCs (Windows, Linux, macOS) and Android devices via non-root Termux (ARM64).

## Architecture Highlights
- **Zero Cloud Reliance**: Offline-first design with direct Hugging Face GGUF downloads, HTTP Range resume, and streaming SHA-256 validation.
- **Physical Memory Physics**: Real-time weights and KV cache computation ($2 \times L \times H_{kv} \times D_{head} \times C \times B_{kv}$) with memory-bandwidth bound decode ceilings ($\le \text{Bandwidth} / \text{Weights}$).
- **Instant GGUF Metadata Parser**: Native binary v2/v3 header parsing in $<0.3\text{ ms}$ without allocating model weights into RAM.
- **Jinja2 Chat Templates**: Seamless formatting with embedded templates (`tokenizer.chat_template`) and ChatML/Llama-3 fallbacks.
- **Hardware-Aware CPU Tuning**: Physical core detection, avoiding SMT/hyperthreading cache contention, with full Termux non-root compatibility.

## Quickstart

### Installation
```bash
pip install -e .
```

### CLI Commands
```bash
# Pull model from Hugging Face
llmhub pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m

# Inspect architecture, layer counts, KV heads, and memory breakdown
llmhub show Qwen2.5-0.5B-Instruct-GGUF:q4_k_m

# List locally cached models
llmhub list

# Run interactive multi-turn chat REPL
llmhub run Qwen2.5-0.5B-Instruct-GGUF:q4_k_m

# One-shot command-line prompt
llmhub run Qwen2.5-0.5B-Instruct-GGUF:q4_k_m -p "What is memory bandwidth?"

# Remove model
llmhub rm Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
```

## Running Tests & Benchmarks
```bash
pytest -v
python benchmarks/bench_single_runner.py
```
