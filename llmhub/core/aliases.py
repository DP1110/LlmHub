"""Central model alias registry and resolver for LLMHub.

Maps user-friendly shorthand model tags (e.g. 'smollm2:135m', 'qwen2.5:0.5b')
to canonical Hugging Face GGUF repository paths.
"""

from typing import Dict

MODEL_ALIASES: Dict[str, str] = {
    # SmolLM2
    "smollm2:135m": "bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M",
    "smollm2-135m": "bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M",
    "smollm2:360m": "bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M",
    "smollm2-360m": "bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M",
    "smollm2:1.7b": "bartowski/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M",
    "smollm2-1.7b": "bartowski/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M",
    "smollm2": "bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M",

    # Qwen 2.5 & standard Qwen shorthand
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m",
    "qwen:0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m",
    "qwen:1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m",
    "qwen:3b": "Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m",
    "qwen:7b": "Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m",
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m",
    "qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m",

    # Llama 3.2 & 3.1
    "llama3.2:1b": "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
    "llama3.2-1b": "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
    "llama3.2:3b": "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
    "llama3.2-3b": "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
    "llama3.2": "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
    "llama3.1:8b": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",
    "llama3.1-8b": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",

    # DeepSeek R1 Distill
    "deepseek-r1": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M",
    "deepseek-r1:1.5b": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M",
    "deepseek-r1-1.5b": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M",
    "deepseek-r1:7b": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
    "deepseek-r1-7b": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",

    # Phi-3.5
    "phi3.5": "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",
    "phi3.5:mini": "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",
    "phi3.5-mini": "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",

    # TinyLlama
    "tinyllama": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:q4_k_m",
    "tinyllama:1.1b": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:q4_k_m",
    "tinyllama-1.1b": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:q4_k_m",
}


def resolve_model_alias(spec: str) -> str:
    """Resolve shorthand model aliases like 'smollm2:135m' or 'qwen2.5:0.5b'
    to full Hugging Face repository spec strings.

    If the spec matches an alias directly, returns the mapped target.
    If the spec contains a custom quantization suffix (e.g. 'smollm2:135m:q8_0'),
    resolves the base alias and attaches the requested quant.
    If the suffix is ':latest' or empty (e.g. trailing colon), resolves to the default quant.
    If no alias matches, returns the spec unchanged.
    """
    cleaned = spec.strip()
    lower = cleaned.lower()

    if lower in MODEL_ALIASES:
        return MODEL_ALIASES[lower]

    if lower.endswith(":latest"):
        trimmed = lower[:-7]
        if trimmed in MODEL_ALIASES:
            return MODEL_ALIASES[trimmed]

    if ":" in cleaned:
        prefix, suffix = (p.strip() for p in cleaned.rsplit(":", 1))
        if prefix.lower() in MODEL_ALIASES:
            base_target = MODEL_ALIASES[prefix.lower()]
            if not suffix or suffix.lower() == "latest":
                return base_target
            base_repo = base_target.split(":", 1)[0]
            return f"{base_repo}:{suffix}"

    return cleaned
