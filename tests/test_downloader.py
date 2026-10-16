import pytest
from llmhub.core.downloader import HFDownloader, DownloadError

def test_parse_spec_valid():
    org, repo, quant, filename = HFDownloader.parse_spec("Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m")
    assert org == "Qwen"
    assert repo == "Qwen2.5-0.5B-Instruct-GGUF"
    assert quant == "q4_k_m"
    assert filename is None

def test_parse_spec_with_filename():
    org, repo, quant, filename = HFDownloader.parse_spec("bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    assert org == "bartowski"
    assert repo == "Llama-3.2-1B-Instruct-GGUF"
    assert filename == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    assert quant == "q4_k_m"

def test_parse_spec_invalid():
    with pytest.raises(DownloadError) as exc_info:
        HFDownloader.parse_spec("invalid_spec_without_slash")
    assert "recognized alias" in str(exc_info.value)

def test_parse_spec_smollm2_aliases():
    org, repo, quant, filename = HFDownloader.parse_spec("smollm2:135m")
    assert org == "bartowski"
    assert repo == "SmolLM2-135M-Instruct-GGUF"
    assert quant == "q4_k_m"
    assert filename is None

    org, repo, quant, filename = HFDownloader.parse_spec("smollm2:360m")
    assert org == "bartowski"
    assert repo == "SmolLM2-360M-Instruct-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, filename = HFDownloader.parse_spec("smollm2:1.7b")
    assert org == "bartowski"
    assert repo == "SmolLM2-1.7B-Instruct-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, filename = HFDownloader.parse_spec("smollm2")
    assert org == "bartowski"
    assert repo == "SmolLM2-135M-Instruct-GGUF"
    assert quant == "q4_k_m"

def test_parse_spec_alias_custom_quant():
    org, repo, quant, filename = HFDownloader.parse_spec("smollm2:135m:q8_0")
    assert org == "bartowski"
    assert repo == "SmolLM2-135M-Instruct-GGUF"
    assert quant == "q8_0"

    org, repo, quant, filename = HFDownloader.parse_spec("qwen2.5:0.5b:q8_0")
    assert org == "Qwen"
    assert repo == "Qwen2.5-0.5B-Instruct-GGUF"
    assert quant == "q8_0"

def test_parse_spec_popular_aliases():
    org, repo, quant, _ = HFDownloader.parse_spec("qwen2.5:0.5b")
    assert org == "Qwen"
    assert repo == "Qwen2.5-0.5B-Instruct-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, _ = HFDownloader.parse_spec("llama3.2:1b")
    assert org == "bartowski"
    assert repo == "Llama-3.2-1B-Instruct-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, _ = HFDownloader.parse_spec("deepseek-r1:1.5b")
    assert org == "unsloth"
    assert repo == "DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, _ = HFDownloader.parse_spec("phi3.5")
    assert org == "bartowski"
    assert repo == "Phi-3.5-mini-instruct-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, _ = HFDownloader.parse_spec("tinyllama")
    assert org == "TheBloke"
    assert repo == "TinyLlama-1.1B-Chat-v1.0-GGUF"
    assert quant == "q4_k_m"

def test_parse_spec_case_insensitivity():
    org, repo, quant, _ = HFDownloader.parse_spec("SmolLM2:135M")
    assert org == "bartowski"
    assert repo == "SmolLM2-135M-Instruct-GGUF"
    assert quant == "q4_k_m"

def test_parse_spec_latest_and_trailing_colons():
    org, repo, quant, _ = HFDownloader.parse_spec("smollm2:latest")
    assert org == "bartowski"
    assert repo == "SmolLM2-135M-Instruct-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, _ = HFDownloader.parse_spec("qwen:latest")
    assert org == "Qwen"
    assert repo == "Qwen2.5-0.5B-Instruct-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, _ = HFDownloader.parse_spec("smollm2:135m:")
    assert org == "bartowski"
    assert repo == "SmolLM2-135M-Instruct-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, _ = HFDownloader.parse_spec("qwen")
    assert org == "Qwen"
    assert repo == "Qwen2.5-0.5B-Instruct-GGUF"
    assert quant == "q4_k_m"

    org, repo, quant, _ = HFDownloader.parse_spec("deepseek-r1")
    assert org == "unsloth"
    assert repo == "DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
    assert quant == "q4_k_m"


