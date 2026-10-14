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
    with pytest.raises(DownloadError):
        HFDownloader.parse_spec("invalid_spec_without_slash")
