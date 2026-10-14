"""Core modules for LLMHub."""

from llmhub.core.gguf_parser import GGUFParser, GGUFMetadata
from llmhub.core.registry import ModelRegistry, ModelRecord
from llmhub.core.downloader import HFDownloader
from llmhub.core.template import ChatTemplateEngine

def __getattr__(name: str):
    if name == "ModelRunner":
        from llmhub.core.runner import ModelRunner
        globals()["ModelRunner"] = ModelRunner
        return ModelRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals().keys()) | set(__all__))

__all__ = [
    "GGUFParser",
    "GGUFMetadata",
    "ModelRegistry",
    "ModelRecord",
    "HFDownloader",
    "ChatTemplateEngine",
    "ModelRunner",
]
