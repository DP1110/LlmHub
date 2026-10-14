"""Hardware detection, auto-tuning, and model manifest management."""

from llmhub.hardware.detector import (
    HardwareDetector,
    HardwareProfile,
    CPUTopology,
    GPUDeviceInfo,
)
from llmhub.hardware.tuner import (
    AutoTuner,
    ModelTuningPlan,
    QuantSuggestion,
)
from llmhub.hardware.manifest import (
    ModelManifest,
    HardwareRequirements,
)

__all__ = [
    "HardwareDetector",
    "HardwareProfile",
    "CPUTopology",
    "GPUDeviceInfo",
    "AutoTuner",
    "ModelTuningPlan",
    "QuantSuggestion",
    "ModelManifest",
    "HardwareRequirements",
]
