"""Prometheus Metrics Exporter for LLMHub.

Provides zero-external-dependency metrics collection and text exposition
conforming to the Prometheus exposition format specification (version 0.0.4).
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def _escape_label_value(val: Any) -> str:
    s = str(val)
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_help(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("\n", "\\n")


def _format_labels(labels: Dict[str, str]) -> str:
    """Format label dictionary into Prometheus label syntax '{k="v",...}' with RFC 0.0.4 escaping."""
    if not labels:
        return ""
    items = [f'{k}="{_escape_label_value(v)}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(items) + "}"


class Metric:
    def __init__(self, name: str, doc: str, metric_type: str) -> None:
        self.name = name
        self.doc = doc
        self.metric_type = metric_type
        self._lock = threading.Lock()

    def generate_exposition(self) -> List[str]:
        raise NotImplementedError


class Counter(Metric):
    """Monotonically increasing cumulative counter."""

    def __init__(self, name: str, doc: str) -> None:
        super().__init__(name, doc, "counter")
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        if amount < 0:
            raise ValueError("Counter increments must be non-negative")
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def generate_exposition(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {_escape_help(self.doc)}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0.0")
            else:
                for label_tuples, val in sorted(self._values.items()):
                    lbl_str = _format_labels(dict(label_tuples))
                    lines.append(f"{self.name}{lbl_str} {val}")
        return lines


class Gauge(Metric):
    """Instantaneous value that can increase, decrease, or be set."""

    def __init__(self, name: str, doc: str) -> None:
        super().__init__(name, doc, "gauge")
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def generate_exposition(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {_escape_help(self.doc)}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0.0")
            else:
                for label_tuples, val in sorted(self._values.items()):
                    lbl_str = _format_labels(dict(label_tuples))
                    lines.append(f"{self.name}{lbl_str} {val}")
        return lines


class Histogram(Metric):
    """Cumulative histogram tracking value distributions over predefined buckets."""

    DEFAULT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(self, name: str, doc: str, buckets: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(name, doc, "histogram")
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[Tuple[Tuple[str, str], ...], Dict[float, int]] = {}
        self._sums: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._totals: Dict[Tuple[Tuple[str, str], ...], int] = {}

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        val = float(value)
        with self._lock:
            if key not in self._counts:
                self._counts[key] = {b: 0 for b in self.buckets}
                self._sums[key] = 0.0
                self._totals[key] = 0

            self._sums[key] += val
            self._totals[key] += 1
            for b in self.buckets:
                if val <= b:
                    self._counts[key][b] += 1

    def generate_exposition(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {_escape_help(self.doc)}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            if not self._totals:
                # Default empty histogram representation
                for b in self.buckets:
                    lines.append(f'{self.name}_bucket{{le="{b}"}} 0')
                lines.append(f'{self.name}_bucket{{le="+Inf"}} 0')
                lines.append(f"{self.name}_sum 0.0")
                lines.append(f"{self.name}_count 0")
            else:
                for key in sorted(self._totals.keys()):
                    labels_dict = dict(key)
                    cum_count = 0
                    for b in self.buckets:
                        cum_count = self._counts[key][b]
                        b_labels = dict(labels_dict)
                        b_labels["le"] = str(b)
                        lines.append(f"{self.name}_bucket{_format_labels(b_labels)} {cum_count}")
                    
                    # +Inf bucket always matches total count
                    inf_labels = dict(labels_dict)
                    inf_labels["le"] = "+Inf"
                    lines.append(f"{self.name}_bucket{_format_labels(inf_labels)} {self._totals[key]}")

                    lbl_str = _format_labels(labels_dict)
                    lines.append(f"{self.name}_sum{lbl_str} {self._sums[key]}")
                    lines.append(f"{self.name}_count{lbl_str} {self._totals[key]}")
        return lines


class MetricsRegistry:
    """Registry coordinating metric objects and scrape-time collectors."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}
        self._collectors: List[Callable[[], None]] = []
        self._lock = threading.Lock()

        # Initialize standard LLMHub core metrics
        self.requests_total = self.register_counter(
            "llmhub_requests_total",
            "Total number of completed inference requests categorized by model and status",
        )
        self.tokens_generated_total = self.register_counter(
            "llmhub_tokens_generated_total",
            "Total number of generated output decode tokens",
        )
        self.prompt_tokens_total = self.register_counter(
            "llmhub_prompt_tokens_total",
            "Total number of input prompt tokens processed",
        )
        self.prefix_cache_hits_total = self.register_counter(
            "llmhub_prefix_cache_hits_total",
            "Total number of prompt prefix cache hits",
        )
        self.prefix_cache_misses_total = self.register_counter(
            "llmhub_prefix_cache_misses_total",
            "Total number of prompt prefix cache misses",
        )
        self.rate_limit_exceeded_total = self.register_counter(
            "llmhub_rate_limit_exceeded_total",
            "Total number of requests rejected by rate limiting",
        )

        # Gauges
        self.active_sessions = self.register_gauge(
            "llmhub_active_sessions",
            "Current number of active sessions in execution or waiting",
        )
        self.queued_sessions = self.register_gauge(
            "llmhub_queued_sessions",
            "Current number of queued sessions in scheduler",
        )
        self.loaded_models_count = self.register_gauge(
            "llmhub_loaded_models_count",
            "Number of GGUF models currently loaded in RAM",
        )
        self.memory_allocated_bytes = self.register_gauge(
            "llmhub_memory_allocated_bytes",
            "Total estimated memory footprint of loaded models (weights + KV cache)",
        )
        self.system_memory_used_ratio = self.register_gauge(
            "llmhub_system_memory_used_ratio",
            "System physical memory usage fraction (0.0 to 1.0)",
        )
        self.thermal_temperature_celsius = self.register_gauge(
            "llmhub_thermal_temperature_celsius",
            "SoC / CPU thermal temperature in degrees Celsius",
        )
        self.memory_pressure_level = self.register_gauge(
            "llmhub_memory_pressure_level",
            "Current memory pressure level (0=nominal, 1=moderate, 2=critical)",
        )
        self.wakelock_held = self.register_gauge(
            "llmhub_wakelock_held",
            "Whether the Android system wake-lock is actively held (0 or 1)",
        )
        self.speculative_acceptance_ratio = self.register_gauge(
            "llmhub_speculative_acceptance_ratio",
            "Empirical acceptance ratio of speculative decoding tokens",
        )

        # Histograms
        self.ttft_seconds = self.register_histogram(
            "llmhub_ttft_seconds",
            "Time To First Token latency in seconds",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        self.request_duration_seconds = self.register_histogram(
            "llmhub_request_duration_seconds",
            "Total end-to-end request duration in seconds",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )

    def register_counter(self, name: str, doc: str) -> Counter:
        with self._lock:
            metric = Counter(name, doc)
            self._metrics[name] = metric
            return metric

    def register_gauge(self, name: str, doc: str) -> Gauge:
        with self._lock:
            metric = Gauge(name, doc)
            self._metrics[name] = metric
            return metric

    def register_histogram(
        self, name: str, doc: str, buckets: Optional[Tuple[float, ...]] = None
    ) -> Histogram:
        with self._lock:
            metric = Histogram(name, doc, buckets)
            self._metrics[name] = metric
            return metric

    def register_collector(self, collector: Callable[[], None]) -> None:
        with self._lock:
            self._collectors.append(collector)

    def export(self) -> str:
        """Run all registered dynamic collectors and serialize metrics to Prometheus text exposition format."""
        with self._lock:
            collectors = list(self._collectors)

        for col in collectors:
            try:
                col()
            except Exception:
                pass  # Dynamic collectors must never crash scrape requests

        lines: List[str] = []
        with self._lock:
            metrics_snapshot = list(self._metrics.values())

        for m in metrics_snapshot:
            lines.extend(m.generate_exposition())

        # Prometheus text format requires a trailing newline
        return "\n".join(lines) + "\n"


_GLOBAL_REGISTRY: Optional[MetricsRegistry] = None
_INIT_LOCK = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """Return the global MetricsRegistry singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _INIT_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = MetricsRegistry()
    return _GLOBAL_REGISTRY
