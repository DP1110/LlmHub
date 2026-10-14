"""Structured JSON Logging & Request Tracing for LLMHub.

Features:
- ContextVar-based request ID (trace_id) propagation across async tasks
- JSON formatted logs with standard ISO-8601 timestamps and context metadata
- Automatic RotatingFileHandler in ~/.llmhub/logs/llmhub.log
- Thread-safe and asyncio-friendly
"""

import contextvars
import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Context variable holding current request trace identifier
current_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_request_id", default=None
)


class RequestContextFilter(logging.Filter):
    """Logging filter that injects trace_id from contextvars into LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = current_request_id.get() or "-"
        return True


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production logging and log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
        }

        # Include exception details if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if passed via extra={}
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "trace_id"
        }
        extras = {
            k: v for k, v in record.__dict__.items() if k not in standard_attrs
        }
        if extras:
            log_entry["extra"] = extras

        return json.dumps(log_entry, default=str)


def setup_logging(
    log_level: str = "INFO",
    logs_dir: Optional[Path] = None,
    json_format: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure root and LLMHub application loggers with console and rotating file handlers."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger("llmhub")
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on re-initialization
    root_logger.handlers.clear()
    root_logger.propagate = False

    ctx_filter = RequestContextFilter()

    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.addFilter(ctx_filter)

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        plain_fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(trace_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(plain_fmt)

    root_logger.addHandler(console_handler)

    # 2. Rotating File Handler
    if logs_dir is None:
        from llmhub.config import get_base_dir
        logs_dir = get_base_dir() / "logs"

    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "llmhub.log"
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.addFilter(ctx_filter)
        # File handler always logs structured JSON for reliable log shipping
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Don't fail entire application if file handler cannot be created (e.g. read-only fs)
        logging.warning("Failed to configure rotating file handler at %s: %s", logs_dir, e)


def get_logger(name: str) -> logging.Logger:
    """Return a logger child within the llmhub namespace with trace_id propagation."""
    if not name.startswith("llmhub"):
        name = f"llmhub.{name}"
    logger = logging.getLogger(name)
    return logger
