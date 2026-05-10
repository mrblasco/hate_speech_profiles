"""Logging configuration for the generation pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    # Silence noisy third-party loggers
    for name in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("pipeline")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"pipeline.{name}")
