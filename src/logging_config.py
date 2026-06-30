"""Logging helpers for the daily report pipeline."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(*, log_level: str = "INFO", log_dir: str | Path = "logs") -> logging.Logger:
    """Configure console and file logging for pipeline runs."""
    logger = logging.getLogger("crypto_trade_plot")
    if logger.handlers:
        logger.setLevel(log_level.upper())
        return logger

    logger.setLevel(log_level.upper())
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / "run_daily.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
