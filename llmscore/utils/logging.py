"""Logging helpers for the llmscore CLI."""

from __future__ import annotations

import logging


def get_logger(name: str = "llmscore") -> logging.Logger:
    """Return a logger configured with a sensible default formatter."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


__all__ = ["get_logger"]
