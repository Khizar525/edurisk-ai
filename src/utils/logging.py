"""
Structured logging setup for EduRisk AI.
"""

import logging
import sys
from pathlib import Path

from src.config import PATHS


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure structured logging for the application.

    Returns:
        Configured logger instance.
    """
    # Ensure logs directory exists
    PATHS.logs.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("edurisk")
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(PATHS.logs / "edurisk.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))
    logger.addHandler(file_handler)

    return logger
