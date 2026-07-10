"""
Structured logging setup for EduRisk AI.

Provides a consistent logging interface across all modules.
Logs to both console and file with structured formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import PATHS


_logger: Optional[logging.Logger] = None


def get_logger(name: str = "edurisk") -> logging.Logger:
    """
    Get or create a named logger.

    Args:
        name: Logger name (usually module name).

    Returns:
        Configured logger instance.
    """
    global _logger

    if _logger is not None and name == "edurisk":
        return _logger

    logger = logging.getLogger(name)

    if _logger is None and name == "edurisk":
        _logger = logger

    return logger


def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Configure structured logging for the application.

    Sets up:
    - Console handler (INFO level, human-readable format)
    - File handler (DEBUG level, detailed format)
    - Root edurisk logger

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files. Defaults to PATHS.logs.

    Returns:
        Configured root logger.
    """
    global _logger

    log_path = log_dir or PATHS.logs
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger
    logger = logging.getLogger("edurisk")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler — clean, readable output
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    # File handler — detailed, persistent log
    file_handler = logging.FileHandler(log_path / "edurisk.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    ))
    logger.addHandler(file_handler)

    _logger = logger
    return logger


def log_training_start(model_name: str, n_samples: int, n_features: int) -> None:
    """Log training start with context."""
    logger = get_logger()
    logger.info(f"Training started: {model_name} | Samples: {n_samples} | Features: {n_features}")


def log_training_complete(model_name: str, metrics: dict) -> None:
    """Log training completion with metrics."""
    logger = get_logger()
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    logger.info(f"Training complete: {model_name} | {metrics_str}")


def log_prediction(risk_level: str, confidence: float) -> None:
    """Log a prediction event."""
    logger = get_logger()
    logger.info(f"Prediction: {risk_level} | Confidence: {confidence:.1f}%")


def log_error(module: str, error: Exception) -> None:
    """Log an error with context."""
    logger = get_logger()
    logger.error(f"Error in {module}: {type(error).__name__}: {error}", exc_info=True)
