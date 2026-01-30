"""Logging configuration for saflow.

This module provides centralized logging configuration for all saflow scripts.
Supports both console and file output with configurable log levels.

Usage:
    from code.utils.logging_config import setup_logging

    logger = setup_logging(
        name=__name__,
        log_file="preprocessing.log",
        level="INFO"
    )
    logger.info("Processing started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from code.utils.config import get_config


class LogFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def __init__(self, fmt: str, use_color: bool = True):
        """Initialize formatter.

        Args:
            fmt: Log message format string
            use_color: Whether to use ANSI color codes
        """
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional color.

        Args:
            record: Log record to format

        Returns:
            Formatted log message string
        """
        if self.use_color and record.levelname in self.COLORS:
            # Add color to level name
            levelname_color = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
            record.levelname = levelname_color

        return super().format(record)


def setup_logging(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: Optional[str] = None,
    config: Optional[dict] = None,
    console: bool = True,
    use_color: bool = True,
) -> logging.Logger:
    """Setup logging for a script or module.

    Creates a logger with both console and file handlers (if log_file specified).
    Log level and format are configurable via config.yaml or function arguments.

    Args:
        name: Logger name (typically __name__ of calling module)
        log_file: Optional log file path (relative to logs directory from config)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               If None, uses level from config.yaml
        config: Configuration dictionary. If None, loads default config.
        console: Whether to add console handler
        use_color: Whether to use color in console output

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(__name__, log_file="preprocessing.log")
        >>> logger.info("Starting preprocessing")
    """
    # Load config if not provided
    if config is None:
        config = get_config()

    # Get log level from config or argument
    if level is None:
        level = config.get("logging", {}).get("level", "INFO")
    log_level = getattr(logging, level.upper())

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Get log format from config
    log_format = config.get("logging", {}).get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if console and config.get("logging", {}).get("to_console", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = LogFormatter(log_format, use_color=use_color)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file and config.get("logging", {}).get("to_file", True):
        # Resolve log file path
        logs_dir = Path(config["paths"]["logs"])

        # Handle both string and Path inputs
        log_file_path = Path(log_file)

        # If log_file is just a filename, put it in logs directory
        # If it's a relative path, resolve relative to logs directory
        if not log_file_path.is_absolute():
            log_file_path = logs_dir / log_file_path

        # Create parent directories
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp to log file if it doesn't exist
        if not log_file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = log_file_path.stem
            suffix = log_file_path.suffix
            log_file_path = log_file_path.parent / f"{stem}_{timestamp}{suffix}"

        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(log_level)
        # Don't use color in file output
        file_formatter = LogFormatter(log_format, use_color=False)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name.

    This is a convenience function for retrieving loggers that have
    already been configured with setup_logging().

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
    """
    return logging.getLogger(name)


def add_file_handler(
    logger: logging.Logger,
    log_file: Union[str, Path],
    level: Optional[str] = None,
    config: Optional[dict] = None,
) -> None:
    """Add a file handler to an existing logger.

    Useful for adding multiple log files to the same logger.

    Args:
        logger: Logger instance to add handler to
        log_file: Log file path
        level: Optional log level (uses logger's level if not specified)
        config: Configuration dictionary

    Example:
        >>> logger = setup_logging(__name__)
        >>> add_file_handler(logger, "detailed.log", level="DEBUG")
    """
    if config is None:
        config = get_config()

    # Resolve log file path
    logs_dir = Path(config["paths"]["logs"])
    log_file_path = Path(log_file)

    if not log_file_path.is_absolute():
        log_file_path = logs_dir / log_file_path

    # Create parent directories
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp if file doesn't exist
    if not log_file_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = log_file_path.stem
        suffix = log_file_path.suffix
        log_file_path = log_file_path.parent / f"{stem}_{timestamp}{suffix}"

    # Create handler
    file_handler = logging.FileHandler(log_file_path, mode='a')

    # Set level
    if level:
        file_handler.setLevel(getattr(logging, level.upper()))
    else:
        file_handler.setLevel(logger.level)

    # Set formatter
    log_format = config.get("logging", {}).get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter = LogFormatter(log_format, use_color=False)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.info(f"Added file handler: {log_file_path}")


def log_provenance(
    logger: logging.Logger,
    script_name: str,
    git_hash: Optional[str] = None,
    config: Optional[dict] = None,
) -> None:
    """Log provenance information (git hash, config, environment).

    Args:
        logger: Logger instance
        script_name: Name of script being executed
        git_hash: Git commit hash (if available)
        config: Configuration dictionary

    Example:
        >>> logger = setup_logging(__name__)
        >>> log_provenance(logger, "run_preprocessing.py", git_hash="abc123")
    """
    logger.info("="*80)
    logger.info(f"Starting: {script_name}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    if git_hash:
        logger.info(f"Git commit: {git_hash}")

    if config:
        logger.info(f"Data root: {config['paths']['data_root']}")

    logger.info("="*80)
