"""
Centralized Logging Configuration for Universal Consciousness Interface

This module provides standardized logging configuration across all consciousness modules.
It supports multiple log levels, structured logging, and consciousness-specific formatting.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

# Define log levels specific to consciousness processing
CONSCIOUSNESS_DEBUG = 5  # Ultra-verbose consciousness state logging


class ConsciousnessFormatter(logging.Formatter):
    """Custom formatter for consciousness-related log messages with emoji indicators"""

    # Color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    # Emoji prefixes for consciousness-related events
    EMOJI_MAP = {
        'DEBUG': '🔍',
        'INFO': '✨',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }

    def __init__(self, use_color: bool = True, use_emoji: bool = True):
        """
        Initialize the consciousness formatter.

        Args:
            use_color: Whether to use terminal colors
            use_emoji: Whether to use emoji prefixes
        """
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.use_color = use_color and sys.stderr.isatty()
        self.use_emoji = use_emoji

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and emojis"""
        # Add emoji prefix
        if self.use_emoji and record.levelname in self.EMOJI_MAP:
            emoji = self.EMOJI_MAP[record.levelname]
            record.msg = f"{emoji} {record.msg}"

        # Format the record
        formatted = super().format(record)

        # Add colors
        if self.use_color and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            formatted = f"{color}{formatted}{reset}"

        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    use_color: bool = True,
    use_emoji: bool = True,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Set up centralized logging for the Universal Consciousness Interface.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name (will be created in log_dir)
        console_output: Whether to output logs to console
        use_color: Whether to use colored output in console
        use_emoji: Whether to use emoji prefixes
        log_dir: Directory for log files (defaults to ./logs)

    Returns:
        Configured root logger
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_formatter = ConsciousnessFormatter(use_color=use_color, use_emoji=use_emoji)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        if log_dir is None:
            log_dir = "logs"
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create log file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_path / f"{log_file}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        # File logs don't use colors but keep emojis
        file_formatter = ConsciousnessFormatter(use_color=False, use_emoji=True)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_file_path}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance configured for the module
    """
    return logging.getLogger(name)


def configure_module_logging(
    module_name: str,
    log_level: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for a specific consciousness module.

    Args:
        module_name: Name of the module
        log_level: Optional specific log level for this module

    Returns:
        Configured logger for the module
    """
    logger = logging.getLogger(module_name)

    if log_level:
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

    return logger


# Environment-based configuration
def setup_from_environment() -> logging.Logger:
    """
    Set up logging based on environment variables.

    Environment variables:
        UCI_LOG_LEVEL: Log level (default: INFO)
        UCI_LOG_FILE: Log file name (default: None)
        UCI_LOG_DIR: Log directory (default: ./logs)
        UCI_LOG_CONSOLE: Enable console output (default: true)
        UCI_LOG_COLOR: Enable colored output (default: true)
        UCI_LOG_EMOJI: Enable emoji prefixes (default: true)

    Returns:
        Configured root logger
    """
    log_level = os.getenv('UCI_LOG_LEVEL', 'INFO')
    log_file = os.getenv('UCI_LOG_FILE', None)
    log_dir = os.getenv('UCI_LOG_DIR', None)
    console_output = os.getenv('UCI_LOG_CONSOLE', 'true').lower() == 'true'
    use_color = os.getenv('UCI_LOG_COLOR', 'true').lower() == 'true'
    use_emoji = os.getenv('UCI_LOG_EMOJI', 'true').lower() == 'true'

    return setup_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=console_output,
        use_color=use_color,
        use_emoji=use_emoji,
        log_dir=log_dir
    )


# Default setup for quick initialization
def quick_setup(level: str = "INFO") -> logging.Logger:
    """
    Quick logging setup for development and testing.

    Args:
        level: Log level (default: INFO)

    Returns:
        Configured root logger
    """
    return setup_logging(
        log_level=level,
        console_output=True,
        use_color=True,
        use_emoji=True
    )


# Initialize default logging when module is imported
# This can be overridden by calling setup_logging() explicitly
if not logging.getLogger().handlers:
    quick_setup()


__all__ = [
    'setup_logging',
    'get_logger',
    'configure_module_logging',
    'setup_from_environment',
    'quick_setup',
    'ConsciousnessFormatter',
    'CONSCIOUSNESS_DEBUG'
]
