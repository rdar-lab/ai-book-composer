"""Logging configuration for AI Book Composer."""

import logging
import logging.handlers
from pathlib import Path

from .config import Settings


def setup_logging(settings: Settings) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        settings: The project settings

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(getattr(logging, settings.logging.level))

    if settings.logging.file:
        log_file = Path(settings.logging.file)
        log_directory = log_file.parent
        log_directory.mkdir(parents=True, exist_ok=True)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, settings.logging.level))
        file_formatter = logging.Formatter(settings.logging.format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Console handler (if enabled)
    if settings.logging.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, settings.logging.level))
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    logging.getLogger(__name__).info("Logging is configured.")

    return logger
