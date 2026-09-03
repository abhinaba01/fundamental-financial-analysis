"""
Centralized logging configuration for the financial fundamentals analysis pipeline.
All modules should import and use the logger from this module instead of print().
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured logger instance with stream handler.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # On Windows, sys.stdout is often bound to a non-UTF-8 console codepage
    # (e.g. cp1252). A log message containing a non-ASCII character would
    # otherwise raise UnicodeEncodeError from inside the logging module.
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except AttributeError:
        pass

    # Stream handler (console output)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
