"""
Logging configuration for pyERGM.

This module provides a centralized logging setup for the pyERGM package.
Users can configure logging verbosity by setting the log level.
"""

import logging
import sys

# Create a logger for the pyERGM package
logger = logging.getLogger("pyERGM")

# Default handler - outputs to stdout
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(_handler)

# Default level is INFO
logger.setLevel(logging.INFO)


def set_log_level(level: str | int) -> None:
    """
    Set the logging level for pyERGM.

    Parameters
    ----------
    level : str or int
        The logging level. Can be a string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        or a logging constant (logging.DEBUG, logging.INFO, etc.).

    Examples
    --------
    >>> from pyERGM.logging_config import set_log_level
    >>> set_log_level('DEBUG')  # Show all messages
    >>> set_log_level('WARNING')  # Show only warnings and errors
    >>> set_log_level('ERROR')  # Show only errors
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)


def disable_logging() -> None:
    """
    Disable all pyERGM logging output.

    Examples
    --------
    >>> from pyERGM.logging_config import disable_logging
    >>> disable_logging()  # Silence all pyERGM log output
    """
    logger.setLevel(logging.CRITICAL + 1)


def enable_logging(level: str | int = logging.INFO) -> None:
    """
    Enable pyERGM logging output.

    Parameters
    ----------
    level : str or int
        The logging level to set. Defaults to INFO.

    Examples
    --------
    >>> from pyERGM.logging_config import enable_logging
    >>> enable_logging()  # Re-enable logging at INFO level
    >>> enable_logging('DEBUG')  # Re-enable at DEBUG level
    """
    set_log_level(level)
