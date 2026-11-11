"""Private standard-library logging helpers for :mod:`SKSurrogate`.

The package does not configure global logging state. :func:`emit` writes to the
calling module's logger and, for users who never configured logging, still
writes the message to ``sys.stdout`` so today's console output is preserved
byte-for-byte. Once any logger in the propagation chain has a handler, stdout
output stops, which keeps a configured ``logging`` setup from producing double
lines.

This module is private: it is not exported from :mod:`SKSurrogate` and adds no
third-party dependency.
"""

import logging
import sys

_DEFAULT_LOGGER_NAME = "SKSurrogate"


def get_logger(name):
    """Return the standard-library logger registered under ``name``."""
    return logging.getLogger(name)


def _caller_logger(depth=2):
    """Return the logger of the module calling :func:`emit`.

    ``depth=1`` is :func:`emit` itself, ``depth=2`` its caller.
    """
    module_name = sys._getframe(depth).f_globals.get("__name__", _DEFAULT_LOGGER_NAME)
    return logging.getLogger(module_name)


def _has_handler(logger):
    """Return True when some logger in the propagate chain has a handler.

    The chain is walked upward while ``propagate`` is true; reaching a logger
    without a handler and without propagation stops the search at that logger.
    """
    current = logger
    while current is not None:
        if current.handlers:
            return True
        if not current.propagate:
            return False
        current = current.parent
    return False


def emit(level, msg):
    """Log ``msg`` at ``level`` and fall back to stdout when logging is unconfigured.

    The record always goes to the calling module's logger. The plain-text
    fallback fires only when no logger in that logger's propagation chain has a
    handler, i.e. when the user has attached no logging configuration; this
    reproduces the pre-logging ``print()`` output exactly.

    Records at :data:`logging.WARNING` and above are **not** duplicated: with no
    handler, the standard library's ``logging.lastResort`` already writes them to
    stderr, so the stdout fallback is skipped for them.
    """
    logger = _caller_logger(2)
    enabled = logger.isEnabledFor(level)
    logger.log(level, msg)
    if _has_handler(logger):
        return
    if enabled and level >= logging.WARNING and logging.lastResort is not None:
        return
    sys.stdout.write(str(msg) + "\n")
