"""Logging configuration using structlog."""

import logging
import sys
from typing import Any
from contextlib import contextmanager

import structlog


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output logs as JSON. If False, use console format.
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper()),
    )

    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Console output for development
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all subsequent log messages.

    This uses structlog's contextvars support to propagate context across
    function calls without explicitly passing it through the call stack.

    Args:
        **kwargs: Key-value pairs to bind to logging context

    Example:
        bind_context(database="prod_db", table="users", catalog="prod")
        logger.info("processing_table")  # Will include database, table, catalog
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Remove specific keys from the logging context.

    Args:
        *keys: Context keys to unbind

    Example:
        unbind_context("table", "catalog")
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """
    Clear all context variables from the logging context.

    Useful for cleaning up context at the end of a request or operation.
    """
    structlog.contextvars.clear_contextvars()


@contextmanager
def logging_context(**kwargs: Any):
    """
    Context manager for temporary logging context.

    Automatically binds context variables on entry and unbinds them on exit.
    This ensures context is properly cleaned up even if exceptions occur.

    Args:
        **kwargs: Key-value pairs to bind to logging context

    Example:
        with logging_context(database="prod_db", table="users"):
            logger.info("processing")  # Includes database and table
            process_table()
        # database and table are automatically removed from context
    """
    # Bind context variables
    bind_context(**kwargs)
    try:
        yield
    finally:
        # Always unbind, even if exception occurs
        unbind_context(*kwargs.keys())


# Configure logging on module import with sensible defaults
configure_logging(level="INFO", json_output=False)
