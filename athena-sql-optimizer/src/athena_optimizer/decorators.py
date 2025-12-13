"""Decorators for MCP tools and other utilities."""

import functools
from typing import Callable, Any

from .logging import get_logger

logger = get_logger(__name__)


def mcp_tool_handler(func: Callable) -> Callable:
    """
    Decorator for MCP tools that handles Pydantic model serialization and logging.

    This decorator:
    - Converts Pydantic models to dictionaries
    - Logs tool execution with structured logging
    - Lets FastMCP handle errors and JSON serialization naturally

    Usage:
        @mcp.tool()
        @mcp_tool_handler
        def my_tool(arg1: str, arg2: int) -> dict:
            # ... tool implementation
            return result_dict
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        # Log tool invocation
        logger.debug(
            "mcp_tool_invoked",
            tool_name=func.__name__,
            **kwargs
        )

        result = func(*args, **kwargs)

        # Handle Pydantic models - convert to dict
        if hasattr(result, "model_dump"):
            result = result.model_dump()

        # Return dict - FastMCP will handle JSON serialization
        return result

    return wrapper
