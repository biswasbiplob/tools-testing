"""Decorators for MCP tools and other utilities."""

import json
import functools
from typing import Callable, Any

from .logging import get_logger

logger = get_logger(__name__)


def mcp_tool_handler(func: Callable) -> Callable:
    """
    Decorator for MCP tools that handles errors and JSON serialization.

    This decorator:
    - Catches exceptions and returns formatted error JSON
    - Logs errors with structured logging
    - Serializes results to JSON with proper formatting
    - Includes function arguments in error responses for debugging

    Usage:
        @mcp.tool()
        @mcp_tool_handler
        def my_tool(arg1: str, arg2: int) -> dict:
            # ... tool implementation
            return result_dict
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            result = func(*args, **kwargs)

            # Handle Pydantic models
            if hasattr(result, "model_dump"):
                result = result.model_dump()

            # Return JSON formatted result
            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            # Get function signature for error context
            func_name = func.__name__

            # Build error context with function arguments
            error_context = {
                "error": str(e),
                "status": "failed",
            }

            # Add all kwargs to error context
            error_context.update(kwargs)

            # Log the error with context
            logger.error(
                "mcp_tool_failed",
                tool_name=func_name,
                error=str(e),
                **kwargs
            )

            return json.dumps(error_context, indent=2)

    return wrapper
