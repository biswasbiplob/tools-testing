"""Test new features: logging context and cache diagnostics."""

import pytest
from unittest.mock import Mock, MagicMock
from athena_optimizer.server import get_server_diagnostics, _engine
from athena_optimizer.engine import OptimizationEngine
from athena_optimizer.models import OptimizerConfig


def test_server_diagnostics_tool():
    """Test that the diagnostics tool returns cache stats."""
    # Create a mock engine with mock caches
    mock_engine = Mock(spec=OptimizationEngine)
    mock_glue = Mock()

    # Mock metadata cache
    mock_metadata_cache = Mock()
    mock_metadata_cache.get_stats.return_value = {
        "size": 10,
        "hits": 50,
        "misses": 5,
        "hit_rate": 90.91,
        "ttl_seconds": 300
    }

    # Mock partition cache
    mock_partition_cache = Mock()
    mock_partition_cache.get_stats.return_value = {
        "size": 5,
        "hits": 30,
        "misses": 10,
        "hit_rate": 75.0,
        "ttl_seconds": 120
    }

    mock_glue._metadata_cache = mock_metadata_cache
    mock_glue._partition_cache = mock_partition_cache
    mock_engine.glue = mock_glue

    # Temporarily set the engine
    import athena_optimizer.server as server_module
    original_engine = server_module._engine
    server_module._engine = mock_engine

    try:
        # Call the diagnostics tool
        result = get_server_diagnostics()

        # Verify the result
        assert result["status"] == "healthy"
        assert "metadata_cache" in result
        assert "partition_cache" in result
        assert result["metadata_cache"]["hit_rate"] == 90.91
        assert result["partition_cache"]["hit_rate"] == 75.0

        # Verify cache stats were called
        mock_metadata_cache.get_stats.assert_called_once()
        mock_partition_cache.get_stats.assert_called_once()

    finally:
        # Restore original engine
        server_module._engine = original_engine


def test_logging_context_basic_functionality():
    """Test that logging context can be used without errors."""
    from athena_optimizer.logging import logging_context, get_logger

    # Just verify logging context works without errors
    logger = get_logger("test")

    # This should not raise any exceptions
    with logging_context(database="test_db", query_id=1234):
        logger.info("test_event", extra_field="value")

    # If we get here, logging context works correctly
    # (actual context propagation is tested via integration tests)
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
