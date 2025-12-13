"""FastMCP server for Athena SQL optimization."""

import json
import os
from typing import Optional
from mcp.server.fastmcp import FastMCP

from .decorators import mcp_tool_handler
from .logging import get_logger
from .exceptions import ConfigurationError
from .models import OptimizerConfig
from .engine import OptimizationEngine

logger = get_logger(__name__)


# Initialize FastMCP server
mcp = FastMCP("athena-sql-optimizer")


# Global engine instance (initialized on startup)
_engine: Optional[OptimizationEngine] = None


def get_engine() -> OptimizationEngine:
    """
    Get the optimization engine instance.

    Returns:
        OptimizationEngine instance

    Raises:
        RuntimeError: If engine not initialized
    """
    if _engine is None:
        raise RuntimeError(
            "Optimizer not initialized. Please check your MCP configuration."
        )
    return _engine


def _create_config_from_env() -> OptimizerConfig:
    """
    Create optimizer configuration from environment variables.

    Returns:
        OptimizerConfig instance

    Raises:
        ConfigurationError: If required environment variables are missing
    """
    workgroup = os.getenv("ATHENA_WORKGROUP")
    s3_output = os.getenv("ATHENA_S3_OUTPUT")

    if not workgroup:
        raise ConfigurationError(
            "ATHENA_WORKGROUP environment variable is required",
            details={"variable": "ATHENA_WORKGROUP"}
        )
    if not s3_output:
        raise ConfigurationError(
            "ATHENA_S3_OUTPUT environment variable is required",
            details={"variable": "ATHENA_S3_OUTPUT"}
        )

    return OptimizerConfig(
        aws_profile=os.getenv("AWS_PROFILE"),
        region=os.getenv("AWS_REGION", "eu-west-1"),
        workgroup=workgroup,
        s3_output_location=s3_output,
        catalog=os.getenv("ATHENA_CATALOG", "AwsDataCatalog"),
        database=os.getenv("ATHENA_DATABASE"),
        run_explain_analyze=os.getenv("RUN_EXPLAIN_ANALYZE", "false").lower() == "true",
        athena_cost_per_tb=float(os.getenv("ATHENA_COST_PER_TB", "5.0")),
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "300"))
    )


def _initialize_engine():
    """Initialize the optimization engine from environment variables."""
    global _engine

    if _engine is not None:
        return  # Already initialized

    try:
        config = _create_config_from_env()
        _engine = OptimizationEngine(config)
        logger.info(
            "mcp_server_started",
            region=config.region,
            workgroup=config.workgroup,
            database=config.database,
        )
    except Exception as e:
        logger.error(
            "mcp_server_startup_failed",
            error=str(e),
            required_vars=["ATHENA_WORKGROUP", "ATHENA_S3_OUTPUT"],
        )
        raise


@mcp.tool()
@mcp_tool_handler
def analyze_sql_query(
    query: str,
    database: Optional[str] = None,
    run_explain_analyze: Optional[bool] = None
) -> dict:
    """
    Analyze an Athena SQL query for performance and cost optimization.

    This tool performs comprehensive analysis including:
    - EXPLAIN plan analysis
    - Partition usage verification
    - Table format recommendations
    - JOIN optimization suggestions
    - Column projection analysis
    - Cost estimation and savings calculation

    Args:
        query: The SQL query to analyze
        database: Database name (optional, uses default from config)
        run_explain_analyze: Whether to run EXPLAIN ANALYZE (executes query, incurs cost)

    Returns:
        JSON string with detailed analysis results and recommendations
    """
    engine = get_engine()
    return engine.analyze_query(query, database, run_explain_analyze)


@mcp.tool()
@mcp_tool_handler
def estimate_query_cost(
    query: str,
    database: Optional[str] = None
) -> dict:
    """
    Estimate the cost of running a query without executing it.

    Uses table metadata and statistics to estimate data scan size and cost.
    Does not execute the query, so no Athena charges are incurred.

    Args:
        query: The SQL query to estimate cost for
        database: Database name (optional, uses default from config)

    Returns:
        JSON string with cost estimation details
    """
    engine = get_engine()
    return engine.estimate_cost(query, database)


@mcp.tool()
@mcp_tool_handler
def check_table_health(
    database: str,
    table: str
) -> dict:
    """
    Analyze table structure, format, and health.

    Provides detailed information about:
    - Table format (Parquet, ORC, CSV, etc.)
    - Partition structure
    - Column definitions
    - Table statistics
    - Optimization recommendations

    Args:
        database: Database name
        table: Table name

    Returns:
        JSON string with table health report
    """
    engine = get_engine()
    return engine.check_table_health(database, table)


@mcp.tool()
@mcp_tool_handler
def get_server_diagnostics() -> dict:
    """
    Get MCP server diagnostics and performance metrics.

    Provides cache statistics and server health information to help
    debug performance issues and monitor the optimizer's effectiveness.

    Returns:
        Server diagnostics including:
        - Metadata cache statistics (hit rate, size, TTL)
        - Partition cache statistics (hit rate, size, TTL)
        - Server status

    Example:
        {
          "metadata_cache": {
            "size": 42,
            "hits": 150,
            "misses": 8,
            "hit_rate": 94.94,
            "ttl_seconds": 300
          },
          "partition_cache": {
            "size": 15,
            "hits": 89,
            "misses": 12,
            "hit_rate": 88.12,
            "ttl_seconds": 120
          },
          "status": "healthy"
        }
    """
    engine = get_engine()
    return {
        "metadata_cache": engine.glue._metadata_cache.get_stats(),
        "partition_cache": engine.glue._partition_cache.get_stats(),
        "status": "healthy"
    }


def main():
    """
    Main entry point for the MCP server.

    Initializes the optimization engine from environment variables,
    then starts the MCP server.

    Required environment variables:
    - ATHENA_WORKGROUP: Athena workgroup name
    - ATHENA_S3_OUTPUT: S3 output location for query results

    Optional environment variables:
    - AWS_PROFILE: AWS profile name
    - AWS_REGION: AWS region (default: eu-west-1)
    - ATHENA_CATALOG: Glue catalog name (default: AwsDataCatalog)
    - ATHENA_DATABASE: Default database name
    - RUN_EXPLAIN_ANALYZE: Run EXPLAIN ANALYZE (default: false)
    - ATHENA_COST_PER_TB: Cost per TB in USD (default: 5.0)
    - TIMEOUT_SECONDS: Query timeout (default: 300)
    """
    # Initialize engine before starting MCP server
    _initialize_engine()

    # Start MCP server
    mcp.run()


if __name__ == "__main__":
    main()
