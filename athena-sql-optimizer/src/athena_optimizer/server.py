"""FastMCP server for Athena SQL optimization."""

import json
import threading
from typing import Optional
from mcp.server.fastmcp import FastMCP

from .decorators import mcp_tool_handler
from .logging import get_logger
from .models import OptimizerConfig
from .engine import OptimizationEngine

logger = get_logger(__name__)


# Initialize FastMCP server
mcp = FastMCP("athena-sql-optimizer")


# Thread-local storage for engine instances
_thread_local = threading.local()


def get_engine() -> OptimizationEngine:
    """
    Get the optimization engine for the current thread.

    Returns:
        OptimizationEngine instance for this thread

    Raises:
        RuntimeError: If engine not initialized for this thread
    """
    engine = getattr(_thread_local, "engine", None)
    if engine is None:
        raise RuntimeError(
            "Optimizer not initialized. Please check your MCP configuration."
        )
    return engine


def initialize_engine(
    aws_profile: Optional[str] = None,
    region: str = "eu-west-1",
    workgroup: Optional[str] = None,
    s3_output_location: Optional[str] = None,
    catalog: str = "AwsDataCatalog",
    database: Optional[str] = None,
    run_explain_analyze: bool = False,
    athena_cost_per_tb: float = 5.0,
    timeout_seconds: int = 300
) -> None:
    """
    Initialize the optimization engine with configuration for the current thread.

    This function creates a thread-local engine instance, ensuring thread-safety
    in multi-threaded environments.

    Args:
        aws_profile: AWS credentials profile name
        region: AWS region (default: eu-west-1)
        workgroup: Athena workgroup name (required)
        s3_output_location: S3 location for query results (required)
        catalog: Glue catalog name (default: AwsDataCatalog)
        database: Default database name
        run_explain_analyze: Whether to run EXPLAIN ANALYZE by default
        athena_cost_per_tb: Cost per TB of data scanned (default: 5.0 USD)
        timeout_seconds: Query timeout in seconds (default: 300)

    Raises:
        ValueError: If required parameters are missing
    """
    # Validate required parameters
    if not workgroup:
        raise ValueError("workgroup parameter is required")
    if not s3_output_location:
        raise ValueError("s3_output_location parameter is required")

    config = OptimizerConfig(
        aws_profile=aws_profile,
        region=region,
        workgroup=workgroup,
        s3_output_location=s3_output_location,
        catalog=catalog,
        database=database,
        run_explain_analyze=run_explain_analyze,
        athena_cost_per_tb=athena_cost_per_tb,
        timeout_seconds=timeout_seconds
    )

    _thread_local.engine = OptimizationEngine(config)
    logger.info(
        "engine_initialized",
        region=region,
        workgroup=workgroup,
        database=database,
    )


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


# Allow initialization via environment or config
def main():
    """Main entry point for the MCP server."""
    import os
    import sys

    # Check if config is provided via command line
    if len(sys.argv) > 1:
        # Expect JSON config as first argument
        try:
            config_json = sys.argv[1]
            config_dict = json.loads(config_json)
            initialize_engine(**config_dict)
        except Exception as e:
            logger.error("failed_to_parse_config", error=str(e))
            sys.exit(1)
    else:
        # Try to initialize from environment variables
        try:
            initialize_engine(
                aws_profile=os.getenv("AWS_PROFILE"),
                region=os.getenv("AWS_REGION", "eu-west-1"),
                workgroup=os.getenv("ATHENA_WORKGROUP"),
                s3_output_location=os.getenv("ATHENA_S3_OUTPUT"),
                catalog=os.getenv("ATHENA_CATALOG", "AwsDataCatalog"),
                database=os.getenv("ATHENA_DATABASE"),
                run_explain_analyze=os.getenv("RUN_EXPLAIN_ANALYZE", "false").lower() == "true",
                athena_cost_per_tb=float(os.getenv("ATHENA_COST_PER_TB", "5.0")),
                timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "300"))
            )
        except Exception as e:
            logger.error(
                "failed_to_initialize_from_environment",
                error=str(e),
                required_vars=["ATHENA_WORKGROUP", "ATHENA_S3_OUTPUT"],
            )
            sys.exit(1)

    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()
