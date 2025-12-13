"""FastMCP server for Athena SQL optimization."""

import json
from typing import Optional
from mcp.server.fastmcp import FastMCP

from .models import OptimizerConfig
from .engine import OptimizationEngine


# Initialize FastMCP server
mcp = FastMCP("athena-sql-optimizer")


# Global engine instance (will be initialized with config)
_engine: Optional[OptimizationEngine] = None


def get_engine() -> OptimizationEngine:
    """Get or create the optimization engine."""
    global _engine
    if _engine is None:
        raise RuntimeError(
            "Optimizer not initialized. Please check your MCP configuration."
        )
    return _engine


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
    """Initialize the optimization engine with configuration."""
    global _engine

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

    _engine = OptimizationEngine(config)


@mcp.tool()
def analyze_sql_query(
    query: str,
    database: Optional[str] = None,
    run_explain_analyze: Optional[bool] = None
) -> str:
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
    try:
        engine = get_engine()
        result = engine.analyze_query(query, database, run_explain_analyze)

        # Convert to dict for JSON serialization
        return json.dumps(result.model_dump(), indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "query": query,
            "status": "failed"
        }, indent=2)


@mcp.tool()
def estimate_query_cost(
    query: str,
    database: Optional[str] = None
) -> str:
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
    try:
        engine = get_engine()
        result = engine.estimate_cost(query, database)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "query": query,
            "status": "failed"
        }, indent=2)


@mcp.tool()
def check_table_health(
    database: str,
    table: str
) -> str:
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
    try:
        engine = get_engine()
        result = engine.check_table_health(database, table)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "database": database,
            "table": table,
            "status": "failed"
        }, indent=2)


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
            print(f"Error parsing config: {e}", file=sys.stderr)
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
            print(f"Error initializing from environment: {e}", file=sys.stderr)
            print("Required: ATHENA_WORKGROUP, ATHENA_S3_OUTPUT", file=sys.stderr)
            sys.exit(1)

    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()
