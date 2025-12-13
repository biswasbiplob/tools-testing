"""Recommendation engine that orchestrates analyzers and computes optimizations."""

from datetime import datetime, timezone
from typing import Optional

from .logging import get_logger
from .sql_parser import extract_table_names
from .cost_calculator import CostCalculator
from .constants import (
    DEFAULT_MAX_PARTITIONS,
    SEVERITY_ORDER_CRITICAL,
    SEVERITY_ORDER_HIGH,
    SEVERITY_ORDER_MEDIUM,
    SEVERITY_ORDER_LOW,
    SEVERITY_ORDER_INFO,
    SEVERITY_ORDER_DEFAULT,
)
from .models import (
    AnalysisResult,
    OptimizerConfig,
    QueryMetrics,
    Recommendation,
    Severity,
)
from .collectors import AthenaCollector, GlueCollector
from .analyzers import (
    CostAnalyzer,
    ExplainAnalyzer,
    FormatAnalyzer,
    JoinAnalyzer,
    PartitionAnalyzer,
    ProjectionAnalyzer,
)

logger = get_logger(__name__)


class OptimizationEngine:
    """Main engine that orchestrates analysis and generates recommendations."""

    def __init__(self, config: OptimizerConfig):
        """Initialize the optimization engine."""
        self.config = config
        self.athena = AthenaCollector(config)
        self.glue = GlueCollector(config)
        self.cost_calculator = CostCalculator(cost_per_tb=config.athena_cost_per_tb)
        self._closed = False

        # Initialize all analyzers
        self.analyzers = [
            ExplainAnalyzer(config),
            PartitionAnalyzer(config),
            FormatAnalyzer(config),
            JoinAnalyzer(config),
            ProjectionAnalyzer(config),
            CostAnalyzer(config),  # Cost analyzer should run last
        ]

    def close(self):
        """Clean up all resources."""
        if not self._closed:
            self.athena.close()
            self.glue.close()
            self._closed = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass  # Ignore errors in __del__

    def analyze_query(
        self,
        query: str,
        database: Optional[str] = None,
        run_explain_analyze: Optional[bool] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis of a SQL query.

        Args:
            query: The SQL query to analyze
            database: Database name (optional, uses config default)
            run_explain_analyze: Whether to run EXPLAIN ANALYZE (optional, uses config default)

        Returns:
            Complete analysis result with recommendations
        """
        db = database or self.config.database

        # Initialize context for analyzers
        context = {
            "query": query,
            "database": db,
            "table_metadata": {},
            "query_metrics": None,
            "explain_plan": None,
            "explain_analyze_plan": None,
        }

        # Extract table names from query using proper SQL parsing
        table_names = extract_table_names(query)

        # Collect table metadata
        for table_name in table_names:
            try:
                # Handle database.table notation
                if "." in table_name:
                    table_db, table = table_name.split(".", 1)
                else:
                    table_db, table = db, table_name

                if table_db:
                    metadata = self.glue.get_table_metadata(table_db, table)
                    context["table_metadata"][table_name] = metadata
            except Exception as e:
                # Table might not exist or access denied - continue anyway
                logger.warning(
                    "failed_to_fetch_table_metadata",
                    table_name=table_name,
                    database=table_db,
                    error=str(e),
                )

        # Get EXPLAIN plan
        try:
            explain_plan = self.athena.get_explain_plan(query, db)
            context["explain_plan"] = explain_plan
        except Exception as e:
            logger.warning(
                "failed_to_get_explain_plan",
                database=db,
                error=str(e),
            )

        # Optionally run EXPLAIN ANALYZE
        should_run_analyze = (
            run_explain_analyze
            if run_explain_analyze is not None
            else self.config.run_explain_analyze
        )

        if should_run_analyze:
            try:
                analyze_plan, metrics = self.athena.get_explain_analyze_plan(query, db)
                context["explain_analyze_plan"] = analyze_plan
                context["query_metrics"] = metrics
            except Exception as e:
                logger.warning(
                    "failed_to_run_explain_analyze",
                    database=db,
                    error=str(e),
                )

        # Run all analyzers
        all_recommendations = []
        for analyzer in self.analyzers:
            if analyzer.enabled:
                try:
                    recommendations = analyzer.analyze(context)
                    all_recommendations.extend(recommendations)
                except Exception as e:
                    logger.warning(
                        "analyzer_failed",
                        analyzer_name=analyzer.name,
                        error=str(e),
                    )

        # Sort recommendations by severity and confidence
        sorted_recommendations = self._sort_recommendations(all_recommendations)

        # Use cost calculator to analyze costs and savings
        cost_analysis = self.cost_calculator.analyze_costs_from_recommendations(
            sorted_recommendations,
            context.get("query_metrics")
        )

        return AnalysisResult(
            query=query,
            recommendations=sorted_recommendations,
            explain_plan=context.get("parsed_explain_plan"),
            query_metrics=context.get("query_metrics"),
            table_metadata=context["table_metadata"],
            total_current_cost_usd=cost_analysis.total_current_cost_usd,
            total_optimized_cost_usd=cost_analysis.total_optimized_cost_usd,
            total_savings_usd=cost_analysis.total_savings_usd,
            total_savings_percentage=cost_analysis.total_savings_percentage,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            config={
                "region": self.config.region,
                "workgroup": self.config.workgroup,
                "database": db,
            }
        )

    def estimate_cost(
        self, query: str, database: Optional[str] = None
    ) -> dict:
        """
        Estimate query cost without execution using table metadata.

        Args:
            query: The SQL query to estimate
            database: Database name (optional)

        Returns:
            Cost estimation with breakdown
        """
        db = database or self.config.database
        table_names = extract_table_names(query)

        total_size_bytes = 0
        table_sizes = {}

        for table_name in table_names:
            try:
                if "." in table_name:
                    table_db, table = table_name.split(".", 1)
                else:
                    table_db, table = db, table_name

                if table_db:
                    stats = self.glue.get_table_statistics(table_db, table)
                    size = int(stats.get("total_size", 0))
                    table_sizes[table_name] = size
                    total_size_bytes += size
            except Exception:
                # Could not get size, skip
                pass

        # Use cost calculator to estimate costs
        cost_estimate = self.cost_calculator.estimate_cost_from_table_sizes(table_sizes)

        return {
            "estimated_cost_usd": cost_estimate.estimated_cost_usd,
            "estimated_scan_tb": cost_estimate.estimated_scan_tb,
            "cost_range_usd": {
                "min": cost_estimate.min_cost_usd,
                "max": cost_estimate.max_cost_usd
            },
            "table_sizes_tb": cost_estimate.table_sizes,
            "table_sizes_bytes": table_sizes,
            "assumptions": cost_estimate.assumptions
        }

    def check_table_health(
        self, database: str, table: str
    ) -> dict:
        """
        Analyze table health and structure.

        Args:
            database: Database name
            table: Table name

        Returns:
            Table health report
        """
        metadata = self.glue.get_table_metadata(database, table)
        stats = self.glue.get_table_statistics(database, table)

        # Get partition info if partitioned
        partition_info = {}
        if metadata.partition_keys:
            partitions = self.glue.get_partitions(database, table, max_partitions=DEFAULT_MAX_PARTITIONS)
            partition_info = {
                "partition_keys": metadata.partition_keys,
                "partition_count": len(partitions),
                "sample_partitions": [p["values"] for p in partitions[:5]]
            }

        # Detect format
        format_type = self._detect_format(metadata.input_format or "")

        # Generate recommendations
        recommendations = []

        if format_type not in ["PARQUET", "ORC"]:
            recommendations.append({
                "severity": "HIGH",
                "category": "FORMAT",
                "message": f"Table uses {format_type} format. Consider Parquet or ORC for better performance."
            })

        if not metadata.partition_keys:
            recommendations.append({
                "severity": "MEDIUM",
                "category": "PARTITION",
                "message": "Table is not partitioned. Consider partitioning for better performance."
            })

        return {
            "database": database,
            "table": table,
            "format": format_type,
            "location": metadata.location,
            "columns": metadata.columns,
            "partition_info": partition_info,
            "statistics": stats,
            "recommendations": recommendations,
            "metadata": metadata.model_dump()
        }

    def _sort_recommendations(
        self, recommendations: list[Recommendation]
    ) -> list[Recommendation]:
        """Sort recommendations by severity and confidence."""
        severity_order = {
            Severity.CRITICAL: SEVERITY_ORDER_CRITICAL,
            Severity.HIGH: SEVERITY_ORDER_HIGH,
            Severity.MEDIUM: SEVERITY_ORDER_MEDIUM,
            Severity.LOW: SEVERITY_ORDER_LOW,
            Severity.INFO: SEVERITY_ORDER_INFO,
        }

        return sorted(
            recommendations,
            key=lambda r: (severity_order.get(r.severity, SEVERITY_ORDER_DEFAULT), -r.confidence)
        )

    def _detect_format(self, input_format: str) -> str:
        """Detect table format from input format string."""
        format_lower = input_format.lower()

        if "parquet" in format_lower:
            return "PARQUET"
        elif "orc" in format_lower:
            return "ORC"
        elif "avro" in format_lower:
            return "AVRO"
        elif "json" in format_lower:
            return "JSON"
        # Check for CSV first (TextInputFormat is commonly used for CSV)
        elif input_format == "org.apache.hadoop.mapred.TextInputFormat":
            return "CSV"
        elif "text" in format_lower:
            return "TEXT"
        else:
            return "CSV"
