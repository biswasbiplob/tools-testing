"""Recommendation engine that orchestrates analyzers and computes optimizations."""

import re
from datetime import datetime
from typing import Optional

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


class OptimizationEngine:
    """Main engine that orchestrates analysis and generates recommendations."""

    def __init__(self, config: OptimizerConfig):
        """Initialize the optimization engine."""
        self.config = config
        self.athena = AthenaCollector(config)
        self.glue = GlueCollector(config)

        # Initialize all analyzers
        self.analyzers = [
            ExplainAnalyzer(config),
            PartitionAnalyzer(config),
            FormatAnalyzer(config),
            JoinAnalyzer(config),
            ProjectionAnalyzer(config),
            CostAnalyzer(config),  # Cost analyzer should run last
        ]

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

        # Extract table names from query
        table_names = self._extract_table_names(query)

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
                print(f"Warning: Could not fetch metadata for {table_name}: {e}")

        # Get EXPLAIN plan
        try:
            explain_plan = self.athena.get_explain_plan(query, db)
            context["explain_plan"] = explain_plan
        except Exception as e:
            print(f"Warning: Could not get EXPLAIN plan: {e}")

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
                print(f"Warning: Could not run EXPLAIN ANALYZE: {e}")

        # Run all analyzers
        all_recommendations = []
        for analyzer in self.analyzers:
            if analyzer.enabled:
                try:
                    recommendations = analyzer.analyze(context)
                    all_recommendations.extend(recommendations)
                except Exception as e:
                    print(f"Warning: Analyzer {analyzer.name} failed: {e}")

        # Sort recommendations by severity and confidence
        sorted_recommendations = self._sort_recommendations(all_recommendations)

        # Calculate aggregate costs and savings
        total_current_cost = 0.0
        total_optimized_cost = 0.0
        total_savings = 0.0

        for rec in sorted_recommendations:
            if rec.current_cost_usd:
                total_current_cost += rec.current_cost_usd
            if rec.optimized_cost_usd:
                total_optimized_cost += rec.optimized_cost_usd
            if rec.savings_usd:
                total_savings += rec.savings_usd

        # Calculate savings based on percentages if direct costs not available
        if total_current_cost == 0 and context.get("query_metrics"):
            metrics = context["query_metrics"]
            data_scanned_tb = metrics.data_scanned_bytes / (1024 ** 4)
            total_current_cost = data_scanned_tb * self.config.athena_cost_per_tb

            # Calculate potential optimized cost based on recommendations
            max_savings_percentage = 0.0
            for rec in sorted_recommendations:
                if rec.savings_percentage and rec.savings_percentage > max_savings_percentage:
                    max_savings_percentage = rec.savings_percentage

            if max_savings_percentage > 0:
                total_optimized_cost = total_current_cost * (1 - max_savings_percentage / 100)
                total_savings = total_current_cost - total_optimized_cost

        total_savings_percentage = (
            (total_savings / total_current_cost * 100)
            if total_current_cost > 0
            else 0.0
        )

        return AnalysisResult(
            query=query,
            recommendations=sorted_recommendations,
            explain_plan=context.get("parsed_explain_plan"),
            query_metrics=context.get("query_metrics"),
            table_metadata=context["table_metadata"],
            total_current_cost_usd=total_current_cost,
            total_optimized_cost_usd=total_optimized_cost,
            total_savings_usd=total_savings,
            total_savings_percentage=total_savings_percentage,
            analysis_timestamp=datetime.utcnow().isoformat(),
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
        table_names = self._extract_table_names(query)

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

        # Estimate based on table sizes
        # Actual scan depends on partitions, projections, etc.
        estimated_scan_tb = total_size_bytes / (1024 ** 4)
        estimated_cost = estimated_scan_tb * self.config.athena_cost_per_tb

        # Provide range based on typical optimizations
        min_cost = estimated_cost * 0.1  # With optimal partitioning and projection
        max_cost = estimated_cost  # Full table scan

        return {
            "estimated_cost_usd": estimated_cost,
            "estimated_scan_tb": estimated_scan_tb,
            "cost_range_usd": {
                "min": min_cost,
                "max": max_cost
            },
            "table_sizes_bytes": table_sizes,
            "note": (
                "This is an estimate based on table sizes. "
                "Actual cost depends on partitions, column selection, and filters."
            )
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
            partitions = self.glue.get_partitions(database, table, max_partitions=100)
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

    def _extract_table_names(self, query: str) -> list[str]:
        """Extract table names from SQL query."""
        # Remove comments
        query = re.sub(r'--[^\n]*', '', query)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)

        # Pattern to match FROM and JOIN clauses
        pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)?)'

        matches = re.finditer(pattern, query, re.IGNORECASE)
        tables = []

        for match in matches:
            table = match.group(1)
            # Remove alias if present
            table = table.split()[0]
            if table.lower() not in ['select', 'where', 'group', 'order', 'limit']:
                tables.append(table)

        return list(set(tables))  # Remove duplicates

    def _sort_recommendations(
        self, recommendations: list[Recommendation]
    ) -> list[Recommendation]:
        """Sort recommendations by severity and confidence."""
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }

        return sorted(
            recommendations,
            key=lambda r: (severity_order.get(r.severity, 999), -r.confidence)
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
        elif "text" in format_lower:
            return "TEXT"
        else:
            return "CSV"
