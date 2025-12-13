"""Cost analysis for Athena queries."""

from typing import Any

from ..models import Recommendation, Severity, Category, Effort
from .base import BaseAnalyzer


class CostAnalyzer(BaseAnalyzer):
    """Analyzes query costs and provides cost optimization recommendations."""

    @property
    def name(self) -> str:
        return "CostAnalyzer"

    def analyze(self, context: dict[str, Any]) -> list[Recommendation]:
        """Analyze query costs."""
        recommendations = []

        metrics = context.get("query_metrics")
        table_metadata = context.get("table_metadata", {})

        if not metrics:
            return recommendations

        # Calculate current cost
        data_scanned_tb = metrics.data_scanned_bytes / (1024 ** 4)
        current_cost = data_scanned_tb * self.config.athena_cost_per_tb

        # High cost warning
        if current_cost > 1.0:
            recommendations.append(Recommendation(
                severity=Severity.HIGH,
                category=Category.COST,
                title="High Query Cost Detected",
                description=(
                    f"This query scans {data_scanned_tb:.2f} TB of data, "
                    f"costing ${current_cost:.2f}. Consider optimizing to reduce costs."
                ),
                current_cost_usd=current_cost,
                data_scanned_current_tb=data_scanned_tb,
                confidence=1.0,
                effort=Effort.MEDIUM,
                action_plan=[
                    "Review partition filters to reduce data scanned",
                    "Select only necessary columns instead of SELECT *",
                    "Use columnar formats (Parquet/ORC) if not already",
                    "Consider materialized views for frequently run queries"
                ],
                references=[
                    "https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html"
                ]
            ))
        elif current_cost > 0.1:
            recommendations.append(Recommendation(
                severity=Severity.MEDIUM,
                category=Category.COST,
                title="Moderate Query Cost",
                description=(
                    f"This query scans {data_scanned_tb:.4f} TB of data, "
                    f"costing ${current_cost:.4f}. There may be room for optimization."
                ),
                current_cost_usd=current_cost,
                data_scanned_current_tb=data_scanned_tb,
                confidence=0.8,
                effort=Effort.LOW,
                action_plan=[
                    "Review if all selected columns are necessary",
                    "Check if partition filters can be applied"
                ],
                references=[
                    "https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html"
                ]
            ))

        # Check for full table scans
        if table_metadata:
            for table_name, metadata in table_metadata.items():
                if metadata.partition_keys:
                    # Table is partitioned but may not be using filters
                    recommendations.append(Recommendation(
                        severity=Severity.INFO,
                        category=Category.COST,
                        title=f"Partitioned Table: {table_name}",
                        description=(
                            f"Table {table_name} has partitions on "
                            f"{', '.join(metadata.partition_keys)}. "
                            "Ensure partition filters are used to minimize costs."
                        ),
                        confidence=0.7,
                        effort=Effort.LOW,
                        action_plan=[
                            f"Add WHERE clause filtering on {', '.join(metadata.partition_keys)}"
                        ],
                        metadata={"table": table_name, "partitions": metadata.partition_keys}
                    ))

        return recommendations
