"""EXPLAIN and EXPLAIN ANALYZE plan analyzer."""

import re
from typing import Any

from ..models import Recommendation, Severity, Category, Effort, ExplainPlan
from .base import BaseAnalyzer


class ExplainAnalyzer(BaseAnalyzer):
    """Analyzes EXPLAIN and EXPLAIN ANALYZE output."""

    @property
    def name(self) -> str:
        return "ExplainAnalyzer"

    def analyze(self, context: dict[str, Any]) -> list[Recommendation]:
        """Analyze explain plans for optimization opportunities."""
        recommendations = []

        explain_plan_text = context.get("explain_plan")
        if not explain_plan_text:
            return recommendations

        # Parse the plan
        plan = self._parse_explain_plan(explain_plan_text)
        context["parsed_explain_plan"] = plan

        # Check for table scans
        if plan.table_scans:
            for table_scan in plan.table_scans:
                if "filter" not in table_scan.lower() and "predicate" not in table_scan.lower():
                    recommendations.append(Recommendation(
                        severity=Severity.MEDIUM,
                        category=Category.QUERY_PATTERN,
                        title="Full Table Scan Detected",
                        description=(
                            f"Query performs a full table scan on {table_scan}. "
                            "This can be expensive for large tables."
                        ),
                        confidence=0.8,
                        effort=Effort.LOW,
                        action_plan=[
                            "Add WHERE clause to filter data",
                            "Use partition columns in filters if available",
                            "Consider if all rows are truly needed"
                        ],
                        metadata={"table": table_scan}
                    ))

        # Check for cross joins
        if "cross join" in explain_plan_text.lower():
            recommendations.append(Recommendation(
                severity=Severity.CRITICAL,
                category=Category.JOIN,
                title="Cross Join Detected",
                description=(
                    "Query contains a CROSS JOIN, which can produce cartesian products "
                    "and result in extremely large result sets."
                ),
                confidence=1.0,
                effort=Effort.MEDIUM,
                action_plan=[
                    "Review join conditions",
                    "Add appropriate JOIN conditions to avoid cartesian products",
                    "Consider if CROSS JOIN is intentional"
                ],
                references=[
                    "https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html#performance-tuning-join-order-and-distribution"
                ]
            ))

        # Check for multiple fragments (distributed query)
        if plan.fragment_count > 5:
            recommendations.append(Recommendation(
                severity=Severity.LOW,
                category=Category.QUERY_PATTERN,
                title="Complex Distributed Query",
                description=(
                    f"Query is distributed across {plan.fragment_count} fragments. "
                    "This may indicate complex joins or aggregations."
                ),
                confidence=0.6,
                effort=Effort.MEDIUM,
                action_plan=[
                    "Review if query can be simplified",
                    "Consider breaking into smaller queries with intermediate tables",
                    "Check if all joins are necessary"
                ]
            ))

        # Check for expensive operations
        expensive_ops = self._find_expensive_operations(explain_plan_text)
        for op in expensive_ops:
            recommendations.append(Recommendation(
                severity=Severity.MEDIUM,
                category=Category.QUERY_PATTERN,
                title=f"Expensive Operation: {op['type']}",
                description=op["description"],
                confidence=0.7,
                effort=Effort.MEDIUM,
                action_plan=op["action_plan"],
                metadata={"operation": op["type"]}
            ))

        return recommendations

    def _parse_explain_plan(self, plan_text: str) -> ExplainPlan:
        """Parse EXPLAIN plan output."""
        lines = plan_text.split("\n")

        # Count fragments
        fragment_count = len([line for line in lines if "Fragment" in line])

        # Extract table scans
        table_scans = []
        scan_pattern = r"TableScan\[(.+?)\]|ScanProject\[table = (.+?)[,\]]"
        for match in re.finditer(scan_pattern, plan_text):
            table = match.group(1) or match.group(2)
            if table:
                table_scans.append(table.strip())

        # Extract operations
        operation_patterns = [
            "TableScan", "Filter", "Project", "Aggregate", "Join",
            "Exchange", "Sort", "Limit", "Window", "GroupBy"
        ]
        operations = []
        for op in operation_patterns:
            if op.lower() in plan_text.lower():
                operations.append(op)

        # Check for partition filters
        partition_filters = []
        if "partition" in plan_text.lower():
            filter_matches = re.finditer(r"(\w+)\s*=\s*['\"]?([^'\"]+)['\"]?", plan_text)
            for match in filter_matches:
                if "partition" in plan_text[max(0, match.start()-50):match.start()].lower():
                    partition_filters.append(f"{match.group(1)} = {match.group(2)}")

        return ExplainPlan(
            raw_plan=plan_text,
            distributed=fragment_count > 1,
            fragment_count=fragment_count,
            operations=operations,
            table_scans=table_scans,
            partition_filters=partition_filters
        )

    def _find_expensive_operations(self, plan_text: str) -> list[dict]:
        """Identify expensive operations in the plan."""
        expensive = []

        # Check for sorts without limits
        if "sort" in plan_text.lower() and "limit" not in plan_text.lower():
            expensive.append({
                "type": "Sort without LIMIT",
                "description": (
                    "Query performs a sort operation without a LIMIT clause. "
                    "This requires sorting the entire result set."
                ),
                "action_plan": [
                    "Add LIMIT clause if you don't need all results",
                    "Consider if ORDER BY is necessary",
                    "Use indexed columns for sorting if available"
                ]
            })

        # Check for DISTINCT with many columns
        if "distinct" in plan_text.lower():
            expensive.append({
                "type": "DISTINCT operation",
                "description": (
                    "DISTINCT requires deduplication which can be expensive. "
                    "Consider if it's necessary."
                ),
                "action_plan": [
                    "Check if data is already unique",
                    "Use GROUP BY with aggregation if more appropriate",
                    "Consider deduplicating at the source"
                ]
            })

        # Check for window functions
        if "window" in plan_text.lower():
            expensive.append({
                "type": "Window function",
                "description": (
                    "Window functions can be expensive, especially with large "
                    "partition windows."
                ),
                "action_plan": [
                    "Ensure PARTITION BY uses appropriate columns",
                    "Consider if aggregation can be used instead",
                    "Limit window frame if possible"
                ]
            })

        return expensive
