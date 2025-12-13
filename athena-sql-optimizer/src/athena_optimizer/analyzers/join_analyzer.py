"""JOIN operation analyzer."""

import re
from typing import Any

from ..models import Recommendation, Severity, Category, Effort
from .base import BaseAnalyzer


class JoinAnalyzer(BaseAnalyzer):
    """Analyzes JOIN operations for optimization opportunities."""

    @property
    def name(self) -> str:
        return "JoinAnalyzer"

    def analyze(self, context: dict[str, Any]) -> list[Recommendation]:
        """Analyze JOIN patterns in the query."""
        recommendations = []

        query = context.get("query", "")
        table_metadata = context.get("table_metadata", {})

        # Extract JOIN information
        joins = self._extract_joins(query)

        if not joins:
            return recommendations

        # Check for multiple joins
        if len(joins) > 3:
            recommendations.append(Recommendation(
                severity=Severity.MEDIUM,
                category=Category.JOIN,
                title="Multiple JOINs Detected",
                description=(
                    f"Query contains {len(joins)} JOIN operations. "
                    "Consider if all joins are necessary or if intermediate "
                    "materialization would help."
                ),
                confidence=0.7,
                effort=Effort.MEDIUM,
                action_plan=[
                    "Review if all joins are necessary",
                    "Consider creating intermediate materialized views",
                    "Ensure join order is optimal (small tables first)",
                    "Use EXPLAIN to verify join order"
                ],
                references=[
                    "https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html#performance-tuning-join-order-and-distribution"
                ]
            ))

        # Check for OUTER JOINs
        outer_joins = [j for j in joins if j["type"] in ["LEFT", "RIGHT", "FULL"]]
        if outer_joins:
            recommendations.append(Recommendation(
                severity=Severity.LOW,
                category=Category.JOIN,
                title="OUTER JOINs May Be Expensive",
                description=(
                    f"Query uses {len(outer_joins)} OUTER JOIN(s). "
                    "These can be more expensive than INNER JOINs. "
                    "Verify they're necessary."
                ),
                confidence=0.6,
                effort=Effort.LOW,
                action_plan=[
                    "Check if INNER JOIN would suffice",
                    "Consider filtering NULL values after join instead",
                    "Ensure join conditions are selective"
                ],
                metadata={"outer_joins": len(outer_joins)}
            ))

        # Check for non-equality joins
        non_equality_joins = [j for j in joins if j["condition"] and
                              "!=" in j["condition"] or "<>" in j["condition"] or
                              "<" in j["condition"] or ">" in j["condition"]]

        if non_equality_joins:
            recommendations.append(Recommendation(
                severity=Severity.HIGH,
                category=Category.JOIN,
                title="Non-Equality JOIN Conditions",
                description=(
                    "Query uses non-equality JOIN conditions (!=, <, >, etc.). "
                    "These can be very expensive and may result in cartesian products."
                ),
                confidence=0.9,
                effort=Effort.MEDIUM,
                action_plan=[
                    "Use equality conditions where possible",
                    "Apply range filters in WHERE clause instead",
                    "Consider if the logic can be restructured"
                ],
                references=[
                    "https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html#performance-tuning-join-order-and-distribution"
                ]
            ))

        # Check for skewed joins based on table metadata
        if table_metadata and len(table_metadata) > 1:
            recommendations.append(Recommendation(
                severity=Severity.INFO,
                category=Category.JOIN,
                title="JOIN Order Optimization",
                description=(
                    "Athena performs better when smaller tables are on the right side "
                    "of JOINs. Verify join order is optimal."
                ),
                confidence=0.7,
                effort=Effort.LOW,
                action_plan=[
                    "Place smaller tables on the right side of JOIN",
                    "Use EXPLAIN to see actual join order",
                    "Consider collecting table statistics",
                    "Apply filters before joins to reduce data size"
                ]
            ))

        return recommendations

    def _extract_joins(self, query: str) -> list[dict]:
        """Extract JOIN information from query."""
        joins = []

        # Pattern for different join types
        join_pattern = r'\b(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+(\w+)(?:\s+AS\s+\w+)?\s+ON\s+([^;]+?)(?=\s+(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN|WHERE|GROUP|ORDER|LIMIT|$)'

        matches = re.finditer(join_pattern, query, re.IGNORECASE | re.DOTALL)

        for match in matches:
            join_type = match.group(1) or "INNER"
            table = match.group(2)
            condition = match.group(3).strip() if match.group(3) else None

            joins.append({
                "type": join_type.upper(),
                "table": table,
                "condition": condition
            })

        return joins
