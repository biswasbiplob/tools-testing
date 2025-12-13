"""Query projection and column selection analyzer."""

import re
from typing import Any

from ..models import Recommendation, Severity, Category, Effort
from .base import BaseAnalyzer


class ProjectionAnalyzer(BaseAnalyzer):
    """Analyzes SELECT projections for optimization opportunities."""

    @property
    def name(self) -> str:
        return "ProjectionAnalyzer"

    def analyze(self, context: dict[str, Any]) -> list[Recommendation]:
        """Analyze column projections in the query."""
        recommendations = []

        query = context.get("query", "")
        table_metadata = context.get("table_metadata", {})

        # Check for SELECT *
        if self._has_select_star(query):
            # Calculate potential savings
            potential_savings = 0.50  # Assume 50% reduction by selecting specific columns

            recommendations.append(Recommendation(
                severity=Severity.HIGH,
                category=Category.PROJECTION,
                title="SELECT * Detected",
                description=(
                    "Query uses SELECT * which scans all columns. "
                    "Selecting only needed columns can reduce costs by 50% or more "
                    "with columnar formats."
                ),
                savings_percentage=potential_savings * 100,
                confidence=0.9,
                effort=Effort.LOW,
                action_plan=[
                    "Replace SELECT * with specific column names",
                    "Select only columns needed for your use case",
                    "This is especially important with columnar formats (Parquet/ORC)"
                ],
                code_example=self._generate_projection_example(query, table_metadata),
                references=[
                    "https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html#performance-tuning-columnar-formats"
                ]
            ))

        # Check for COUNT(*)
        if re.search(r'\bCOUNT\s*\(\s*\*\s*\)', query, re.IGNORECASE):
            recommendations.append(Recommendation(
                severity=Severity.MEDIUM,
                category=Category.PROJECTION,
                title="COUNT(*) Usage",
                description=(
                    "COUNT(*) scans all columns. Using COUNT(1) or COUNT(specific_column) "
                    "can be more efficient, especially with columnar formats."
                ),
                confidence=0.7,
                effort=Effort.LOW,
                action_plan=[
                    "Replace COUNT(*) with COUNT(1)",
                    "Or use COUNT(partition_column) if table is partitioned",
                    "This allows Athena to read fewer columns"
                ],
                code_example=query.replace("COUNT(*)", "COUNT(1)").replace("count(*)", "count(1)")
            ))

        # Check for complex expressions in SELECT
        complex_expressions = self._find_complex_expressions(query)
        if complex_expressions:
            recommendations.append(Recommendation(
                severity=Severity.LOW,
                category=Category.PROJECTION,
                title="Complex Expressions in SELECT",
                description=(
                    f"Found {len(complex_expressions)} complex expression(s) in SELECT clause. "
                    "Consider pre-computing these values or using CTEs for readability."
                ),
                confidence=0.6,
                effort=Effort.LOW,
                action_plan=[
                    "Move complex calculations to CTE for reusability",
                    "Consider pre-computing values at ingestion time",
                    "Use simpler expressions where possible"
                ],
                metadata={"expression_count": len(complex_expressions)}
            ))

        # Check for DISTINCT on many columns
        distinct_match = re.search(r'SELECT\s+DISTINCT\s+(.+?)\s+FROM', query,
                                  re.IGNORECASE | re.DOTALL)
        if distinct_match:
            columns = distinct_match.group(1)
            if columns.count(',') > 5 or '*' in columns:
                recommendations.append(Recommendation(
                    severity=Severity.MEDIUM,
                    category=Category.PROJECTION,
                    title="DISTINCT on Many Columns",
                    description=(
                        "Using DISTINCT on many columns or SELECT * is expensive. "
                        "Consider if all columns are needed for uniqueness."
                    ),
                    confidence=0.8,
                    effort=Effort.LOW,
                    action_plan=[
                        "Use only columns needed for uniqueness",
                        "Consider GROUP BY with MIN/MAX for additional columns",
                        "Check if data is already unique at the source"
                    ]
                ))

        return recommendations

    def _has_select_star(self, query: str) -> bool:
        """Check if query uses SELECT *."""
        # Look for SELECT * but not COUNT(*)
        pattern = r'SELECT\s+\*\s+FROM'
        return bool(re.search(pattern, query, re.IGNORECASE))

    def _find_complex_expressions(self, query: str) -> list[str]:
        """Find complex expressions in SELECT clause."""
        complex = []

        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query,
                                re.IGNORECASE | re.DOTALL)
        if not select_match:
            return complex

        select_clause = select_match.group(1)

        # Look for complex patterns
        patterns = [
            r'CASE\s+WHEN',
            r'CAST\s*\(',
            r'SUBSTRING\s*\(',
            r'REGEXP_EXTRACT\s*\(',
            r'JSON_EXTRACT\s*\(',
            r'\b\w+\s*\(\s*\w+\s*\(\s*',  # Nested functions
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, select_clause, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 20)
                end = min(len(select_clause), match.end() + 50)
                complex.append(select_clause[start:end].strip())

        return complex

    def _generate_projection_example(
        self, query: str, table_metadata: dict
    ) -> str:
        """Generate example with specific column selection."""
        if not table_metadata:
            return "-- Replace SELECT * with specific columns:\nSELECT col1, col2, col3 FROM ..."

        # Get first table's columns
        first_table = next(iter(table_metadata.values()))
        columns = first_table.columns[:5]  # First 5 columns as example

        column_list = ", ".join(col["name"] for col in columns)

        # Replace SELECT * with column list
        modified = re.sub(
            r'SELECT\s+\*',
            f'SELECT {column_list}',
            query,
            flags=re.IGNORECASE,
            count=1
        )

        return f"""-- Original query scans all columns
-- Modified query scans only needed columns:
{modified}

-- Add more columns as needed for your use case"""
