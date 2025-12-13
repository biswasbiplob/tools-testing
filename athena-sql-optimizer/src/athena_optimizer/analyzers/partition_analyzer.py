"""Partition usage and optimization analyzer."""

import re
from typing import Any
import sqlparse
from sqlparse.sql import Where, Comparison
from sqlparse.tokens import Keyword, Whitespace

from ..models import Recommendation, Severity, Category, Effort, TableMetadata
from .base import BaseAnalyzer


class PartitionAnalyzer(BaseAnalyzer):
    """Analyzes partition usage and provides optimization recommendations."""

    @property
    def name(self) -> str:
        return "PartitionAnalyzer"

    def analyze(self, context: dict[str, Any]) -> list[Recommendation]:
        """Analyze partition usage in the query."""
        recommendations = []

        query = context.get("query", "")
        table_metadata = context.get("table_metadata", {})
        explain_plan = context.get("parsed_explain_plan")

        if not table_metadata:
            return recommendations

        # Parse query to extract WHERE conditions
        where_filters = self._extract_where_filters(query)

        # Check each table
        for table_name, metadata in table_metadata.items():
            if not metadata.partition_keys:
                # Table is not partitioned
                recommendations.append(Recommendation(
                    severity=Severity.MEDIUM,
                    category=Category.PARTITION,
                    title=f"Table Not Partitioned: {table_name}",
                    description=(
                        f"Table {table_name} is not partitioned. "
                        "For large tables, partitioning can significantly reduce costs."
                    ),
                    confidence=0.7,
                    effort=Effort.HIGH,
                    action_plan=[
                        f"Consider partitioning {table_name} by common filter columns",
                        "Typical partition columns: date, region, category, year/month/day",
                        "Use CREATE TABLE AS SELECT (CTAS) to create partitioned version"
                    ],
                    code_example=self._generate_partition_example(metadata),
                    references=[
                        "https://docs.aws.amazon.com/athena/latest/ug/partitions.html"
                    ],
                    metadata={"table": table_name}
                ))
            else:
                # Table is partitioned - check if filters are used
                missing_filters = self._check_partition_filters(
                    metadata.partition_keys, where_filters, explain_plan
                )

                if missing_filters:
                    # Calculate potential impact
                    potential_savings = 0.80  # Assume 80% reduction with proper partitioning

                    recommendations.append(Recommendation(
                        severity=Severity.CRITICAL,
                        category=Category.PARTITION,
                        title=f"Missing Partition Filters: {table_name}",
                        description=(
                            f"Table {table_name} is partitioned by "
                            f"{', '.join(metadata.partition_keys)}, but your query "
                            f"doesn't filter on: {', '.join(missing_filters)}. "
                            "This causes a full table scan across all partitions."
                        ),
                        savings_percentage=potential_savings * 100,
                        confidence=0.9,
                        effort=Effort.LOW,
                        action_plan=[
                            f"Add WHERE clause filtering on: {', '.join(missing_filters)}",
                            "Use partition column values that match your needs",
                            "Example: WHERE partition_date >= '2024-01-01'"
                        ],
                        code_example=self._generate_filter_example(
                            query, metadata.partition_keys
                        ),
                        references=[
                            "https://docs.aws.amazon.com/athena/latest/ug/partitions.html"
                        ],
                        metadata={
                            "table": table_name,
                            "missing_filters": missing_filters,
                            "available_partitions": metadata.partition_keys
                        }
                    ))

                # Check for LIKE or inequality operators on partition columns
                inefficient_filters = self._check_inefficient_partition_filters(
                    metadata.partition_keys, where_filters
                )

                if inefficient_filters:
                    recommendations.append(Recommendation(
                        severity=Severity.HIGH,
                        category=Category.PARTITION,
                        title=f"Inefficient Partition Filters: {table_name}",
                        description=(
                            f"Partition filters using LIKE or range operators "
                            f"({', '.join(inefficient_filters)}) are less efficient. "
                            "Use equality comparisons or IN clauses when possible."
                        ),
                        confidence=0.8,
                        effort=Effort.LOW,
                        action_plan=[
                            "Replace LIKE with exact matches or IN clause",
                            "Use date_col IN ('2024-01-01', '2024-01-02') instead of ranges",
                            "Consider partitioning at different granularity"
                        ],
                        metadata={
                            "table": table_name,
                            "inefficient_filters": inefficient_filters
                        }
                    ))

        return recommendations

    def _extract_where_filters(self, query: str) -> dict[str, str]:
        """Extract WHERE clause filters from query."""
        filters = {}

        try:
            parsed = sqlparse.parse(query)
            if not parsed:
                return filters

            # Walk through tokens to find WHERE clause
            for statement in parsed:
                where_clause = None
                for token in statement.tokens:
                    if isinstance(token, Where):
                        where_clause = str(token)
                        break

                if where_clause:
                    # Extract column = value patterns
                    patterns = [
                        r'(\w+)\s*=\s*[\'"]([^\'"]+)[\'"]',
                        r'(\w+)\s*=\s*(\d+)',
                        r'(\w+)\s+IN\s*\(',
                        r'(\w+)\s+LIKE\s+',
                        r'(\w+)\s*>\s*',
                        r'(\w+)\s*<\s*',
                        r'(\w+)\s+BETWEEN\s+'
                    ]

                    for pattern in patterns:
                        matches = re.finditer(pattern, where_clause, re.IGNORECASE)
                        for match in matches:
                            column = match.group(1).lower()
                            filters[column] = match.group(0)

        except Exception:
            # Fallback to regex if parsing fails
            where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|$)',
                                   query, re.IGNORECASE | re.DOTALL)
            if where_match:
                where_clause = where_match.group(1)
                matches = re.finditer(r'(\w+)\s*[=<>]|\b(\w+)\s+IN\s+',
                                     where_clause, re.IGNORECASE)
                for match in matches:
                    column = (match.group(1) or match.group(2)).lower()
                    filters[column] = "filtered"

        return filters

    def _check_partition_filters(
        self,
        partition_keys: list[str],
        where_filters: dict[str, str],
        explain_plan
    ) -> list[str]:
        """Check which partition keys are missing filters."""
        missing = []

        for partition_key in partition_keys:
            key_lower = partition_key.lower()

            # Check in WHERE filters
            if key_lower not in where_filters:
                # Also check explain plan for partition filters
                if explain_plan and explain_plan.partition_filters:
                    # Check if this partition is in the explain plan
                    found = any(key_lower in pf.lower()
                              for pf in explain_plan.partition_filters)
                    if not found:
                        missing.append(partition_key)
                else:
                    missing.append(partition_key)

        return missing

    def _check_inefficient_partition_filters(
        self,
        partition_keys: list[str],
        where_filters: dict[str, str]
    ) -> list[str]:
        """Check for inefficient partition filter patterns."""
        inefficient = []

        for partition_key in partition_keys:
            key_lower = partition_key.lower()
            if key_lower in where_filters:
                filter_expr = where_filters[key_lower].lower()

                # Check for LIKE, <, >, BETWEEN
                if any(op in filter_expr for op in ['like', '<', '>', 'between']):
                    inefficient.append(partition_key)

        return inefficient

    def _generate_partition_example(self, metadata: TableMetadata) -> str:
        """Generate example for partitioning a table."""
        # Suggest common partition columns
        date_columns = [col for col in metadata.columns
                       if 'date' in col['name'].lower() or 'time' in col['name'].lower()]

        if date_columns:
            partition_col = date_columns[0]['name']
            return f"""-- Create partitioned version of table
CREATE TABLE {metadata.table}_partitioned
WITH (
    partitioned_by = ARRAY['{partition_col}'],
    format = 'PARQUET'
)
AS SELECT * FROM {metadata.table};"""
        else:
            return f"""-- Create partitioned version of table
CREATE TABLE {metadata.table}_partitioned
WITH (
    partitioned_by = ARRAY['year', 'month'],
    format = 'PARQUET'
)
AS SELECT *, year(date_column) as year, month(date_column) as month
FROM {metadata.table};"""

    def _generate_filter_example(self, query: str, partition_keys: list[str]) -> str:
        """Generate example with partition filters added."""
        # Simple example - add WHERE clause if missing
        if "where" in query.lower():
            return f"""-- Add partition filters to existing WHERE clause:
-- AND {partition_keys[0]} = 'value'
-- Example:
{query.split('WHERE')[0]}WHERE {partition_keys[0]} = '2024-01-01'
  AND ... (your existing filters)"""
        else:
            from_idx = query.lower().find('from')
            if from_idx > 0:
                # Find end of FROM clause
                next_keyword = re.search(r'\b(GROUP BY|ORDER BY|LIMIT|;|$)',
                                        query[from_idx:], re.IGNORECASE)
                if next_keyword:
                    insert_pos = from_idx + next_keyword.start()
                    return (
                        f"{query[:insert_pos]}\n"
                        f"WHERE {partition_keys[0]} = 'value'\n"
                        f"{query[insert_pos:]}"
                    )

        return f"-- Add: WHERE {partition_keys[0]} = 'value'"
