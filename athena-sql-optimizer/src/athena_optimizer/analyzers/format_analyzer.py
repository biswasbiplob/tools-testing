"""Table format and compression analyzer."""

from typing import Any

from ..models import Recommendation, Severity, Category, Effort, TableMetadata
from .base import BaseAnalyzer


class FormatAnalyzer(BaseAnalyzer):
    """Analyzes table storage formats and compression."""

    @property
    def name(self) -> str:
        return "FormatAnalyzer"

    def analyze(self, context: dict[str, Any]) -> list[Recommendation]:
        """Analyze table formats for optimization opportunities."""
        recommendations = []

        table_metadata = context.get("table_metadata", {})

        for table_name, metadata in table_metadata.items():
            format_type = self._detect_format(metadata)

            # Check for non-columnar formats
            if format_type in ["CSV", "TEXT", "JSON"]:
                # Estimate potential savings
                potential_savings = 0.70  # 70% reduction typical for columnar

                recommendations.append(Recommendation(
                    severity=Severity.HIGH,
                    category=Category.FORMAT,
                    title=f"Non-Columnar Format: {table_name}",
                    description=(
                        f"Table {table_name} uses {format_type} format. "
                        "Columnar formats (Parquet, ORC) can reduce costs by 70%+ "
                        "for analytical queries."
                    ),
                    savings_percentage=potential_savings * 100,
                    confidence=0.9,
                    effort=Effort.MEDIUM,
                    action_plan=[
                        f"Convert {table_name} to Parquet or ORC format",
                        "Use CTAS (CREATE TABLE AS SELECT) for conversion",
                        "Enable compression (SNAPPY for Parquet, ZLIB for ORC)",
                        "Test queries after conversion to verify performance"
                    ],
                    code_example=self._generate_format_conversion(metadata, "PARQUET"),
                    references=[
                        "https://docs.aws.amazon.com/athena/latest/ug/columnar-storage.html"
                    ],
                    metadata={
                        "table": table_name,
                        "current_format": format_type,
                        "recommended_format": "PARQUET"
                    }
                ))

            # Check compression
            if not metadata.compressed and format_type not in ["PARQUET", "ORC"]:
                recommendations.append(Recommendation(
                    severity=Severity.MEDIUM,
                    category=Category.COMPRESSION,
                    title=f"Uncompressed Data: {table_name}",
                    description=(
                        f"Table {table_name} appears to be uncompressed. "
                        "Compression reduces storage costs and scan times."
                    ),
                    savings_percentage=50.0,
                    confidence=0.7,
                    effort=Effort.MEDIUM,
                    action_plan=[
                        "Enable compression when converting format",
                        "Use GZIP for text-based formats",
                        "Use SNAPPY for Parquet (fast, good compression)",
                        "Use ZLIB for ORC (better compression, slightly slower)"
                    ],
                    metadata={"table": table_name}
                ))

            # Check for PARQUET/ORC optimization
            if format_type in ["PARQUET", "ORC"]:
                recommendations.append(Recommendation(
                    severity=Severity.INFO,
                    category=Category.FORMAT,
                    title=f"Optimal Format: {table_name}",
                    description=(
                        f"Table {table_name} uses {format_type} format, which is optimal "
                        "for Athena queries. Consider column ordering for better performance."
                    ),
                    confidence=0.8,
                    effort=Effort.LOW,
                    action_plan=[
                        "Order frequently filtered columns first in table definition",
                        "Place frequently selected columns together",
                        "Consider partition projection for time-series data"
                    ],
                    references=[
                        "https://docs.aws.amazon.com/athena/latest/ug/columnar-storage.html"
                    ],
                    metadata={
                        "table": table_name,
                        "format": format_type
                    }
                ))

        return recommendations

    def _detect_format(self, metadata: TableMetadata) -> str:
        """Detect table format from metadata."""
        input_format = metadata.input_format or ""

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

    def _generate_format_conversion(
        self, metadata: TableMetadata, target_format: str
    ) -> str:
        """Generate SQL for format conversion."""
        partition_clause = ""
        if metadata.partition_keys:
            partition_clause = f"\n    partitioned_by = ARRAY{metadata.partition_keys},"

        return f"""-- Convert to {target_format} format
CREATE TABLE {metadata.table}_{target_format.lower()}
WITH ({partition_clause}
    format = '{target_format}',
    parquet_compression = 'SNAPPY'
)
AS SELECT * FROM {metadata.table};

-- Verify row counts match
SELECT
    (SELECT COUNT(*) FROM {metadata.table}) as original_count,
    (SELECT COUNT(*) FROM {metadata.table}_{target_format.lower()}) as new_count;

-- Once verified, you can:
-- 1. Drop old table: DROP TABLE {metadata.table};
-- 2. Rename new table: ALTER TABLE {metadata.table}_{target_format.lower()} RENAME TO {metadata.table};"""
