"""Constants and configuration values for the optimizer."""

from typing import Final


# === Byte Conversion Constants ===
BYTES_PER_TB: Final[int] = 1024 ** 4
BYTES_PER_GB: Final[int] = 1024 ** 3
BYTES_PER_MB: Final[int] = 1024 ** 2


# === Cost Thresholds (USD) ===
HIGH_COST_THRESHOLD_USD: Final[float] = 1.0
MEDIUM_COST_THRESHOLD_USD: Final[float] = 0.1


# === Confidence Levels ===
CONFIDENCE_VERY_HIGH: Final[float] = 1.0
CONFIDENCE_HIGH: Final[float] = 0.9
CONFIDENCE_MEDIUM_HIGH: Final[float] = 0.8
CONFIDENCE_MEDIUM: Final[float] = 0.7
CONFIDENCE_LOW: Final[float] = 0.6


# === Potential Savings Percentages ===
# These represent typical savings achievable with various optimizations
SAVINGS_PARTITION_FILTER: Final[float] = 0.80  # 80% reduction with proper partitioning
SAVINGS_COLUMNAR_FORMAT: Final[float] = 0.70  # 70% reduction with Parquet/ORC
SAVINGS_COLUMN_PROJECTION: Final[float] = 0.50  # 50% reduction selecting specific columns
SAVINGS_COMPRESSION: Final[float] = 0.50  # 50% reduction with compression


# === Cost Estimation Constants ===
MIN_COST_MULTIPLIER: Final[float] = 0.1  # Best case with optimal optimizations
MAX_COST_MULTIPLIER: Final[float] = 1.0  # Worst case full table scan


# === Query Limits ===
DEFAULT_MAX_PARTITIONS: Final[int] = 100
MAX_PARTITIONS_FOR_HEALTH_CHECK: Final[int] = 100
GLUE_MAX_PARTITIONS_DEFAULT: Final[int] = 1000


# === Text Analysis Context Windows ===
TEXT_CONTEXT_BEFORE: Final[int] = 50
TEXT_CONTEXT_AFTER: Final[int] = 50
COMPLEX_EXPRESSION_CONTEXT_BEFORE: Final[int] = 20
COMPLEX_EXPRESSION_CONTEXT_AFTER: Final[int] = 50


# === Severity Constants ===
# Used for sorting recommendations
SEVERITY_ORDER_CRITICAL: Final[int] = 0
SEVERITY_ORDER_HIGH: Final[int] = 1
SEVERITY_ORDER_MEDIUM: Final[int] = 2
SEVERITY_ORDER_LOW: Final[int] = 3
SEVERITY_ORDER_INFO: Final[int] = 4
SEVERITY_ORDER_DEFAULT: Final[int] = 999


# === Percentage Calculations ===
PERCENTAGE_MULTIPLIER: Final[int] = 100
