"""Constants organized by domain for better discoverability and maintainability."""

from typing import Final


# ============================================================================
# Domain-Organized Constants
# ============================================================================


class ByteConversions:
    """Byte conversion constants for data size calculations."""

    BYTES_PER_TB: Final[int] = 1024 ** 4
    BYTES_PER_GB: Final[int] = 1024 ** 3
    BYTES_PER_MB: Final[int] = 1024 ** 2


class CostThresholds:
    """Cost threshold constants (USD) for query cost analysis."""

    HIGH_USD: Final[float] = 1.0
    MEDIUM_USD: Final[float] = 0.1


class ConfidenceLevels:
    """Confidence level constants for recommendation scoring."""

    VERY_HIGH: Final[float] = 1.0
    HIGH: Final[float] = 0.9
    MEDIUM_HIGH: Final[float] = 0.8
    MEDIUM: Final[float] = 0.7
    LOW: Final[float] = 0.6


class SavingsRates:
    """Expected savings percentages for various optimizations."""

    PARTITION_FILTER: Final[float] = 0.80  # 80% reduction
    COLUMNAR_FORMAT: Final[float] = 0.70  # 70% reduction
    COLUMN_PROJECTION: Final[float] = 0.50  # 50% reduction
    COMPRESSION: Final[float] = 0.50  # 50% reduction


class CostEstimation:
    """Cost estimation multipliers for range calculations."""

    MIN_COST_MULTIPLIER: Final[float] = 0.1  # Best case
    MAX_COST_MULTIPLIER: Final[float] = 1.0  # Worst case


class QueryLimits:
    """Query and partition limits for AWS API calls."""

    DEFAULT_MAX_PARTITIONS: Final[int] = 100
    MAX_PARTITIONS_FOR_HEALTH_CHECK: Final[int] = 100
    GLUE_MAX_PARTITIONS_DEFAULT: Final[int] = 1000


class TextAnalysis:
    """Text analysis context window sizes for query parsing."""

    CONTEXT_BEFORE: Final[int] = 50
    CONTEXT_AFTER: Final[int] = 50
    COMPLEX_EXPRESSION_BEFORE: Final[int] = 20
    COMPLEX_EXPRESSION_AFTER: Final[int] = 50


class SeverityOrder:
    """Severity ordering constants for recommendation sorting."""

    CRITICAL: Final[int] = 0
    HIGH: Final[int] = 1
    MEDIUM: Final[int] = 2
    LOW: Final[int] = 3
    INFO: Final[int] = 4
    DEFAULT: Final[int] = 999


class Percentages:
    """Percentage calculation constants."""

    MULTIPLIER: Final[int] = 100


# ============================================================================
# Backward Compatibility Exports
# Maintain old flat structure for existing code
# ============================================================================

# Byte conversions
BYTES_PER_TB = ByteConversions.BYTES_PER_TB
BYTES_PER_GB = ByteConversions.BYTES_PER_GB
BYTES_PER_MB = ByteConversions.BYTES_PER_MB

# Cost thresholds
HIGH_COST_THRESHOLD_USD = CostThresholds.HIGH_USD
MEDIUM_COST_THRESHOLD_USD = CostThresholds.MEDIUM_USD

# Confidence levels
CONFIDENCE_VERY_HIGH = ConfidenceLevels.VERY_HIGH
CONFIDENCE_HIGH = ConfidenceLevels.HIGH
CONFIDENCE_MEDIUM_HIGH = ConfidenceLevels.MEDIUM_HIGH
CONFIDENCE_MEDIUM = ConfidenceLevels.MEDIUM
CONFIDENCE_LOW = ConfidenceLevels.LOW

# Savings rates
SAVINGS_PARTITION_FILTER = SavingsRates.PARTITION_FILTER
SAVINGS_COLUMNAR_FORMAT = SavingsRates.COLUMNAR_FORMAT
SAVINGS_COLUMN_PROJECTION = SavingsRates.COLUMN_PROJECTION
SAVINGS_COMPRESSION = SavingsRates.COMPRESSION

# Cost estimation
MIN_COST_MULTIPLIER = CostEstimation.MIN_COST_MULTIPLIER
MAX_COST_MULTIPLIER = CostEstimation.MAX_COST_MULTIPLIER

# Query limits
DEFAULT_MAX_PARTITIONS = QueryLimits.DEFAULT_MAX_PARTITIONS
MAX_PARTITIONS_FOR_HEALTH_CHECK = QueryLimits.MAX_PARTITIONS_FOR_HEALTH_CHECK
GLUE_MAX_PARTITIONS_DEFAULT = QueryLimits.GLUE_MAX_PARTITIONS_DEFAULT

# Text analysis
TEXT_CONTEXT_BEFORE = TextAnalysis.CONTEXT_BEFORE
TEXT_CONTEXT_AFTER = TextAnalysis.CONTEXT_AFTER
COMPLEX_EXPRESSION_CONTEXT_BEFORE = TextAnalysis.COMPLEX_EXPRESSION_BEFORE
COMPLEX_EXPRESSION_CONTEXT_AFTER = TextAnalysis.COMPLEX_EXPRESSION_AFTER

# Severity order
SEVERITY_ORDER_CRITICAL = SeverityOrder.CRITICAL
SEVERITY_ORDER_HIGH = SeverityOrder.HIGH
SEVERITY_ORDER_MEDIUM = SeverityOrder.MEDIUM
SEVERITY_ORDER_LOW = SeverityOrder.LOW
SEVERITY_ORDER_INFO = SeverityOrder.INFO
SEVERITY_ORDER_DEFAULT = SeverityOrder.DEFAULT

# Percentages
PERCENTAGE_MULTIPLIER = Percentages.MULTIPLIER
