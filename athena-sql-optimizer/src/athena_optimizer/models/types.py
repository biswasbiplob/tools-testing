"""Data models for Athena SQL Optimizer."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Recommendation severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Category(str, Enum):
    """Recommendation categories."""
    PARTITION = "PARTITION"
    FORMAT = "FORMAT"
    JOIN = "JOIN"
    PROJECTION = "PROJECTION"
    COST = "COST"
    QUERY_PATTERN = "QUERY_PATTERN"
    STATISTICS = "STATISTICS"
    COMPRESSION = "COMPRESSION"


class Effort(str, Enum):
    """Implementation effort levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Recommendation(BaseModel):
    """A single optimization recommendation."""
    severity: Severity
    category: Category
    title: str
    description: str
    current_cost_usd: Optional[float] = None
    optimized_cost_usd: Optional[float] = None
    savings_usd: Optional[float] = None
    savings_percentage: Optional[float] = None
    data_scanned_current_tb: Optional[float] = None
    data_scanned_optimized_tb: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    effort: Effort = Effort.MEDIUM
    action_plan: list[str] = Field(default_factory=list)
    code_example: Optional[str] = None
    references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableMetadata(BaseModel):
    """Metadata about an Athena table."""
    database: str
    table: str
    location: Optional[str] = None
    input_format: Optional[str] = None
    output_format: Optional[str] = None
    serde_info: Optional[dict[str, Any]] = None
    partition_keys: list[str] = Field(default_factory=list)
    columns: list[dict[str, str]] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    table_type: Optional[str] = None
    compressed: Optional[bool] = None
    num_buckets: Optional[int] = None
    storage_descriptor: Optional[dict[str, Any]] = None


class QueryMetrics(BaseModel):
    """Metrics from query execution."""
    data_scanned_bytes: int = 0
    execution_time_ms: int = 0
    query_queue_time_ms: int = 0
    query_planning_time_ms: int = 0
    service_processing_time_ms: int = 0
    output_location: Optional[str] = None
    statistics: dict[str, Any] = Field(default_factory=dict)


class ExplainPlan(BaseModel):
    """Query execution plan from EXPLAIN."""
    raw_plan: str
    distributed: bool = False
    fragment_count: int = 0
    operations: list[str] = Field(default_factory=list)
    table_scans: list[str] = Field(default_factory=list)
    partition_filters: list[str] = Field(default_factory=list)
    estimated_cost: Optional[float] = None


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    query: str
    recommendations: list[Recommendation]
    explain_plan: Optional[ExplainPlan] = None
    query_metrics: Optional[QueryMetrics] = None
    table_metadata: dict[str, TableMetadata] = Field(default_factory=dict)
    total_current_cost_usd: float = 0.0
    total_optimized_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    total_savings_percentage: float = 0.0
    analysis_timestamp: str
    config: dict[str, Any] = Field(default_factory=dict)


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer."""
    aws_profile: Optional[str] = None
    region: str = "eu-west-1"
    workgroup: str
    s3_output_location: str
    catalog: str = "AwsDataCatalog"
    database: Optional[str] = None
    run_explain_analyze: bool = False
    athena_cost_per_tb: float = 5.0  # USD per TB scanned
    timeout_seconds: int = 300
