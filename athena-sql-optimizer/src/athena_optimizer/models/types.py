"""Data models for Athena SQL Optimizer."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


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
    title: str = Field(min_length=1, description="Recommendation title")
    description: str = Field(min_length=1, description="Detailed description")
    current_cost_usd: Optional[float] = Field(default=None, ge=0.0, description="Current cost in USD")
    optimized_cost_usd: Optional[float] = Field(default=None, ge=0.0, description="Optimized cost in USD")
    savings_usd: Optional[float] = Field(default=None, ge=0.0, description="Savings in USD")
    savings_percentage: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Savings percentage")
    data_scanned_current_tb: Optional[float] = Field(default=None, ge=0.0, description="Current data scanned in TB")
    data_scanned_optimized_tb: Optional[float] = Field(default=None, ge=0.0, description="Optimized data scanned in TB")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Confidence level")
    effort: Effort = Effort.MEDIUM
    action_plan: list[str] = Field(default_factory=list)
    code_example: Optional[str] = None
    references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_cost_consistency(self) -> 'Recommendation':
        """Validate cost and savings values are consistent."""
        if self.current_cost_usd is not None and self.optimized_cost_usd is not None:
            if self.optimized_cost_usd > self.current_cost_usd:
                raise ValueError("optimized_cost_usd cannot be greater than current_cost_usd")

            # Validate savings if provided
            expected_savings = self.current_cost_usd - self.optimized_cost_usd
            if self.savings_usd is not None and abs(self.savings_usd - expected_savings) > 0.01:
                raise ValueError(f"savings_usd {self.savings_usd} does not match calculated savings {expected_savings}")

        # Validate data scanned consistency
        if self.data_scanned_current_tb is not None and self.data_scanned_optimized_tb is not None:
            if self.data_scanned_optimized_tb > self.data_scanned_current_tb:
                raise ValueError("data_scanned_optimized_tb cannot be greater than data_scanned_current_tb")

        return self


class TableMetadata(BaseModel):
    """Metadata about an Athena table."""
    database: str = Field(min_length=1, description="Database name")
    table: str = Field(min_length=1, description="Table name")
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
    data_scanned_bytes: int = Field(default=0, ge=0, description="Data scanned in bytes")
    execution_time_ms: int = Field(default=0, ge=0, description="Execution time in milliseconds")
    query_queue_time_ms: int = Field(default=0, ge=0, description="Queue time in milliseconds")
    query_planning_time_ms: int = Field(default=0, ge=0, description="Planning time in milliseconds")
    service_processing_time_ms: int = Field(default=0, ge=0, description="Service processing time in milliseconds")
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
    athena_cost_per_tb: float = Field(default=5.0, gt=0.0, description="USD per TB scanned")
    timeout_seconds: int = Field(default=300, gt=0, le=3600, description="Query timeout in seconds")

    @field_validator("workgroup")
    @classmethod
    def validate_workgroup(cls, v: str) -> str:
        """Validate workgroup is not empty."""
        if not v or not v.strip():
            raise ValueError("workgroup cannot be empty")
        return v.strip()

    @field_validator("s3_output_location")
    @classmethod
    def validate_s3_output_location(cls, v: str) -> str:
        """Validate S3 output location format."""
        if not v or not v.strip():
            raise ValueError("s3_output_location cannot be empty")
        v = v.strip()
        if not v.startswith("s3://"):
            raise ValueError("s3_output_location must start with 's3://'")
        if len(v) < 6:  # s3:// plus at least one character
            raise ValueError("s3_output_location must specify a bucket")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate AWS region format."""
        if not v or not v.strip():
            raise ValueError("region cannot be empty")
        v = v.strip()
        # Basic validation for AWS region format (e.g., us-east-1, eu-west-1)
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("region must contain only alphanumeric characters, hyphens, and underscores")
        return v

    @field_validator("catalog")
    @classmethod
    def validate_catalog(cls, v: str) -> str:
        """Validate catalog is not empty."""
        if not v or not v.strip():
            raise ValueError("catalog cannot be empty")
        return v.strip()

    @field_validator("database")
    @classmethod
    def validate_database(cls, v: Optional[str]) -> Optional[str]:
        """Validate database name if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
            # Database names should be alphanumeric with underscores
            if not v.replace("_", "").replace("-", "").isalnum():
                raise ValueError("database name must contain only alphanumeric characters, hyphens, and underscores")
        return v
