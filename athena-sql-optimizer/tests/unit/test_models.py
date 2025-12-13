"""Tests for data models."""

import pytest
from pydantic import ValidationError
from athena_optimizer.models import (
    OptimizerConfig,
    Recommendation,
    Severity,
    Category,
    Effort,
    TableMetadata,
    QueryMetrics,
    ExplainPlan,
    AnalysisResult,
)


class TestOptimizerConfig:
    """Tests for OptimizerConfig model."""

    def test_valid_config(self):
        """Test creating a valid configuration."""
        config = OptimizerConfig(
            workgroup="test",
            s3_output_location="s3://bucket/path/"
        )
        assert config.workgroup == "test"
        assert config.region == "eu-west-1"  # Default
        assert config.catalog == "AwsDataCatalog"  # Default

    def test_missing_required_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError):
            OptimizerConfig()  # Missing workgroup and s3_output_location

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = OptimizerConfig(
            aws_profile="custom",
            region="us-west-2",
            workgroup="custom-wg",
            s3_output_location="s3://custom/",
            database="custom_db",
            run_explain_analyze=True,
            athena_cost_per_tb=10.0,
            timeout_seconds=600
        )
        assert config.aws_profile == "custom"
        assert config.region == "us-west-2"
        assert config.run_explain_analyze is True
        assert config.athena_cost_per_tb == 10.0


class TestRecommendation:
    """Tests for Recommendation model."""

    def test_valid_recommendation(self):
        """Test creating a valid recommendation."""
        rec = Recommendation(
            severity=Severity.HIGH,
            category=Category.PARTITION,
            title="Test Recommendation",
            description="Test description",
            confidence=0.9,
            effort=Effort.LOW
        )
        assert rec.severity == Severity.HIGH
        assert rec.category == Category.PARTITION
        assert rec.confidence == 0.9

    def test_recommendation_with_costs(self):
        """Test recommendation with cost information."""
        rec = Recommendation(
            severity=Severity.HIGH,
            category=Category.COST,
            title="High Cost",
            description="Query is expensive",
            current_cost_usd=10.0,
            optimized_cost_usd=2.0,
            savings_usd=8.0,
            savings_percentage=80.0
        )
        assert rec.current_cost_usd == 10.0
        assert rec.savings_usd == 8.0
        assert rec.savings_percentage == 80.0

    def test_confidence_validation(self):
        """Test that confidence is validated between 0 and 1."""
        with pytest.raises(ValidationError):
            Recommendation(
                severity=Severity.HIGH,
                category=Category.PARTITION,
                title="Test",
                description="Test",
                confidence=1.5  # Invalid: > 1.0
            )

    def test_recommendation_defaults(self):
        """Test default values."""
        rec = Recommendation(
            severity=Severity.INFO,
            category=Category.FORMAT,
            title="Test",
            description="Test"
        )
        assert rec.confidence == 0.8  # Default
        assert rec.effort == Effort.MEDIUM  # Default
        assert rec.action_plan == []
        assert rec.code_example is None


class TestTableMetadata:
    """Tests for TableMetadata model."""

    def test_valid_table_metadata(self):
        """Test creating valid table metadata."""
        metadata = TableMetadata(
            database="test_db",
            table="test_table",
            location="s3://bucket/data/",
            partition_keys=["year", "month"],
            columns=[{"name": "id", "type": "bigint"}]
        )
        assert metadata.database == "test_db"
        assert len(metadata.partition_keys) == 2
        assert len(metadata.columns) == 1

    def test_empty_partition_keys(self):
        """Test table without partitions."""
        metadata = TableMetadata(
            database="test_db",
            table="test_table"
        )
        assert metadata.partition_keys == []
        assert metadata.columns == []


class TestQueryMetrics:
    """Tests for QueryMetrics model."""

    def test_query_metrics(self):
        """Test creating query metrics."""
        metrics = QueryMetrics(
            data_scanned_bytes=1099511627776,  # 1 TB
            execution_time_ms=5000
        )
        assert metrics.data_scanned_bytes == 1099511627776
        assert metrics.execution_time_ms == 5000

    def test_query_metrics_defaults(self):
        """Test default values."""
        metrics = QueryMetrics()
        assert metrics.data_scanned_bytes == 0
        assert metrics.execution_time_ms == 0


class TestExplainPlan:
    """Tests for ExplainPlan model."""

    def test_explain_plan(self):
        """Test creating explain plan."""
        plan = ExplainPlan(
            raw_plan="Fragment 0 [SINGLE]...",
            distributed=True,
            fragment_count=2,
            table_scans=["table1", "table2"]
        )
        assert plan.distributed is True
        assert plan.fragment_count == 2
        assert len(plan.table_scans) == 2


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_analysis_result(self, optimizer_config):
        """Test creating analysis result."""
        result = AnalysisResult(
            query="SELECT * FROM test",
            recommendations=[],
            total_current_cost_usd=10.0,
            total_optimized_cost_usd=2.0,
            total_savings_usd=8.0,
            total_savings_percentage=80.0,
            analysis_timestamp="2024-01-01T00:00:00"
        )
        assert result.query == "SELECT * FROM test"
        assert result.total_savings_percentage == 80.0
        assert len(result.recommendations) == 0

    def test_analysis_result_with_recommendations(self):
        """Test analysis result with recommendations."""
        rec = Recommendation(
            severity=Severity.HIGH,
            category=Category.PARTITION,
            title="Test",
            description="Test"
        )
        result = AnalysisResult(
            query="SELECT * FROM test",
            recommendations=[rec],
            total_current_cost_usd=0.0,
            total_optimized_cost_usd=0.0,
            total_savings_usd=0.0,
            total_savings_percentage=0.0,
            analysis_timestamp="2024-01-01T00:00:00"
        )
        assert len(result.recommendations) == 1
        assert result.recommendations[0].severity == Severity.HIGH
