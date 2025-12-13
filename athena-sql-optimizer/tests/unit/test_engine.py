"""Tests for the optimization engine."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from athena_optimizer.engine import OptimizationEngine
from athena_optimizer.sql_parser import extract_table_names
from athena_optimizer.models import (
    OptimizerConfig,
    AnalysisResult,
    TableMetadata,
    QueryMetrics,
)


class TestOptimizationEngine:
    """Tests for OptimizationEngine."""

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_initialization(self, mock_athena, mock_glue, optimizer_config):
        """Test engine initialization."""
        engine = OptimizationEngine(optimizer_config)

        assert engine.config == optimizer_config
        assert len(engine.analyzers) == 6  # All 6 analyzers
        mock_athena.assert_called_once_with(optimizer_config)
        mock_glue.assert_called_once_with(optimizer_config)

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_analyze_query_basic(self, mock_athena_cls, mock_glue_cls, optimizer_config):
        """Test basic query analysis."""
        # Setup mocks
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        mock_athena.get_explain_plan.return_value = "Fragment 0 [SINGLE]"
        mock_glue.get_table_metadata.return_value = TableMetadata(
            database="test_db",
            table="test_table",
            partition_keys=[],
            columns=[]
        )

        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query("SELECT * FROM test_table")

        assert isinstance(result, AnalysisResult)
        assert result.query == "SELECT * FROM test_table"
        assert isinstance(result.recommendations, list)
        assert result.analysis_timestamp

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_analyze_query_with_explain_analyze(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test query analysis with EXPLAIN ANALYZE."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        mock_athena.get_explain_plan.return_value = "Fragment 0"
        mock_athena.get_explain_analyze_plan.return_value = (
            "Analyze plan",
            QueryMetrics(data_scanned_bytes=1099511627776)
        )
        mock_glue.get_table_metadata.return_value = TableMetadata(
            database="test_db",
            table="test_table",
            partition_keys=[],
            columns=[]
        )

        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query(
            "SELECT * FROM test_table",
            run_explain_analyze=True
        )

        assert result.query_metrics is not None
        assert result.query_metrics.data_scanned_bytes == 1099511627776
        assert result.total_current_cost_usd > 0

    def test_extract_table_names(self):
        """Test table name extraction from queries."""
        # Simple FROM
        tables = extract_table_names("SELECT * FROM test_table")
        assert "test_table" in tables

        # Multiple tables with JOINs
        tables = extract_table_names(
            "SELECT * FROM table1 JOIN table2 ON table1.id = table2.id"
        )
        assert "table1" in tables
        assert "table2" in tables

        # Database.table notation
        tables = extract_table_names("SELECT * FROM db.table")
        assert "db.table" in tables

        # With comments
        tables = extract_table_names(
            "-- Comment\nSELECT * FROM test_table /* inline comment */"
        )
        assert "test_table" in tables

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_estimate_cost(self, mock_athena_cls, mock_glue_cls, optimizer_config):
        """Test cost estimation."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        # Mock table statistics
        mock_glue.get_table_statistics.return_value = {
            "total_size": "1099511627776"  # 1 TB
        }

        engine = OptimizationEngine(optimizer_config)
        result = engine.estimate_cost("SELECT * FROM test_table")

        assert "estimated_cost_usd" in result
        assert "estimated_scan_tb" in result
        assert "cost_range_usd" in result
        assert result["estimated_cost_usd"] > 0

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_check_table_health(self, mock_athena_cls, mock_glue_cls, optimizer_config):
        """Test table health check."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        # Mock table metadata
        mock_glue.get_table_metadata.return_value = TableMetadata(
            database="test_db",
            table="test_table",
            input_format="org.apache.hadoop.mapred.TextInputFormat",
            partition_keys=[],
            columns=[
                {"name": "col1", "type": "string"},
                {"name": "col2", "type": "int"},
            ]
        )

        mock_glue.get_table_statistics.return_value = {
            "num_rows": "1000",
            "total_size": "1024"
        }

        mock_glue.get_partitions.return_value = []

        engine = OptimizationEngine(optimizer_config)
        result = engine.check_table_health("test_db", "test_table")

        assert result["database"] == "test_db"
        assert result["table"] == "test_table"
        assert "format" in result
        assert "columns" in result
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0  # CSV format should trigger recommendations

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_sort_recommendations(self, mock_athena_cls, mock_glue_cls, optimizer_config):
        """Test recommendation sorting."""
        from athena_optimizer.models import Recommendation, Severity, Category, Effort

        engine = OptimizationEngine(optimizer_config)

        recommendations = [
            Recommendation(
                severity=Severity.LOW,
                category=Category.FORMAT,
                title="Low priority",
                description="Test",
                confidence=0.5
            ),
            Recommendation(
                severity=Severity.CRITICAL,
                category=Category.PARTITION,
                title="Critical issue",
                description="Test",
                confidence=0.9
            ),
            Recommendation(
                severity=Severity.HIGH,
                category=Category.COST,
                title="High priority",
                description="Test",
                confidence=0.8
            ),
            Recommendation(
                severity=Severity.CRITICAL,
                category=Category.JOIN,
                title="Another critical",
                description="Test",
                confidence=0.7
            ),
        ]

        sorted_recs = engine._sort_recommendations(recommendations)

        # CRITICAL should be first
        assert sorted_recs[0].severity == Severity.CRITICAL
        assert sorted_recs[1].severity == Severity.CRITICAL

        # Within same severity, higher confidence first
        if sorted_recs[0].confidence != sorted_recs[1].confidence:
            assert sorted_recs[0].confidence > sorted_recs[1].confidence

        # HIGH should come after CRITICAL
        assert sorted_recs[2].severity == Severity.HIGH

        # LOW should be last
        assert sorted_recs[3].severity == Severity.LOW

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_analyze_query_handles_missing_metadata(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test that analysis continues even if table metadata is unavailable."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        mock_athena.get_explain_plan.return_value = "Fragment 0"
        mock_glue.get_table_metadata.side_effect = Exception("Table not found")

        engine = OptimizationEngine(optimizer_config)

        # Should not raise exception
        result = engine.analyze_query("SELECT * FROM missing_table")

        assert isinstance(result, AnalysisResult)
        # Should still have some recommendations (at least from explain analyzer)

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_analyze_query_graceful_explain_failure(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test graceful handling of EXPLAIN plan failures."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        mock_athena.get_explain_plan.side_effect = Exception("EXPLAIN failed")
        mock_glue.get_table_metadata.return_value = TableMetadata(
            database="test_db",
            table="test_table",
            partition_keys=[],
            columns=[]
        )

        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query("SELECT * FROM test_table")

        assert isinstance(result, AnalysisResult)
        assert result.explain_plan is None
        # Should still have recommendations from other analyzers

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_cost_calculation_from_metrics(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test cost calculation from query metrics."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        # 1 TB = 1024^4 bytes
        data_scanned = 1024 ** 4

        mock_athena.get_explain_plan.return_value = "Fragment 0"
        mock_athena.get_explain_analyze_plan.return_value = (
            "Plan",
            QueryMetrics(data_scanned_bytes=data_scanned)
        )
        mock_glue.get_table_metadata.return_value = TableMetadata(
            database="test_db",
            table="test_table",
            partition_keys=[],
            columns=[]
        )

        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query(
            "SELECT * FROM test_table",
            run_explain_analyze=True
        )

        # Cost should be $5 per TB (default)
        expected_cost = 1.0 * 5.0  # 1 TB * $5/TB
        assert abs(result.total_current_cost_usd - expected_cost) < 0.01
