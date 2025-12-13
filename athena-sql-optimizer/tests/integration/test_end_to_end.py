"""End-to-end integration tests."""

import pytest
from unittest.mock import patch, Mock
import json

from athena_optimizer.engine import OptimizationEngine
from athena_optimizer.models import OptimizerConfig, TableMetadata, QueryMetrics


class TestEndToEndAnalysis:
    """End-to-end tests for complete analysis workflows."""

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_select_star_csv_table_analysis(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test complete analysis of SELECT * on CSV table."""
        # Setup mocks
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        # Mock EXPLAIN plan
        mock_athena.get_explain_plan.return_value = """
        Fragment 0 [SINGLE]
        - TableScan[awsdatacatalog:test_db:csv_table]
        """

        # Mock EXPLAIN ANALYZE with metrics
        mock_athena.get_explain_analyze_plan.return_value = (
            "Analyze plan",
            QueryMetrics(data_scanned_bytes=5 * 1024**4)  # 5 TB
        )

        # Mock CSV table metadata (no partitions)
        mock_glue.get_table_metadata.return_value = TableMetadata(
            database="test_db",
            table="csv_table",
            location="s3://bucket/csv-data/",
            input_format="org.apache.hadoop.mapred.TextInputFormat",
            partition_keys=[],
            columns=[
                {"name": "col1", "type": "string"},
                {"name": "col2", "type": "int"},
                {"name": "col3", "type": "string"},
                {"name": "col4", "type": "bigint"},
            ],
            compressed=False
        )

        # Run analysis
        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query(
            "SELECT * FROM csv_table",
            run_explain_analyze=True
        )

        # Verify results
        assert result.query == "SELECT * FROM csv_table"
        assert len(result.recommendations) > 0

        # Should have SELECT * recommendation
        select_star_recs = [r for r in result.recommendations if "SELECT *" in r.title]
        assert len(select_star_recs) > 0

        # Should have format recommendation (CSV -> Parquet)
        format_recs = [r for r in result.recommendations if "Format" in r.title]
        assert len(format_recs) > 0

        # Should have unpartitioned table recommendation
        partition_recs = [r for r in result.recommendations if "Partition" in r.title]
        assert len(partition_recs) > 0

        # Should have high cost warning (5 TB)
        assert result.total_current_cost_usd > 20.0  # 5 TB * $5/TB = $25

        # Should have potential savings
        assert result.total_savings_percentage > 0

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_missing_partition_filter_analysis(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test analysis of query missing partition filters."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        mock_athena.get_explain_plan.return_value = """
        Fragment 0 [SINGLE]
        - TableScan[awsdatacatalog:test_db:partitioned_table]
        """

        # Partitioned table metadata
        mock_glue.get_table_metadata.return_value = TableMetadata(
            database="test_db",
            table="partitioned_table",
            location="s3://bucket/data/",
            input_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            partition_keys=["year", "month", "day"],
            columns=[
                {"name": "id", "type": "bigint"},
                {"name": "value", "type": "decimal(10,2)"},
            ],
            compressed=True
        )

        # Query without partition filters
        query = "SELECT id, value FROM partitioned_table WHERE id > 1000"

        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query(query)

        # Should have CRITICAL missing partition filter warning
        critical_recs = [r for r in result.recommendations if r.severity.value == "CRITICAL"]
        assert len(critical_recs) > 0

        missing_filter_recs = [
            r for r in critical_recs if "Missing Partition Filters" in r.title
        ]
        assert len(missing_filter_recs) > 0
        assert "year" in missing_filter_recs[0].description
        assert "month" in missing_filter_recs[0].description

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_cross_join_analysis(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test analysis of CROSS JOIN query."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        mock_athena.get_explain_plan.return_value = """
        Fragment 0 [SINGLE]
        - CROSS JOIN
          - TableScan[table1]
          - TableScan[table2]
        """

        mock_glue.get_table_metadata.side_effect = [
            TableMetadata(
                database="test_db",
                table="table1",
                partition_keys=[],
                columns=[]
            ),
            TableMetadata(
                database="test_db",
                table="table2",
                partition_keys=[],
                columns=[]
            ),
        ]

        query = "SELECT * FROM table1 CROSS JOIN table2"

        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query(query)

        # Should have CRITICAL cross join warning
        critical_recs = [r for r in result.recommendations if r.severity.value == "CRITICAL"]
        assert len(critical_recs) > 0

        cross_join_recs = [r for r in critical_recs if "Cross Join" in r.title]
        assert len(cross_join_recs) > 0

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_optimized_query_analysis(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test analysis of well-optimized query."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        mock_athena.get_explain_plan.return_value = """
        Fragment 0 [SINGLE]
        - TableScan[awsdatacatalog:test_db:optimized_table]
          Partition filter: year = 2024, month = 01
        """

        mock_athena.get_explain_analyze_plan.return_value = (
            "Plan",
            QueryMetrics(data_scanned_bytes=1024**3)  # 1 GB
        )

        # Well-optimized Parquet table with partitions
        mock_glue.get_table_metadata.return_value = TableMetadata(
            database="test_db",
            table="optimized_table",
            input_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            partition_keys=["year", "month"],
            columns=[
                {"name": "id", "type": "bigint"},
                {"name": "name", "type": "string"},
                {"name": "amount", "type": "decimal(10,2)"},
            ],
            compressed=True
        )

        # Well-optimized query
        query = """
        SELECT id, name, amount
        FROM optimized_table
        WHERE year = '2024' AND month = '01'
        LIMIT 1000
        """

        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query(query, run_explain_analyze=True)

        # Should have mostly INFO level recommendations
        critical_high_recs = [
            r for r in result.recommendations
            if r.severity.value in ["CRITICAL", "HIGH"]
        ]

        # Well-optimized query should have few or no critical/high recommendations
        assert len(critical_high_recs) <= 1  # Maybe some JOIN order suggestion

        # Cost should be low
        assert result.total_current_cost_usd < 0.01  # < 1 cent

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_complex_join_query_analysis(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test analysis of complex multi-join query."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        mock_athena.get_explain_plan.return_value = """
        Fragment 0 [SINGLE]
        - Join [LEFT]
          - Join [INNER]
            - Join [INNER]
              - TableScan[table1]
              - TableScan[table2]
            - TableScan[table3]
          - TableScan[table4]
        """

        # Mock multiple table metadata calls
        mock_glue.get_table_metadata.side_effect = [
            TableMetadata(database="test_db", table="table1", partition_keys=[], columns=[]),
            TableMetadata(database="test_db", table="table2", partition_keys=[], columns=[]),
            TableMetadata(database="test_db", table="table3", partition_keys=[], columns=[]),
            TableMetadata(database="test_db", table="table4", partition_keys=[], columns=[]),
        ]

        query = """
        SELECT *
        FROM table1
        JOIN table2 ON table1.id = table2.id
        JOIN table3 ON table2.id = table3.id
        LEFT JOIN table4 ON table3.id = table4.id
        """

        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query(query)

        # Should have multiple recommendations
        assert len(result.recommendations) >= 3

        # Should mention SELECT *
        select_star_recs = [r for r in result.recommendations if "SELECT *" in r.title]
        assert len(select_star_recs) > 0

        # Should mention multiple JOINs
        join_recs = [r for r in result.recommendations if "JOIN" in r.title]
        assert len(join_recs) > 0

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_cost_estimation_without_execution(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test cost estimation without executing query."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        # Mock table statistics
        mock_glue.get_table_statistics.return_value = {
            "total_size": str(10 * 1024**4)  # 10 TB
        }

        engine = OptimizationEngine(optimizer_config)
        result = engine.estimate_cost("SELECT * FROM large_table")

        assert "estimated_cost_usd" in result
        assert result["estimated_cost_usd"] > 0

        # 10 TB * $5/TB = $50
        assert result["estimated_cost_usd"] >= 40  # Allow for some variance
        assert result["estimated_scan_tb"] >= 9

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_table_health_check(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test table health check functionality."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        # CSV table
        mock_glue.get_table_metadata.return_value = TableMetadata(
            database="test_db",
            table="csv_table",
            input_format="org.apache.hadoop.mapred.TextInputFormat",
            partition_keys=[],
            columns=[
                {"name": "col1", "type": "string"},
                {"name": "col2", "type": "int"},
            ],
            compressed=False
        )

        mock_glue.get_table_statistics.return_value = {
            "num_rows": "1000000",
            "total_size": "1073741824"  # 1 GB
        }

        mock_glue.get_partitions.return_value = []

        engine = OptimizationEngine(optimizer_config)
        result = engine.check_table_health("test_db", "csv_table")

        # Should identify issues
        assert len(result["recommendations"]) > 0

        # Should recommend format change
        format_recs = [r for r in result["recommendations"] if r["category"] == "FORMAT"]
        assert len(format_recs) > 0

        # Should recommend partitioning
        partition_recs = [r for r in result["recommendations"] if r["category"] == "PARTITION"]
        assert len(partition_recs) > 0

        # Verify structure
        assert result["database"] == "test_db"
        assert result["table"] == "csv_table"
        assert result["format"] == "CSV"
        assert len(result["columns"]) == 2


class TestErrorHandling:
    """Tests for error handling in integration scenarios."""

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_handles_missing_table_gracefully(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test that missing tables are handled gracefully."""
        mock_athena = Mock()
        mock_glue = Mock()
        mock_athena_cls.return_value = mock_athena
        mock_glue_cls.return_value = mock_glue

        mock_athena.get_explain_plan.return_value = "Fragment 0"
        mock_glue.get_table_metadata.side_effect = Exception("Table not found")

        engine = OptimizationEngine(optimizer_config)
        result = engine.analyze_query("SELECT * FROM missing_table")

        # Should complete without raising
        assert isinstance(result, dict) or hasattr(result, 'recommendations')

    @patch('athena_optimizer.engine.GlueCollector')
    @patch('athena_optimizer.engine.AthenaCollector')
    def test_handles_explain_failure_gracefully(
        self, mock_athena_cls, mock_glue_cls, optimizer_config
    ):
        """Test that EXPLAIN failures are handled gracefully."""
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

        # Should still produce results
        assert result.recommendations is not None
