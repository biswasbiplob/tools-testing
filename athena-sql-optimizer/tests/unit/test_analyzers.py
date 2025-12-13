"""Tests for analyzer modules."""

import pytest
from athena_optimizer.analyzers import (
    CostAnalyzer,
    ExplainAnalyzer,
    FormatAnalyzer,
    JoinAnalyzer,
    PartitionAnalyzer,
    ProjectionAnalyzer,
)
from athena_optimizer.models import (
    Severity,
    Category,
    QueryMetrics,
    TableMetadata,
)


class TestCostAnalyzer:
    """Tests for CostAnalyzer."""

    def test_high_cost_detection(self, optimizer_config):
        """Test detection of high-cost queries."""
        analyzer = CostAnalyzer(optimizer_config)

        # Create metrics for 5 TB scan
        metrics = QueryMetrics(data_scanned_bytes=5 * 1024**4)

        context = {
            "query": "SELECT * FROM large_table",
            "query_metrics": metrics,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        assert len(recommendations) > 0
        assert any(rec.severity == Severity.HIGH for rec in recommendations)
        assert any("High Query Cost" in rec.title for rec in recommendations)

    def test_moderate_cost_detection(self, optimizer_config):
        """Test detection of moderate-cost queries."""
        analyzer = CostAnalyzer(optimizer_config)

        # Create metrics for 0.1 TB scan (costs $0.50 with $5/TB)
        # This is between $0.1 and $1.0, so should be MEDIUM
        metrics = QueryMetrics(data_scanned_bytes=int(0.1 * 1024**4))

        context = {
            "query": "SELECT * FROM table",
            "query_metrics": metrics,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        assert len(recommendations) > 0
        assert any(rec.severity == Severity.MEDIUM for rec in recommendations)

    def test_no_metrics(self, optimizer_config):
        """Test when no metrics are available."""
        analyzer = CostAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM table",
            "query_metrics": None,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)
        assert recommendations == []

    def test_partitioned_table_info(self, optimizer_config, sample_table_metadata):
        """Test information about partitioned tables."""
        analyzer = CostAnalyzer(optimizer_config)

        metrics = QueryMetrics(data_scanned_bytes=1024**4)  # 1 TB

        context = {
            "query": "SELECT * FROM test_table",
            "query_metrics": metrics,
            "table_metadata": {"test_table": sample_table_metadata}
        }

        recommendations = analyzer.analyze(context)

        # Should mention partitions
        partition_recs = [r for r in recommendations if "partition" in r.description.lower()]
        assert len(partition_recs) > 0


class TestExplainAnalyzer:
    """Tests for ExplainAnalyzer."""

    def test_full_table_scan_detection(self, optimizer_config):
        """Test detection of full table scans."""
        analyzer = ExplainAnalyzer(optimizer_config)

        explain_plan = """
        Fragment 0 [SINGLE]
        - TableScan[test_table]
        """

        context = {
            "query": "SELECT * FROM test_table",
            "explain_plan": explain_plan,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        assert len(recommendations) > 0
        assert any("Table Scan" in rec.title for rec in recommendations)

    def test_cross_join_detection(self, optimizer_config):
        """Test detection of CROSS JOINs."""
        analyzer = ExplainAnalyzer(optimizer_config)

        explain_plan = "CROSS JOIN detected in plan..."

        context = {
            "query": "SELECT * FROM t1 CROSS JOIN t2",
            "explain_plan": explain_plan,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        cross_join_recs = [r for r in recommendations if "Cross Join" in r.title]
        assert len(cross_join_recs) > 0
        assert cross_join_recs[0].severity == Severity.CRITICAL

    def test_no_explain_plan(self, optimizer_config):
        """Test when no explain plan is available."""
        analyzer = ExplainAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM table",
            "explain_plan": None,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)
        assert recommendations == []

    def test_complex_distributed_query(self, optimizer_config):
        """Test detection of complex distributed queries."""
        analyzer = ExplainAnalyzer(optimizer_config)

        explain_plan = """
        Fragment 0 [SINGLE]
        Fragment 1 [HASH]
        Fragment 2 [HASH]
        Fragment 3 [HASH]
        Fragment 4 [HASH]
        Fragment 5 [HASH]
        Fragment 6 [HASH]
        """

        context = {
            "query": "Complex query",
            "explain_plan": explain_plan,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        complex_recs = [r for r in recommendations if "Complex" in r.title or "Distributed" in r.title]
        assert len(complex_recs) > 0


class TestPartitionAnalyzer:
    """Tests for PartitionAnalyzer."""

    def test_unpartitioned_table(self, optimizer_config, sample_csv_table_metadata):
        """Test detection of unpartitioned tables."""
        analyzer = PartitionAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM csv_table",
            "table_metadata": {"csv_table": sample_csv_table_metadata},
            "parsed_explain_plan": None
        }

        recommendations = analyzer.analyze(context)

        unpart_recs = [r for r in recommendations if "Not Partitioned" in r.title]
        assert len(unpart_recs) > 0
        assert unpart_recs[0].severity == Severity.MEDIUM

    def test_missing_partition_filters(self, optimizer_config, sample_table_metadata):
        """Test detection of missing partition filters."""
        analyzer = PartitionAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM test_table WHERE id > 100",
            "table_metadata": {"test_table": sample_table_metadata},
            "parsed_explain_plan": None
        }

        recommendations = analyzer.analyze(context)

        missing_filter_recs = [r for r in recommendations if "Missing Partition Filters" in r.title]
        assert len(missing_filter_recs) > 0
        assert missing_filter_recs[0].severity == Severity.CRITICAL

    def test_with_partition_filters(self, optimizer_config, sample_table_metadata):
        """Test when partition filters are present."""
        analyzer = PartitionAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM test_table WHERE year = '2024' AND month = '01'",
            "table_metadata": {"test_table": sample_table_metadata},
            "parsed_explain_plan": None
        }

        recommendations = analyzer.analyze(context)

        # Should not have critical missing filter warnings
        missing_filter_recs = [r for r in recommendations
                              if "Missing Partition Filters" in r.title
                              and r.severity == Severity.CRITICAL]
        assert len(missing_filter_recs) == 0

    def test_no_tables(self, optimizer_config):
        """Test when no table metadata is available."""
        analyzer = PartitionAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM table",
            "table_metadata": {},
            "parsed_explain_plan": None
        }

        recommendations = analyzer.analyze(context)
        assert recommendations == []


class TestFormatAnalyzer:
    """Tests for FormatAnalyzer."""

    def test_csv_format_detection(self, optimizer_config, sample_csv_table_metadata):
        """Test detection of non-columnar CSV format."""
        analyzer = FormatAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM csv_table",
            "table_metadata": {"csv_table": sample_csv_table_metadata}
        }

        recommendations = analyzer.analyze(context)

        format_recs = [r for r in recommendations if r.category == Category.FORMAT]
        assert len(format_recs) > 0
        assert any("Non-Columnar Format" in rec.title for rec in format_recs)
        assert any(rec.severity == Severity.HIGH for rec in format_recs)

    def test_parquet_format_optimal(self, optimizer_config, sample_table_metadata):
        """Test that Parquet format is recognized as optimal."""
        analyzer = FormatAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM test_table",
            "table_metadata": {"test_table": sample_table_metadata}
        }

        recommendations = analyzer.analyze(context)

        format_recs = [r for r in recommendations if r.category == Category.FORMAT]
        # Should have INFO level recommendation for optimal format
        optimal_recs = [r for r in format_recs if r.severity == Severity.INFO]
        assert len(optimal_recs) > 0

    def test_uncompressed_data(self, optimizer_config):
        """Test detection of uncompressed data."""
        analyzer = FormatAnalyzer(optimizer_config)

        uncompressed_metadata = TableMetadata(
            database="test_db",
            table="uncompressed_table",
            input_format="org.apache.hadoop.mapred.TextInputFormat",
            compressed=False,
            partition_keys=[],
            columns=[]
        )

        context = {
            "query": "SELECT * FROM uncompressed_table",
            "table_metadata": {"uncompressed_table": uncompressed_metadata}
        }

        recommendations = analyzer.analyze(context)

        compression_recs = [r for r in recommendations if r.category == Category.COMPRESSION]
        assert len(compression_recs) > 0


class TestJoinAnalyzer:
    """Tests for JoinAnalyzer."""

    def test_multiple_joins(self, optimizer_config):
        """Test detection of multiple JOINs."""
        analyzer = JoinAnalyzer(optimizer_config)

        query = """
        SELECT *
        FROM t1
        JOIN t2 ON t1.id = t2.id
        JOIN t3 ON t2.id = t3.id
        JOIN t4 ON t3.id = t4.id
        JOIN t5 ON t4.id = t5.id
        """

        context = {
            "query": query,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        join_recs = [r for r in recommendations if "JOIN" in r.title]
        assert len(join_recs) > 0

    def test_outer_joins(self, optimizer_config):
        """Test detection of OUTER JOINs."""
        analyzer = JoinAnalyzer(optimizer_config)

        query = "SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id"

        context = {
            "query": query,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        outer_join_recs = [r for r in recommendations if "OUTER" in r.title]
        assert len(outer_join_recs) > 0

    def test_non_equality_joins(self, optimizer_config):
        """Test detection of non-equality JOINs."""
        analyzer = JoinAnalyzer(optimizer_config)

        query = "SELECT * FROM t1 JOIN t2 ON t1.id != t2.id"

        context = {
            "query": query,
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        non_eq_recs = [r for r in recommendations if "Non-Equality" in r.title]
        assert len(non_eq_recs) > 0
        assert non_eq_recs[0].severity == Severity.HIGH

    def test_no_joins(self, optimizer_config):
        """Test when query has no JOINs."""
        analyzer = JoinAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM table",
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)
        # May have INFO recommendations but no critical ones
        critical_recs = [r for r in recommendations if r.severity in [Severity.CRITICAL, Severity.HIGH]]
        assert len(critical_recs) == 0


class TestProjectionAnalyzer:
    """Tests for ProjectionAnalyzer."""

    def test_select_star_detection(self, optimizer_config):
        """Test detection of SELECT *."""
        analyzer = ProjectionAnalyzer(optimizer_config)

        context = {
            "query": "SELECT * FROM test_table",
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        select_star_recs = [r for r in recommendations if "SELECT *" in r.title]
        assert len(select_star_recs) > 0
        assert select_star_recs[0].severity == Severity.HIGH

    def test_count_star_detection(self, optimizer_config):
        """Test detection of COUNT(*)."""
        analyzer = ProjectionAnalyzer(optimizer_config)

        context = {
            "query": "SELECT COUNT(*) FROM test_table",
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        count_star_recs = [r for r in recommendations if "COUNT(*)" in r.title]
        assert len(count_star_recs) > 0

    def test_distinct_on_many_columns(self, optimizer_config):
        """Test detection of DISTINCT on many columns."""
        analyzer = ProjectionAnalyzer(optimizer_config)

        context = {
            "query": "SELECT DISTINCT col1, col2, col3, col4, col5, col6, col7 FROM test_table",
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        distinct_recs = [r for r in recommendations if "DISTINCT" in r.title]
        assert len(distinct_recs) > 0

    def test_specific_columns(self, optimizer_config):
        """Test that specific column selection is recognized."""
        analyzer = ProjectionAnalyzer(optimizer_config)

        context = {
            "query": "SELECT id, name, amount FROM test_table",
            "table_metadata": {}
        }

        recommendations = analyzer.analyze(context)

        # Should not have SELECT * warnings
        select_star_recs = [r for r in recommendations if "SELECT *" in r.title]
        assert len(select_star_recs) == 0


class TestAnalyzerBase:
    """Tests for base analyzer functionality."""

    def test_all_analyzers_have_name(self, optimizer_config):
        """Test that all analyzers have a name property."""
        analyzers = [
            CostAnalyzer(optimizer_config),
            ExplainAnalyzer(optimizer_config),
            FormatAnalyzer(optimizer_config),
            JoinAnalyzer(optimizer_config),
            PartitionAnalyzer(optimizer_config),
            ProjectionAnalyzer(optimizer_config),
        ]

        for analyzer in analyzers:
            assert analyzer.name
            assert isinstance(analyzer.name, str)
            assert len(analyzer.name) > 0

    def test_all_analyzers_enabled_by_default(self, optimizer_config):
        """Test that all analyzers are enabled by default."""
        analyzers = [
            CostAnalyzer(optimizer_config),
            ExplainAnalyzer(optimizer_config),
            FormatAnalyzer(optimizer_config),
            JoinAnalyzer(optimizer_config),
            PartitionAnalyzer(optimizer_config),
            ProjectionAnalyzer(optimizer_config),
        ]

        for analyzer in analyzers:
            assert analyzer.enabled is True
