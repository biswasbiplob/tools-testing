"""Tests for collector modules."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from athena_optimizer.collectors import AthenaCollector, GlueCollector
from athena_optimizer.models import QueryMetrics, TableMetadata
from athena_optimizer.exceptions import (
    QueryExecutionError,
    QueryTimeoutError,
    TableNotFoundError,
)


class TestAthenaCollector:
    """Tests for AthenaCollector."""

    @patch('boto3.Session')
    def test_initialization(self, mock_session, optimizer_config):
        """Test collector initialization."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        collector = AthenaCollector(optimizer_config)

        assert collector.config == optimizer_config
        # Access client property to trigger lazy loading
        _ = collector.client
        mock_session.assert_called_once()

    @patch('boto3.Session')
    def test_execute_query_success(self, mock_session, optimizer_config):
        """Test successful query execution."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock responses
        mock_client.start_query_execution.return_value = {
            "QueryExecutionId": "test-query-123"
        }

        mock_client.get_query_execution.return_value = {
            "QueryExecution": {
                "Status": {"State": "SUCCEEDED"},
                "Statistics": {
                    "DataScannedInBytes": 1099511627776,
                    "EngineExecutionTimeInMillis": 5000,
                },
                "ResultConfiguration": {
                    "OutputLocation": "s3://test/results.csv"
                }
            }
        }

        collector = AthenaCollector(optimizer_config)
        query_id, metrics = collector.execute_query("SELECT * FROM test")

        assert query_id == "test-query-123"
        assert isinstance(metrics, QueryMetrics)
        assert metrics.data_scanned_bytes == 1099511627776

    @patch('boto3.Session')
    def test_execute_query_failure(self, mock_session, optimizer_config):
        """Test query execution failure."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_client.start_query_execution.return_value = {
            "QueryExecutionId": "test-query-123"
        }

        mock_client.get_query_execution.return_value = {
            "QueryExecution": {
                "Status": {
                    "State": "FAILED",
                    "StateChangeReason": "Syntax error"
                }
            }
        }

        collector = AthenaCollector(optimizer_config)

        with pytest.raises(QueryExecutionError):
            collector.execute_query("SELECT * FROM test")

    @patch('boto3.Session')
    def test_execute_query_without_wait(self, mock_session, optimizer_config):
        """Test query execution without waiting."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_client.start_query_execution.return_value = {
            "QueryExecutionId": "test-query-123"
        }

        collector = AthenaCollector(optimizer_config)
        query_id, metrics = collector.execute_query(
            "SELECT * FROM test",
            wait_for_completion=False
        )

        assert query_id == "test-query-123"
        assert metrics is None

    @patch('boto3.Session')
    def test_get_explain_plan(self, mock_session, optimizer_config):
        """Test getting EXPLAIN plan."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_client.start_query_execution.return_value = {
            "QueryExecutionId": "test-query-123"
        }

        mock_client.get_query_execution.return_value = {
            "QueryExecution": {
                "Status": {"State": "SUCCEEDED"},
                "Statistics": {},
                "ResultConfiguration": {}
            }
        }

        # Mock paginator for results
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "ResultSet": {
                    "ColumnInfo": [{"Name": "Explain"}],
                    "Rows": [
                        {"Data": [{"VarCharValue": "Explain"}]},  # Header
                        {"Data": [{"VarCharValue": "Fragment 0 [SINGLE]"}]},
                        {"Data": [{"VarCharValue": "- TableScan[test_table]"}]},
                    ]
                }
            }
        ]

        collector = AthenaCollector(optimizer_config)
        plan = collector.get_explain_plan("SELECT * FROM test")

        assert "Fragment 0" in plan
        assert "TableScan" in plan

    @patch('boto3.Session')
    def test_query_timeout(self, mock_session, optimizer_config):
        """Test query timeout."""
        optimizer_config.timeout_seconds = 4  # Very short timeout

        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_client.start_query_execution.return_value = {
            "QueryExecutionId": "test-query-123"
        }

        # Always return RUNNING status
        mock_client.get_query_execution.return_value = {
            "QueryExecution": {
                "Status": {"State": "RUNNING"}
            }
        }

        collector = AthenaCollector(optimizer_config)

        with pytest.raises(QueryTimeoutError):
            collector.execute_query("SELECT * FROM test")


class TestGlueCollector:
    """Tests for GlueCollector."""

    @patch('boto3.Session')
    def test_initialization(self, mock_session, optimizer_config):
        """Test collector initialization."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        collector = GlueCollector(optimizer_config)

        assert collector.config == optimizer_config
        # Access client property to trigger lazy loading
        _ = collector.client
        mock_session.assert_called_once()

    @patch('boto3.Session')
    def test_get_table_metadata_success(self, mock_session, optimizer_config):
        """Test successful table metadata retrieval."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_client.get_table.return_value = {
            "Table": {
                "Name": "test_table",
                "DatabaseName": "test_db",
                "TableType": "EXTERNAL_TABLE",
                "StorageDescriptor": {
                    "Location": "s3://bucket/data/",
                    "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                    "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                    "Compressed": True,
                    "Columns": [
                        {"Name": "id", "Type": "bigint"},
                        {"Name": "name", "Type": "string"},
                    ],
                    "SerdeInfo": {}
                },
                "PartitionKeys": [
                    {"Name": "year", "Type": "string"},
                ],
                "Parameters": {"numRows": "1000"}
            }
        }

        collector = GlueCollector(optimizer_config)
        metadata = collector.get_table_metadata("test_db", "test_table")

        assert isinstance(metadata, TableMetadata)
        assert metadata.database == "test_db"
        assert metadata.table == "test_table"
        assert len(metadata.partition_keys) == 1
        assert len(metadata.columns) == 2
        assert metadata.compressed is True

    @patch('boto3.Session')
    def test_get_table_metadata_not_found(self, mock_session, optimizer_config):
        """Test table not found error."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        error_response = {"Error": {"Code": "EntityNotFoundException"}}
        mock_client.get_table.side_effect = ClientError(error_response, "get_table")

        collector = GlueCollector(optimizer_config)

        with pytest.raises(TableNotFoundError):
            collector.get_table_metadata("test_db", "nonexistent")

    @patch('boto3.Session')
    def test_get_partitions(self, mock_session, optimizer_config):
        """Test getting table partitions."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock paginator
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Partitions": [
                    {
                        "Values": ["2024", "01"],
                        "StorageDescriptor": {"Location": "s3://bucket/data/year=2024/month=01"},
                        "Parameters": {},
                    },
                    {
                        "Values": ["2024", "02"],
                        "StorageDescriptor": {"Location": "s3://bucket/data/year=2024/month=02"},
                        "Parameters": {},
                    }
                ]
            }
        ]

        collector = GlueCollector(optimizer_config)
        partitions = collector.get_partitions("test_db", "test_table")

        assert len(partitions) == 2
        assert partitions[0]["values"] == ["2024", "01"]
        assert partitions[1]["values"] == ["2024", "02"]

    @patch('boto3.Session')
    def test_get_partitions_no_partitions(self, mock_session, optimizer_config):
        """Test table with no partitions."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        error_response = {"Error": {"Code": "EntityNotFoundException"}}
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = ClientError(error_response, "get_partitions")

        collector = GlueCollector(optimizer_config)
        partitions = collector.get_partitions("test_db", "test_table")

        assert partitions == []

    @patch('boto3.Session')
    def test_get_table_statistics(self, mock_session, optimizer_config):
        """Test getting table statistics."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_client.get_table.return_value = {
            "Table": {
                "StorageDescriptor": {},
                "Parameters": {
                    "numRows": "1000000",
                    "totalSize": "536870912",
                    "numFiles": "100"
                }
            }
        }

        collector = GlueCollector(optimizer_config)
        stats = collector.get_table_statistics("test_db", "test_table")

        assert stats["num_rows"] == "1000000"
        assert stats["total_size"] == "536870912"
        assert stats["num_files"] == "100"

    @patch('boto3.Session')
    def test_list_databases(self, mock_session, optimizer_config):
        """Test listing databases."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "DatabaseList": [
                    {"Name": "db1"},
                    {"Name": "db2"},
                ]
            }
        ]

        collector = GlueCollector(optimizer_config)
        databases = collector.list_databases()

        assert len(databases) == 2
        assert "db1" in databases
        assert "db2" in databases

    @patch('boto3.Session')
    def test_list_tables(self, mock_session, optimizer_config):
        """Test listing tables in a database."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "TableList": [
                    {"Name": "table1"},
                    {"Name": "table2"},
                ]
            }
        ]

        collector = GlueCollector(optimizer_config)
        tables = collector.list_tables("test_db")

        assert len(tables) == 2
        assert "table1" in tables
        assert "table2" in tables

    @patch('boto3.Session')
    def test_format_detection(self, mock_session, optimizer_config):
        """Test table format detection."""
        collector = GlueCollector(optimizer_config)

        # Test Parquet
        assert collector._detect_format(
            "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            ""
        ) == "PARQUET"

        # Test ORC
        assert collector._detect_format(
            "org.apache.hadoop.hive.ql.io.orc.OrcInputFormat",
            ""
        ) == "ORC"

        # Test CSV
        assert collector._detect_format(
            "org.apache.hadoop.mapred.TextInputFormat",
            ""
        ) == "CSV"

        # Test unknown
        assert collector._detect_format(
            "unknown.format",
            ""
        ) == "UNKNOWN"
