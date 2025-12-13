"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import Mock, MagicMock
from athena_optimizer.models import OptimizerConfig, TableMetadata, QueryMetrics, ExplainPlan


@pytest.fixture
def optimizer_config():
    """Create a test optimizer configuration."""
    return OptimizerConfig(
        aws_profile=None,
        region="us-east-1",
        workgroup="test-workgroup",
        s3_output_location="s3://test-bucket/results/",
        catalog="AwsDataCatalog",
        database="test_database",
        run_explain_analyze=False,
        athena_cost_per_tb=5.0,
        timeout_seconds=300
    )


@pytest.fixture
def sample_table_metadata():
    """Create sample table metadata."""
    return TableMetadata(
        database="test_db",
        table="test_table",
        location="s3://test-bucket/data/",
        input_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
        output_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
        partition_keys=["year", "month"],
        columns=[
            {"name": "id", "type": "bigint"},
            {"name": "name", "type": "string"},
            {"name": "amount", "type": "decimal(10,2)"},
            {"name": "created_date", "type": "date"},
        ],
        parameters={"numRows": "1000000"},
        table_type="EXTERNAL_TABLE",
        compressed=True
    )


@pytest.fixture
def sample_csv_table_metadata():
    """Create sample CSV table metadata."""
    return TableMetadata(
        database="test_db",
        table="csv_table",
        location="s3://test-bucket/csv-data/",
        input_format="org.apache.hadoop.mapred.TextInputFormat",
        output_format="org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
        partition_keys=[],
        columns=[
            {"name": "col1", "type": "string"},
            {"name": "col2", "type": "int"},
        ],
        parameters={},
        table_type="EXTERNAL_TABLE",
        compressed=False
    )


@pytest.fixture
def sample_query_metrics():
    """Create sample query metrics."""
    return QueryMetrics(
        data_scanned_bytes=1099511627776,  # 1 TB
        execution_time_ms=5000,
        query_queue_time_ms=100,
        query_planning_time_ms=200,
        service_processing_time_ms=100,
        output_location="s3://test-bucket/results/query-123.csv",
        statistics={
            "DataScannedInBytes": 1099511627776,
            "EngineExecutionTimeInMillis": 5000,
        }
    )


@pytest.fixture
def sample_explain_plan():
    """Create sample EXPLAIN plan."""
    return ExplainPlan(
        raw_plan="""Fragment 0 [SINGLE]
    Output layout: [customer_id, total]
    Output partitioning: SINGLE []
    - Aggregate(FINAL)[customer_id]
            Estimates: {rows: 1000 (16.00kB), cpu: ?, memory: ?, network: ?}
            total := sum(amount)
        - LocalExchange[HASH][$hashvalue] ("customer_id")
                Estimates: {rows: ? (?), cpu: ?, memory: ?, network: ?}
            - RemoteSource[1]
                    MemoryEstimate: {rows: ?, bytes: ?}

Fragment 1 [SOURCE]
    Output layout: [customer_id, amount, $hashvalue_2]
    Output partitioning: HASH [customer_id][$hashvalue_2]
    - ScanProject[table = awsdatacatalog:test_db:test_table]
            Estimates: {rows: 1000000 (16.00MB), cpu: 16.00M, memory: 0B, network: 0B}
            $hashvalue_2 := combine_hash(bigint '0', COALESCE("$operator$hash_code"("customer_id"), 0))
            amount := amount:decimal(10,2):REGULAR
            customer_id := customer_id:bigint:REGULAR
""",
        distributed=True,
        fragment_count=2,
        operations=["Aggregate", "LocalExchange", "RemoteSource", "ScanProject"],
        table_scans=["awsdatacatalog:test_db:test_table"],
        partition_filters=[]
    )


@pytest.fixture
def mock_athena_client():
    """Create a mock Athena client."""
    mock_client = Mock()

    # Mock start_query_execution
    mock_client.start_query_execution.return_value = {
        "QueryExecutionId": "test-query-123"
    }

    # Mock get_query_execution
    mock_client.get_query_execution.return_value = {
        "QueryExecution": {
            "Status": {"State": "SUCCEEDED"},
            "Statistics": {
                "DataScannedInBytes": 1099511627776,
                "EngineExecutionTimeInMillis": 5000,
                "QueryQueueTimeInMillis": 100,
                "QueryPlanningTimeInMillis": 200,
                "ServiceProcessingTimeInMillis": 100,
            },
            "ResultConfiguration": {
                "OutputLocation": "s3://test-bucket/results/test-query-123.csv"
            }
        }
    }

    # Mock get_query_results
    mock_client.get_paginator.return_value.paginate.return_value = [
        {
            "ResultSet": {
                "ColumnInfo": [{"Name": "Explain"}],
                "Rows": [
                    {"Data": [{"VarCharValue": "Explain"}]},  # Header
                    {"Data": [{"VarCharValue": "Fragment 0 [SINGLE]"}]},
                ]
            }
        }
    ]

    return mock_client


@pytest.fixture
def mock_glue_client():
    """Create a mock Glue client."""
    mock_client = Mock()

    # Mock get_table
    mock_client.get_table.return_value = {
        "Table": {
            "Name": "test_table",
            "DatabaseName": "test_db",
            "TableType": "EXTERNAL_TABLE",
            "StorageDescriptor": {
                "Location": "s3://test-bucket/data/",
                "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                "Compressed": True,
                "Columns": [
                    {"Name": "id", "Type": "bigint"},
                    {"Name": "name", "Type": "string"},
                ],
                "SerdeInfo": {
                    "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
                }
            },
            "PartitionKeys": [
                {"Name": "year", "Type": "string"},
                {"Name": "month", "Type": "string"},
            ],
            "Parameters": {"numRows": "1000000"}
        }
    }

    # Mock get_partitions
    mock_client.get_paginator.return_value.paginate.return_value = [
        {
            "Partitions": [
                {
                    "Values": ["2024", "01"],
                    "StorageDescriptor": {"Location": "s3://test-bucket/data/year=2024/month=01"},
                    "Parameters": {},
                    "CreationTime": "2024-01-01T00:00:00Z"
                }
            ]
        }
    ]

    return mock_client


@pytest.fixture
def sample_queries():
    """Common test queries."""
    return {
        "select_star": "SELECT * FROM test_table",
        "with_partition": "SELECT * FROM test_table WHERE year = '2024' AND month = '01'",
        "without_partition": "SELECT * FROM test_table WHERE id > 100",
        "cross_join": "SELECT * FROM table1 CROSS JOIN table2",
        "multiple_joins": """
            SELECT a.*, b.*, c.*
            FROM table1 a
            JOIN table2 b ON a.id = b.id
            LEFT JOIN table3 c ON b.id = c.id
        """,
        "optimized": """
            SELECT id, name, amount
            FROM test_table
            WHERE year = '2024' AND month = '01'
            LIMIT 100
        """,
    }
