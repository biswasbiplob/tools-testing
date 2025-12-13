"""Glue Data Catalog collector for table metadata."""

from typing import Optional
import boto3
from botocore.exceptions import ClientError

from ..models import TableMetadata, OptimizerConfig


class GlueCollector:
    """Collects table metadata from Glue Data Catalog."""

    def __init__(self, config: OptimizerConfig):
        """Initialize Glue collector."""
        self.config = config

        session_kwargs = {"region_name": config.region}
        if config.aws_profile:
            session_kwargs["profile_name"] = config.aws_profile

        session = boto3.Session(**session_kwargs)
        self.client = session.client("glue")

    def get_table_metadata(
        self, database: str, table: str, catalog: Optional[str] = None
    ) -> TableMetadata:
        """Get metadata for a specific table."""
        params = {
            "DatabaseName": database,
            "Name": table
        }

        if catalog or self.config.catalog:
            params["CatalogId"] = catalog or self.config.catalog

        try:
            response = self.client.get_table(**params)
            table_data = response["Table"]
            storage_desc = table_data.get("StorageDescriptor", {})

            # Extract partition keys
            partition_keys = [
                pk["Name"] for pk in table_data.get("PartitionKeys", [])
            ]

            # Extract columns
            columns = [
                {"name": col["Name"], "type": col["Type"]}
                for col in storage_desc.get("Columns", [])
            ]

            # Determine if compressed
            compressed = storage_desc.get("Compressed", False)

            # Get input/output format
            input_format = storage_desc.get("InputFormat", "")
            output_format = storage_desc.get("OutputFormat", "")

            # Detect table format from input format
            table_format = self._detect_format(input_format, output_format)

            return TableMetadata(
                database=database,
                table=table,
                location=storage_desc.get("Location"),
                input_format=input_format,
                output_format=output_format,
                serde_info=storage_desc.get("SerdeInfo"),
                partition_keys=partition_keys,
                columns=columns,
                parameters=table_data.get("Parameters", {}),
                table_type=table_data.get("TableType"),
                compressed=compressed,
                num_buckets=storage_desc.get("NumberOfBuckets"),
                storage_descriptor=storage_desc
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityNotFoundException":
                raise ValueError(f"Table {database}.{table} not found") from e
            raise RuntimeError(f"Failed to get table metadata: {e}") from e

    def get_partitions(
        self, database: str, table: str, max_partitions: int = 1000
    ) -> list[dict]:
        """Get partition information for a table."""
        try:
            paginator = self.client.get_paginator("get_partitions")
            page_iterator = paginator.paginate(
                DatabaseName=database,
                TableName=table,
                MaxResults=max_partitions
            )

            partitions = []
            for page in page_iterator:
                for partition in page.get("Partitions", []):
                    partition_info = {
                        "values": partition.get("Values", []),
                        "location": partition.get("StorageDescriptor", {}).get("Location"),
                        "parameters": partition.get("Parameters", {}),
                        "creation_time": partition.get("CreationTime"),
                        "last_access_time": partition.get("LastAccessTime"),
                    }
                    partitions.append(partition_info)

            return partitions

        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityNotFoundException":
                return []  # Table has no partitions
            raise RuntimeError(f"Failed to get partitions: {e}") from e

    def get_table_statistics(self, database: str, table: str) -> dict:
        """Get table statistics if available."""
        try:
            # Try to get column statistics
            response = self.client.get_table(DatabaseName=database, Name=table)
            parameters = response["Table"].get("Parameters", {})

            stats = {
                "num_rows": parameters.get("numRows"),
                "total_size": parameters.get("totalSize"),
                "num_files": parameters.get("numFiles"),
                "raw_data_size": parameters.get("rawDataSize"),
            }

            # Clean up None values
            return {k: v for k, v in stats.items() if v is not None}

        except ClientError:
            return {}

    def _detect_format(self, input_format: str, output_format: str) -> str:
        """Detect table format from input/output format strings."""
        format_lower = input_format.lower()

        if "parquet" in format_lower:
            return "PARQUET"
        elif "orc" in format_lower:
            return "ORC"
        elif "avro" in format_lower:
            return "AVRO"
        elif "json" in format_lower:
            return "JSON"
        # Check exact match first before partial matches
        elif input_format == "org.apache.hadoop.mapred.TextInputFormat":
            return "CSV"
        elif "textinput" in format_lower or "textoutput" in format_lower:
            return "TEXT"
        else:
            return "UNKNOWN"

    def list_databases(self, catalog: Optional[str] = None) -> list[str]:
        """List all databases in the catalog."""
        params = {}
        if catalog or self.config.catalog:
            params["CatalogId"] = catalog or self.config.catalog

        try:
            paginator = self.client.get_paginator("get_databases")
            page_iterator = paginator.paginate(**params)

            databases = []
            for page in page_iterator:
                for db in page.get("DatabaseList", []):
                    databases.append(db["Name"])

            return databases

        except ClientError as e:
            raise RuntimeError(f"Failed to list databases: {e}") from e

    def list_tables(self, database: str, catalog: Optional[str] = None) -> list[str]:
        """List all tables in a database."""
        params = {"DatabaseName": database}
        if catalog or self.config.catalog:
            params["CatalogId"] = catalog or self.config.catalog

        try:
            paginator = self.client.get_paginator("get_tables")
            page_iterator = paginator.paginate(**params)

            tables = []
            for page in page_iterator:
                for table in page.get("TableList", []):
                    tables.append(table["Name"])

            return tables

        except ClientError as e:
            raise RuntimeError(f"Failed to list tables: {e}") from e
