"""Glue Data Catalog collector for table metadata."""

from typing import Optional
from botocore.exceptions import ClientError

from ..models import TableMetadata, OptimizerConfig
from ..cache import TTLCache
from ..metrics import get_metrics_collector
from ..exceptions import TableNotFoundError, MetadataFetchError
from .base import BaseCollector


class GlueCollector(BaseCollector):
    """Collects table metadata from Glue Data Catalog with caching."""

    def __init__(self, config: OptimizerConfig):
        """
        Initialize collector with caching support.

        Args:
            config: Optimizer configuration
        """
        super().__init__(config)
        # Cache table metadata for 5 minutes
        self._metadata_cache = TTLCache(default_ttl_seconds=300)
        # Cache partitions for 2 minutes (they change more frequently)
        self._partition_cache = TTLCache(default_ttl_seconds=120)
        self._metrics = get_metrics_collector()

    @property
    def client_name(self) -> str:
        """Return the AWS service name."""
        return "glue"

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._metadata_cache.clear()
        self._partition_cache.clear()

    def get_table_metadata(
        self, database: str, table: str, catalog: Optional[str] = None
    ) -> TableMetadata:
        """
        Get metadata for a specific table with caching.

        Results are cached for 5 minutes to reduce API calls to Glue.

        Args:
            database: Database name
            table: Table name
            catalog: Catalog ID (optional)

        Returns:
            Table metadata

        Raises:
            ValueError: If table not found
            RuntimeError: If API call fails
        """
        # Create cache key
        cache_key = f"{database}.{table}.{catalog or self.config.catalog}"

        # Try to get from cache
        cached_metadata = self._metadata_cache.get(cache_key)
        if cached_metadata is not None:
            self._metrics.record_cache_hit()
            return cached_metadata

        # Not in cache, fetch from Glue
        self._metrics.record_cache_miss()
        self._metrics.record_api_call("glue")
        params = {
            "DatabaseName": database,
            "Name": table
        }
        self._add_catalog_to_params(params, catalog)

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

            metadata = TableMetadata(
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

            # Cache the result
            self._metadata_cache.set(cache_key, metadata)

            return metadata

        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityNotFoundException":
                raise TableNotFoundError(
                    database,
                    table,
                    details={"catalog": catalog or self.config.catalog}
                ) from e
            raise MetadataFetchError(
                "table",
                f"{database}.{table}",
                str(e),
                details={"catalog": catalog or self.config.catalog}
            ) from e

    def get_partitions(
        self, database: str, table: str, max_partitions: int = 1000
    ) -> list[dict]:
        """
        Get partition information for a table with caching.

        Results are cached for 2 minutes to reduce API calls to Glue.

        Args:
            database: Database name
            table: Table name
            max_partitions: Maximum number of partitions to fetch

        Returns:
            List of partition information dictionaries

        Raises:
            RuntimeError: If API call fails (returns [] if table has no partitions)
        """
        # Create cache key
        cache_key = f"{database}.{table}.partitions"

        # Try to get from cache
        cached_partitions = self._partition_cache.get(cache_key)
        if cached_partitions is not None:
            self._metrics.record_cache_hit()
            return cached_partitions

        # Not in cache, fetch from Glue
        self._metrics.record_cache_miss()
        self._metrics.record_api_call("glue")
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

            # Cache the result
            self._partition_cache.set(cache_key, partitions)

            return partitions

        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityNotFoundException":
                # Cache empty result too
                self._partition_cache.set(cache_key, [])
                return []  # Table has no partitions
            raise MetadataFetchError(
                "partitions",
                f"{database}.{table}",
                str(e),
                details={"max_partitions": max_partitions}
            ) from e

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
        self._add_catalog_to_params(params, catalog)

        try:
            paginator = self.client.get_paginator("get_databases")
            page_iterator = paginator.paginate(**params)

            databases = []
            for page in page_iterator:
                for db in page.get("DatabaseList", []):
                    databases.append(db["Name"])

            return databases

        except ClientError as e:
            raise MetadataFetchError(
                "databases",
                catalog or self.config.catalog or "default",
                str(e)
            ) from e

    def list_tables(self, database: str, catalog: Optional[str] = None) -> list[str]:
        """List all tables in a database."""
        params = {"DatabaseName": database}
        self._add_catalog_to_params(params, catalog)

        try:
            paginator = self.client.get_paginator("get_tables")
            page_iterator = paginator.paginate(**params)

            tables = []
            for page in page_iterator:
                for table in page.get("TableList", []):
                    tables.append(table["Name"])

            return tables

        except ClientError as e:
            raise MetadataFetchError(
                "tables",
                database,
                str(e),
                details={"catalog": catalog or self.config.catalog}
            ) from e
