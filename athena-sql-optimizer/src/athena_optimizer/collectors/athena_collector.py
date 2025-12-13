"""Athena query collector for executing queries and retrieving results."""

import time
from typing import Optional
from botocore.exceptions import ClientError

from ..models import QueryMetrics, OptimizerConfig
from .base import BaseCollector


class AthenaCollector(BaseCollector):
    """Collects query execution data from Athena."""

    @property
    def client_name(self) -> str:
        """Return the AWS service name."""
        return "athena"

    def execute_query(
        self,
        query: str,
        database: Optional[str] = None,
        wait_for_completion: bool = True
    ) -> tuple[str, Optional[QueryMetrics]]:
        """
        Execute a query in Athena.

        Returns:
            Tuple of (query_execution_id, metrics)
        """
        execution_params = {
            "QueryString": query,
            "ResultConfiguration": {
                "OutputLocation": self.config.s3_output_location
            },
            "WorkGroup": self.config.workgroup,
            "QueryExecutionContext": {
                "Catalog": self.config.catalog
            }
        }

        # Add database if provided
        db = database or self.config.database
        if db:
            execution_params["QueryExecutionContext"]["Database"] = db

        try:
            response = self.client.start_query_execution(**execution_params)
            query_execution_id = response["QueryExecutionId"]

            if not wait_for_completion:
                return query_execution_id, None

            # Wait for query to complete
            metrics = self._wait_for_query(query_execution_id)
            return query_execution_id, metrics

        except ClientError as e:
            raise RuntimeError(f"Failed to execute query: {e}") from e

    def _wait_for_query(self, query_execution_id: str) -> QueryMetrics:
        """Wait for query completion and return metrics."""
        max_attempts = self.config.timeout_seconds // 2

        for _ in range(max_attempts):
            try:
                response = self.client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )

                status = response["QueryExecution"]["Status"]["State"]

                if status == "SUCCEEDED":
                    return self._extract_metrics(response["QueryExecution"])
                elif status == "FAILED":
                    reason = response["QueryExecution"]["Status"].get(
                        "StateChangeReason", "Unknown error"
                    )
                    raise RuntimeError(f"Query failed: {reason}")
                elif status == "CANCELLED":
                    raise RuntimeError("Query was cancelled")

                # Still running, wait and retry
                time.sleep(2)

            except ClientError as e:
                raise RuntimeError(f"Failed to get query status: {e}") from e

        raise TimeoutError(
            f"Query did not complete within {self.config.timeout_seconds} seconds"
        )

    def _extract_metrics(self, query_execution: dict) -> QueryMetrics:
        """Extract metrics from query execution response."""
        stats = query_execution.get("Statistics", {})
        result_config = query_execution.get("ResultConfiguration", {})

        return QueryMetrics(
            data_scanned_bytes=stats.get("DataScannedInBytes", 0),
            execution_time_ms=stats.get("EngineExecutionTimeInMillis", 0),
            query_queue_time_ms=stats.get("QueryQueueTimeInMillis", 0),
            query_planning_time_ms=stats.get("QueryPlanningTimeInMillis", 0),
            service_processing_time_ms=stats.get("ServiceProcessingTimeInMillis", 0),
            output_location=result_config.get("OutputLocation"),
            statistics=stats
        )

    def get_query_results(self, query_execution_id: str) -> list[dict]:
        """Get query results as list of row dictionaries."""
        try:
            paginator = self.client.get_paginator("get_query_results")
            page_iterator = paginator.paginate(QueryExecutionId=query_execution_id)

            all_rows = []
            columns = []

            for page in page_iterator:
                result_set = page.get("ResultSet", {})

                # Get column names from first page
                if not columns and "ColumnInfo" in result_set:
                    columns = [col["Name"] for col in result_set["ColumnInfo"]]

                # Process rows (skip header row on first page)
                rows = result_set.get("Rows", [])
                start_idx = 1 if len(all_rows) == 0 else 0

                for row in rows[start_idx:]:
                    data = row.get("Data", [])
                    row_dict = {
                        columns[i]: data[i].get("VarCharValue", "")
                        for i in range(len(columns))
                    }
                    all_rows.append(row_dict)

            return all_rows

        except ClientError as e:
            raise RuntimeError(f"Failed to get query results: {e}") from e

    def get_explain_plan(self, query: str, database: Optional[str] = None) -> str:
        """Get EXPLAIN plan for a query."""
        explain_query = f"EXPLAIN {query}"
        query_id, _ = self.execute_query(explain_query, database)
        results = self.get_query_results(query_id)

        # Combine all rows into a single plan string
        plan = "\n".join(row.get("Explain", "") for row in results)
        return plan

    def get_explain_analyze_plan(
        self, query: str, database: Optional[str] = None
    ) -> tuple[str, QueryMetrics]:
        """
        Get EXPLAIN ANALYZE plan for a query.
        WARNING: This executes the query and incurs costs.
        """
        explain_query = f"EXPLAIN ANALYZE {query}"
        query_id, metrics = self.execute_query(explain_query, database)
        results = self.get_query_results(query_id)

        # Combine all rows into a single plan string
        plan = "\n".join(row.get("Query Plan", "") for row in results if row)
        return plan, metrics or QueryMetrics()
