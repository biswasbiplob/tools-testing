"""Custom exceptions for the Athena SQL Optimizer."""


class OptimizerError(Exception):
    """Base exception for all optimizer errors."""

    def __init__(self, message: str, details: dict = None):
        """
        Initialize optimizer error.

        Args:
            message: Error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(OptimizerError):
    """Raised when configuration is invalid or missing required parameters."""
    pass


class AWSConnectionError(OptimizerError):
    """Raised when connection to AWS services fails."""
    pass


class TableNotFoundError(OptimizerError):
    """Raised when a requested table does not exist."""

    def __init__(self, database: str, table: str, details: dict = None):
        """
        Initialize table not found error.

        Args:
            database: Database name
            table: Table name
            details: Optional additional context
        """
        message = f"Table {database}.{table} not found"
        super().__init__(message, details)
        self.database = database
        self.table = table


class QueryExecutionError(OptimizerError):
    """Raised when Athena query execution fails."""

    def __init__(self, query_id: str, reason: str, details: dict = None):
        """
        Initialize query execution error.

        Args:
            query_id: Athena query execution ID
            reason: Failure reason
            details: Optional additional context
        """
        message = f"Query {query_id} failed: {reason}"
        super().__init__(message, details)
        self.query_id = query_id
        self.reason = reason


class QueryTimeoutError(OptimizerError):
    """Raised when a query exceeds the timeout limit."""

    def __init__(self, query_id: str, timeout_seconds: int, details: dict = None):
        """
        Initialize query timeout error.

        Args:
            query_id: Athena query execution ID
            timeout_seconds: Timeout limit that was exceeded
            details: Optional additional context
        """
        message = f"Query {query_id} timed out after {timeout_seconds} seconds"
        super().__init__(message, details)
        self.query_id = query_id
        self.timeout_seconds = timeout_seconds


class MetadataFetchError(OptimizerError):
    """Raised when fetching metadata from Glue fails."""

    def __init__(self, resource: str, resource_id: str, reason: str, details: dict = None):
        """
        Initialize metadata fetch error.

        Args:
            resource: Resource type (table, partition, etc.)
            resource_id: Resource identifier
            reason: Failure reason
            details: Optional additional context
        """
        message = f"Failed to fetch {resource} metadata for {resource_id}: {reason}"
        super().__init__(message, details)
        self.resource = resource
        self.resource_id = resource_id
        self.reason = reason


class SQLParsingError(OptimizerError):
    """Raised when SQL parsing fails."""

    def __init__(self, query: str, reason: str, details: dict = None):
        """
        Initialize SQL parsing error.

        Args:
            query: SQL query that failed to parse
            reason: Parsing failure reason
            details: Optional additional context
        """
        message = f"Failed to parse SQL query: {reason}"
        super().__init__(message, details)
        self.query = query
        self.reason = reason


class AnalyzerError(OptimizerError):
    """Raised when an analyzer encounters an error."""

    def __init__(self, analyzer_name: str, reason: str, details: dict = None):
        """
        Initialize analyzer error.

        Args:
            analyzer_name: Name of the analyzer that failed
            reason: Failure reason
            details: Optional additional context
        """
        message = f"Analyzer {analyzer_name} failed: {reason}"
        super().__init__(message, details)
        self.analyzer_name = analyzer_name
        self.reason = reason


class CacheError(OptimizerError):
    """Raised when cache operations fail."""
    pass


class ValidationError(OptimizerError):
    """Raised when input validation fails."""

    def __init__(self, field: str, reason: str, details: dict = None):
        """
        Initialize validation error.

        Args:
            field: Field name that failed validation
            reason: Validation failure reason
            details: Optional additional context
        """
        message = f"Validation failed for {field}: {reason}"
        super().__init__(message, details)
        self.field = field
        self.reason = reason
