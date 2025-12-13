"""Metrics and observability for the Athena SQL Optimizer."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from threading import Lock


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def complete(self, success: bool = True, error: Optional[str] = None):
        """Mark operation as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all operations."""
    queries_analyzed: int = 0
    queries_succeeded: int = 0
    queries_failed: int = 0
    total_analysis_time_ms: float = 0.0
    avg_analysis_time_ms: float = 0.0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Cost metrics
    total_current_cost_usd: float = 0.0
    total_optimized_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    avg_savings_percentage: float = 0.0

    # Recommendation metrics
    total_recommendations: int = 0
    recommendations_by_severity: Dict[str, int] = field(default_factory=dict)
    recommendations_by_category: Dict[str, int] = field(default_factory=dict)

    # AWS API metrics
    athena_api_calls: int = 0
    glue_api_calls: int = 0

    # Analyzer metrics
    analyzer_executions: Dict[str, int] = field(default_factory=dict)
    analyzer_failures: Dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics for observability.

    This class is thread-safe and tracks various operational metrics
    including performance, success rates, cache efficiency, and cost savings.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._lock = Lock()
        self._operations: List[OperationMetrics] = []
        self._aggregate = AggregateMetrics()
        self._start_time = time.time()

    def start_operation(self, operation_name: str, metadata: Optional[Dict] = None) -> OperationMetrics:
        """
        Start tracking an operation.

        Args:
            operation_name: Name of the operation
            metadata: Optional metadata to attach

        Returns:
            OperationMetrics object to track the operation
        """
        op = OperationMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        return op

    def record_operation(self, operation: OperationMetrics):
        """
        Record a completed operation.

        Args:
            operation: The completed operation metrics
        """
        with self._lock:
            self._operations.append(operation)

            # Update aggregate metrics based on operation type
            if operation.operation_name == "analyze_query":
                self._aggregate.queries_analyzed += 1
                if operation.success:
                    self._aggregate.queries_succeeded += 1
                else:
                    self._aggregate.queries_failed += 1

                if operation.duration_ms:
                    self._aggregate.total_analysis_time_ms += operation.duration_ms
                    self._aggregate.avg_analysis_time_ms = (
                        self._aggregate.total_analysis_time_ms / self._aggregate.queries_analyzed
                    )

    def record_cache_hit(self):
        """Record a cache hit."""
        with self._lock:
            self._aggregate.cache_hits += 1
            self._update_cache_hit_rate()

    def record_cache_miss(self):
        """Record a cache miss."""
        with self._lock:
            self._aggregate.cache_misses += 1
            self._update_cache_hit_rate()

    def _update_cache_hit_rate(self):
        """Update the cache hit rate."""
        total = self._aggregate.cache_hits + self._aggregate.cache_misses
        if total > 0:
            self._aggregate.cache_hit_rate = (self._aggregate.cache_hits / total) * 100

    def record_cost_analysis(self, current_cost: float, optimized_cost: float, savings: float):
        """
        Record cost analysis results.

        Args:
            current_cost: Current query cost in USD
            optimized_cost: Optimized query cost in USD
            savings: Savings in USD
        """
        with self._lock:
            self._aggregate.total_current_cost_usd += current_cost
            self._aggregate.total_optimized_cost_usd += optimized_cost
            self._aggregate.total_savings_usd += savings

            if self._aggregate.total_current_cost_usd > 0:
                self._aggregate.avg_savings_percentage = (
                    (self._aggregate.total_savings_usd / self._aggregate.total_current_cost_usd) * 100
                )

    def record_recommendations(self, recommendations: List):
        """
        Record generated recommendations.

        Args:
            recommendations: List of Recommendation objects
        """
        with self._lock:
            self._aggregate.total_recommendations += len(recommendations)

            for rec in recommendations:
                # Count by severity
                severity = str(rec.severity.value if hasattr(rec.severity, 'value') else rec.severity)
                self._aggregate.recommendations_by_severity[severity] = (
                    self._aggregate.recommendations_by_severity.get(severity, 0) + 1
                )

                # Count by category
                category = str(rec.category.value if hasattr(rec.category, 'value') else rec.category)
                self._aggregate.recommendations_by_category[category] = (
                    self._aggregate.recommendations_by_category.get(category, 0) + 1
                )

    def record_api_call(self, service: str):
        """
        Record an AWS API call.

        Args:
            service: AWS service name (athena or glue)
        """
        with self._lock:
            if service.lower() == "athena":
                self._aggregate.athena_api_calls += 1
            elif service.lower() == "glue":
                self._aggregate.glue_api_calls += 1

    def record_analyzer_execution(self, analyzer_name: str, success: bool = True):
        """
        Record analyzer execution.

        Args:
            analyzer_name: Name of the analyzer
            success: Whether the analyzer succeeded
        """
        with self._lock:
            self._aggregate.analyzer_executions[analyzer_name] = (
                self._aggregate.analyzer_executions.get(analyzer_name, 0) + 1
            )

            if not success:
                self._aggregate.analyzer_failures[analyzer_name] = (
                    self._aggregate.analyzer_failures.get(analyzer_name, 0) + 1
                )

    def get_metrics(self) -> AggregateMetrics:
        """
        Get current aggregate metrics.

        Returns:
            AggregateMetrics object with current statistics
        """
        with self._lock:
            return self._aggregate

    def get_operations(self, limit: Optional[int] = None) -> List[OperationMetrics]:
        """
        Get recorded operations.

        Args:
            limit: Maximum number of operations to return (most recent first)

        Returns:
            List of OperationMetrics
        """
        with self._lock:
            ops = self._operations.copy()
            if limit:
                ops = ops[-limit:]
            return ops

    def get_summary(self) -> Dict:
        """
        Get a summary of all metrics.

        Returns:
            Dictionary with formatted metrics summary
        """
        with self._lock:
            uptime_seconds = time.time() - self._start_time

            return {
                "uptime_seconds": uptime_seconds,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "queries": {
                    "total_analyzed": self._aggregate.queries_analyzed,
                    "succeeded": self._aggregate.queries_succeeded,
                    "failed": self._aggregate.queries_failed,
                    "success_rate": (
                        (self._aggregate.queries_succeeded / self._aggregate.queries_analyzed * 100)
                        if self._aggregate.queries_analyzed > 0 else 0.0
                    ),
                    "avg_analysis_time_ms": round(self._aggregate.avg_analysis_time_ms, 2),
                },
                "cache": {
                    "hits": self._aggregate.cache_hits,
                    "misses": self._aggregate.cache_misses,
                    "hit_rate": round(self._aggregate.cache_hit_rate, 2),
                },
                "costs": {
                    "total_current_usd": round(self._aggregate.total_current_cost_usd, 2),
                    "total_optimized_usd": round(self._aggregate.total_optimized_cost_usd, 2),
                    "total_savings_usd": round(self._aggregate.total_savings_usd, 2),
                    "avg_savings_percentage": round(self._aggregate.avg_savings_percentage, 2),
                },
                "recommendations": {
                    "total": self._aggregate.total_recommendations,
                    "by_severity": self._aggregate.recommendations_by_severity,
                    "by_category": self._aggregate.recommendations_by_category,
                },
                "aws_api_calls": {
                    "athena": self._aggregate.athena_api_calls,
                    "glue": self._aggregate.glue_api_calls,
                    "total": self._aggregate.athena_api_calls + self._aggregate.glue_api_calls,
                },
                "analyzers": {
                    "executions": self._aggregate.analyzer_executions,
                    "failures": self._aggregate.analyzer_failures,
                },
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._operations.clear()
            self._aggregate = AggregateMetrics()
            self._start_time = time.time()


# Global metrics collector instance
_global_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        Global MetricsCollector instance
    """
    return _global_metrics
