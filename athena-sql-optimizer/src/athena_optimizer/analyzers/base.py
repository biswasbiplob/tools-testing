"""Base analyzer interface for extensible analysis plugins."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

from ..models import Recommendation, OptimizerConfig, TableMetadata, QueryMetrics
from ..types import AnalysisContext


class BaseAnalyzer(ABC):
    """Base class for all analyzers with helper methods."""

    def __init__(self, config: OptimizerConfig):
        """Initialize analyzer with configuration."""
        self.config = config

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> list[Recommendation]:
        """
        Analyze the query and return recommendations.

        Args:
            context: Type-safe analysis context containing query, metadata, etc.

        Returns:
            List of recommendations
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return analyzer name."""
        pass

    @property
    def enabled(self) -> bool:
        """Check if analyzer is enabled."""
        return True

    # Helper methods to reduce duplication across analyzers
    def get_table_metadata(self, context: AnalysisContext) -> Dict[str, TableMetadata]:
        """
        Safely get table metadata from context.

        Args:
            context: Analysis context

        Returns:
            Dictionary of table metadata (empty dict if not present)
        """
        return context.get("table_metadata", {})

    def get_query_metrics(self, context: AnalysisContext) -> Optional[QueryMetrics]:
        """
        Safely get query metrics from context.

        Args:
            context: Analysis context

        Returns:
            Query metrics if available, None otherwise
        """
        return context.get("query_metrics")

    def get_explain_plan(self, context: AnalysisContext) -> Optional[str]:
        """
        Safely get EXPLAIN plan from context.

        Args:
            context: Analysis context

        Returns:
            EXPLAIN plan string if available, None otherwise
        """
        return context.get("explain_plan")

    def get_explain_analyze_plan(self, context: AnalysisContext) -> Optional[str]:
        """
        Safely get EXPLAIN ANALYZE plan from context.

        Args:
            context: Analysis context

        Returns:
            EXPLAIN ANALYZE plan string if available, None otherwise
        """
        return context.get("explain_analyze_plan")

    def has_explain_plan(self, context: AnalysisContext) -> bool:
        """
        Check if EXPLAIN plan is available.

        Args:
            context: Analysis context

        Returns:
            True if EXPLAIN plan is present
        """
        return self.get_explain_plan(context) is not None

    def has_explain_analyze(self, context: AnalysisContext) -> bool:
        """
        Check if EXPLAIN ANALYZE was run.

        Args:
            context: Analysis context

        Returns:
            True if EXPLAIN ANALYZE plan is present
        """
        return self.get_explain_analyze_plan(context) is not None

    def has_query_metrics(self, context: AnalysisContext) -> bool:
        """
        Check if query metrics are available.

        Args:
            context: Analysis context

        Returns:
            True if query metrics are present
        """
        return self.get_query_metrics(context) is not None
