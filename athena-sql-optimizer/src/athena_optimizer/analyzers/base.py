"""Base analyzer interface for extensible analysis plugins."""

from abc import ABC, abstractmethod
from typing import Any

from ..models import Recommendation, OptimizerConfig


class BaseAnalyzer(ABC):
    """Base class for all analyzers."""

    def __init__(self, config: OptimizerConfig):
        """Initialize analyzer with configuration."""
        self.config = config

    @abstractmethod
    def analyze(self, context: dict[str, Any]) -> list[Recommendation]:
        """
        Analyze the query and return recommendations.

        Args:
            context: Analysis context containing query, metadata, etc.

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
