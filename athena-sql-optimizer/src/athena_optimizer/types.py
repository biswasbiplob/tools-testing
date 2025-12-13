"""Type definitions for type-safe analysis context."""

from typing import TypedDict, Optional, Dict
from .models import TableMetadata, QueryMetrics


class AnalysisContext(TypedDict, total=False):
    """
    Type-safe analysis context passed to analyzers.

    Using TypedDict with total=False allows optional keys while
    maintaining type safety for values. This provides:
    - IDE autocomplete for context keys
    - Type checking with mypy
    - Self-documenting code
    - Catch bugs at development time

    All fields are technically optional since they're populated
    progressively during analysis.
    """
    # Core fields (always present after initialization)
    query: str
    database: str
    table_metadata: Dict[str, TableMetadata]

    # Optional fields (may not be present depending on analysis stage)
    query_metrics: Optional[QueryMetrics]
    explain_plan: Optional[str]
    explain_analyze_plan: Optional[str]
    parsed_explain_plan: Optional[dict]
