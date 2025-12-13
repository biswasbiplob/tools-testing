"""Cost calculation utilities for Athena queries."""

from typing import Optional
from dataclasses import dataclass

from .constants import (
    BYTES_PER_TB,
    MIN_COST_MULTIPLIER,
    MAX_COST_MULTIPLIER,
    PERCENTAGE_MULTIPLIER,
)
from .models import QueryMetrics, Recommendation


@dataclass
class CostEstimate:
    """Cost estimation results."""

    estimated_cost_usd: float
    estimated_scan_tb: float
    min_cost_usd: float
    max_cost_usd: float
    table_sizes: dict[str, float]
    assumptions: list[str]


@dataclass
class CostAnalysis:
    """Complete cost analysis results."""

    total_current_cost_usd: float
    total_optimized_cost_usd: float
    total_savings_usd: float
    total_savings_percentage: float


class CostCalculator:
    """
    Centralized cost calculation for Athena queries.

    This class handles all cost-related calculations including:
    - Converting bytes to TB
    - Calculating query costs from metrics
    - Estimating costs from table sizes
    - Computing savings and optimization potential
    """

    def __init__(self, cost_per_tb: float = 5.0):
        """
        Initialize cost calculator.

        Args:
            cost_per_tb: Cost per TB of data scanned in USD (default: 5.0)
        """
        self.cost_per_tb = cost_per_tb

    def bytes_to_tb(self, bytes_value: int) -> float:
        """
        Convert bytes to terabytes.

        Args:
            bytes_value: Number of bytes

        Returns:
            Value in terabytes
        """
        return bytes_value / BYTES_PER_TB

    def calculate_cost_from_bytes(self, bytes_scanned: int) -> float:
        """
        Calculate Athena query cost from bytes scanned.

        Args:
            bytes_scanned: Number of bytes scanned

        Returns:
            Cost in USD
        """
        tb_scanned = self.bytes_to_tb(bytes_scanned)
        return tb_scanned * self.cost_per_tb

    def calculate_cost_from_metrics(self, metrics: QueryMetrics) -> tuple[float, float]:
        """
        Calculate cost from query execution metrics.

        Args:
            metrics: Query execution metrics

        Returns:
            Tuple of (cost_usd, tb_scanned)
        """
        tb_scanned = self.bytes_to_tb(metrics.data_scanned_bytes)
        cost_usd = tb_scanned * self.cost_per_tb
        return cost_usd, tb_scanned

    def estimate_cost_from_table_sizes(
        self,
        table_sizes: dict[str, int]
    ) -> CostEstimate:
        """
        Estimate query cost based on table sizes.

        This provides a range estimate assuming different optimization scenarios:
        - min_cost: With optimal partitioning, column projection, and format
        - max_cost: Full table scan without optimizations

        Args:
            table_sizes: Dictionary mapping table names to sizes in bytes

        Returns:
            CostEstimate with estimated costs and ranges
        """
        total_size_bytes = sum(table_sizes.values())
        estimated_scan_tb = self.bytes_to_tb(total_size_bytes)
        estimated_cost = estimated_scan_tb * self.cost_per_tb

        # Calculate range based on typical optimizations
        min_cost = estimated_cost * MIN_COST_MULTIPLIER  # 10% with optimal optimizations
        max_cost = estimated_cost * MAX_COST_MULTIPLIER  # 100% full table scan

        # Convert table sizes to TB for readability
        table_sizes_tb = {
            name: self.bytes_to_tb(size)
            for name, size in table_sizes.items()
        }

        assumptions = [
            "Assumes full table scan without partition filters",
            f"Min cost ({MIN_COST_MULTIPLIER * 100:.0f}%): With partition pruning, column projection, and columnar format",
            f"Max cost ({MAX_COST_MULTIPLIER * 100:.0f}%): Full table scan on all columns",
            "Actual cost depends on WHERE clauses, selected columns, and table format"
        ]

        return CostEstimate(
            estimated_cost_usd=estimated_cost,
            estimated_scan_tb=estimated_scan_tb,
            min_cost_usd=min_cost,
            max_cost_usd=max_cost,
            table_sizes=table_sizes_tb,
            assumptions=assumptions
        )

    def analyze_costs_from_recommendations(
        self,
        recommendations: list[Recommendation],
        metrics: Optional[QueryMetrics] = None
    ) -> CostAnalysis:
        """
        Analyze total costs and savings from recommendations.

        Args:
            recommendations: List of recommendations with cost data
            metrics: Optional query metrics for actual cost calculation

        Returns:
            CostAnalysis with totals and savings
        """
        total_current_cost = 0.0
        total_optimized_cost = 0.0
        total_savings = 0.0

        # Aggregate costs from recommendations
        for rec in recommendations:
            if rec.current_cost_usd:
                total_current_cost += rec.current_cost_usd
            if rec.optimized_cost_usd:
                total_optimized_cost += rec.optimized_cost_usd
            if rec.savings_usd:
                total_savings += rec.savings_usd

        # Calculate cost from metrics if available
        if metrics and total_current_cost == 0:
            metrics_cost, _ = self.calculate_cost_from_metrics(metrics)
            total_current_cost = metrics_cost

        # Estimate optimized cost from savings percentages if needed
        if total_current_cost > 0 and total_optimized_cost == 0:
            max_savings_percentage = 0.0
            for rec in recommendations:
                if rec.savings_percentage and rec.savings_percentage > max_savings_percentage:
                    max_savings_percentage = rec.savings_percentage

            if max_savings_percentage > 0:
                total_optimized_cost = total_current_cost * (
                    1 - max_savings_percentage / PERCENTAGE_MULTIPLIER
                )
                total_savings = total_current_cost - total_optimized_cost

        # Calculate savings percentage
        total_savings_percentage = (
            (total_savings / total_current_cost * PERCENTAGE_MULTIPLIER)
            if total_current_cost > 0
            else 0.0
        )

        return CostAnalysis(
            total_current_cost_usd=total_current_cost,
            total_optimized_cost_usd=total_optimized_cost,
            total_savings_usd=total_savings,
            total_savings_percentage=total_savings_percentage
        )

    def calculate_savings(
        self,
        current_cost: float,
        savings_percentage: float
    ) -> tuple[float, float]:
        """
        Calculate optimized cost and savings from percentage.

        Args:
            current_cost: Current cost in USD
            savings_percentage: Savings percentage (0-100)

        Returns:
            Tuple of (optimized_cost_usd, savings_usd)
        """
        optimized_cost = current_cost * (1 - savings_percentage / PERCENTAGE_MULTIPLIER)
        savings = current_cost - optimized_cost
        return optimized_cost, savings
