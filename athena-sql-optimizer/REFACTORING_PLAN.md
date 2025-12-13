# Refactoring Implementation Plan

This document provides specific, actionable steps to fix the issues identified in DESIGN_EVALUATION.md.

## Phase 1: Critical Fixes (Week 1)

### Fix 1: Remove Global State

**Current Code** (`server.py`):
```python
_engine: Optional[OptimizationEngine] = None

def get_engine() -> OptimizationEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Optimizer not initialized")
    return _engine
```

**Refactored Code**:
```python
# server.py
from contextlib import contextmanager
from typing import Generator

class EngineContext:
    """Context manager for engine lifecycle."""
    def __init__(self):
        self.engine: Optional[OptimizationEngine] = None

    @contextmanager
    def initialize(self, config: OptimizerConfig) -> Generator[OptimizationEngine, None, None]:
        """Initialize engine for request scope."""
        self.engine = OptimizationEngine(config)
        try:
            yield self.engine
        finally:
            self.engine.close()  # Clean up resources

# Create context instance
_engine_context = EngineContext()

def initialize_engine(config: OptimizerConfig):
    """Initialize engine with config."""
    with _engine_context.initialize(config) as engine:
        # Engine is now available in tools
        pass

@mcp.tool()
def analyze_sql_query(query: str, database: Optional[str] = None) -> str:
    """Analyze query using context engine."""
    if not _engine_context.engine:
        return json.dumps({"error": "Engine not initialized"}, indent=2)

    engine = _engine_context.engine
    result = engine.analyze_query(query, database)
    return json.dumps(result.model_dump(), indent=2, default=str)
```

### Fix 2: Add Resource Cleanup

**Create Base Collector**:
```python
# collectors/base.py
from abc import ABC, abstractmethod
from typing import Optional
import boto3

class BaseCollector(ABC):
    """Base class for AWS collectors with resource management."""

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self._session: Optional[boto3.Session] = None
        self._client = None
        self._closed = False

    @property
    def session(self) -> boto3.Session:
        """Get or create boto3 session."""
        if self._session is None:
            session_kwargs = {"region_name": self.config.region}
            if self.config.aws_profile:
                session_kwargs["profile_name"] = self.config.aws_profile
            self._session = boto3.Session(**session_kwargs)
        return self._session

    @property
    @abstractmethod
    def client_name(self) -> str:
        """AWS service name (e.g., 'athena', 'glue')."""
        pass

    @property
    def client(self):
        """Get or create service client."""
        if self._client is None:
            self._client = self.session.client(self.client_name)
        return self._client

    def close(self):
        """Clean up resources."""
        if not self._closed:
            if self._client:
                # boto3 clients don't have explicit close, but we can clear references
                self._client = None
            self._session = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
```

**Update AthenaCollector**:
```python
# collectors/athena_collector.py
from .base import BaseCollector

class AthenaCollector(BaseCollector):
    """Collects query execution data from Athena."""

    @property
    def client_name(self) -> str:
        return "athena"

    # Remove __init__ that creates client
    # Use self.client property from base class
```

**Update Engine**:
```python
# engine.py
class OptimizationEngine:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.athena = AthenaCollector(config)
        self.glue = GlueCollector(config)
        self.analyzers = [...]

    def close(self):
        """Clean up all resources."""
        self.athena.close()
        self.glue.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

### Fix 3: Implement Proper Logging

**Install structlog**:
```toml
# pyproject.toml
dependencies = [
    "structlog>=23.0.0",
    ...
]
```

**Create Logging Config**:
```python
# logging_config.py
import logging
import structlog
from typing import Optional

def configure_logging(level: str = "INFO", json_logs: bool = False):
    """Configure structured logging."""

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
    )

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger."""
    return structlog.get_logger(name)
```

**Update Engine**:
```python
# engine.py
from .logging_config import get_logger

class OptimizationEngine:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info(
            "Initializing optimization engine",
            region=config.region,
            workgroup=config.workgroup
        )

    def analyze_query(self, query: str, ...) -> AnalysisResult:
        self.logger.info(
            "Starting query analysis",
            query_length=len(query),
            database=database or self.config.database
        )

        try:
            # ... analysis code ...

            self.logger.info(
                "Analysis complete",
                recommendation_count=len(recommendations),
                duration_ms=duration
            )

        except Exception as e:
            self.logger.error(
                "Analysis failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    def _collect_table_metadata(self, table_name: str) -> Optional[TableMetadata]:
        try:
            metadata = self.glue.get_table_metadata(db, table)
            self.logger.debug("Table metadata fetched", table=table_name)
            return metadata
        except Exception as e:
            self.logger.warning(
                "Failed to fetch table metadata",
                table=table_name,
                error=str(e)
            )
            return None
```

### Fix 4: DRY Decorator for MCP Tools

**Create Decorator**:
```python
# server_utils.py
from functools import wraps
from typing import Callable, Any, Union
import json
from pydantic import BaseModel

def mcp_tool(func: Callable) -> Callable:
    """
    Decorator to handle common MCP tool patterns.

    - Gets engine from context
    - Handles errors consistently
    - Converts Pydantic models to JSON
    - Logs tool invocations
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> str:
        logger = get_logger(f"tool.{func.__name__}")

        logger.info("Tool invoked", arguments=kwargs)

        try:
            # Get engine
            if not _engine_context.engine:
                raise RuntimeError("Engine not initialized")

            engine = _engine_context.engine

            # Call the actual function
            result = func(engine, *args, **kwargs)

            # Convert result to JSON
            if isinstance(result, BaseModel):
                data = result.model_dump()
            elif isinstance(result, dict):
                data = result
            else:
                data = {"result": result}

            logger.info("Tool completed successfully")
            return json.dumps(data, indent=2, default=str)

        except Exception as e:
            logger.error(
                "Tool failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )

            return json.dumps({
                "error": str(e),
                "error_type": type(e).__name__,
                "status": "failed",
                "parameters": kwargs
            }, indent=2)

    return wrapper
```

**Update Tools**:
```python
# server.py
from .server_utils import mcp_tool

@mcp.tool()
@mcp_tool
def analyze_sql_query(
    engine: OptimizationEngine,
    query: str,
    database: Optional[str] = None,
    run_explain_analyze: Optional[bool] = None
) -> AnalysisResult:
    """
    Analyze an Athena SQL query.

    Note: Engine is injected by decorator.
    """
    return engine.analyze_query(query, database, run_explain_analyze)

@mcp.tool()
@mcp_tool
def estimate_query_cost(
    engine: OptimizationEngine,
    query: str,
    database: Optional[str] = None
) -> dict:
    """Estimate query cost."""
    return engine.estimate_cost(query, database)

@mcp.tool()
@mcp_tool
def check_table_health(
    engine: OptimizationEngine,
    database: str,
    table: str
) -> dict:
    """Check table health."""
    return engine.check_table_health(database, table)
```

---

## Phase 2: High Priority Fixes (Week 2)

### Fix 5: Replace Regex SQL Parsing

**Install sqlparse** (already in dependencies):
```python
# sql/parser.py
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Token
from sqlparse.tokens import Keyword, DML
from typing import Set, List
import logging

logger = logging.getLogger(__name__)

class SQLTableExtractor:
    """Extract table names from SQL using proper parsing."""

    @staticmethod
    def extract_tables(query: str) -> List[str]:
        """Extract all table names from SQL query."""
        try:
            return SQLTableExtractor._extract_with_sqlparse(query)
        except Exception as e:
            logger.warning(f"sqlparse failed, using regex fallback: {e}")
            return SQLTableExtractor._extract_with_regex(query)

    @staticmethod
    def _extract_with_sqlparse(query: str) -> List[str]:
        """Use sqlparse library."""
        parsed = sqlparse.parse(query)
        tables: Set[str] = set()

        for statement in parsed:
            tables.update(SQLTableExtractor._process_statement(statement))

        return sorted(tables)

    @staticmethod
    def _process_statement(statement) -> Set[str]:
        """Process a single SQL statement."""
        tables: Set[str] = set()
        from_seen = False

        for token in statement.tokens:
            if token.ttype is Keyword and token.value.upper() in ('FROM', 'JOIN'):
                from_seen = True
                continue

            if from_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        table_name = SQLTableExtractor._extract_table_name(identifier)
                        if table_name:
                            tables.add(table_name)
                elif isinstance(token, Identifier):
                    table_name = SQLTableExtractor._extract_table_name(token)
                    if table_name:
                        tables.add(table_name)

                if token.ttype is Keyword:
                    from_seen = False

        return tables

    @staticmethod
    def _extract_table_name(identifier) -> Optional[str]:
        """Extract clean table name from identifier."""
        name = identifier.get_real_name()
        if name:
            return name

        # Handle database.table notation
        parts = str(identifier).split()
        if parts:
            return parts[0].strip('`"[]')

        return None

    @staticmethod
    def _extract_with_regex(query: str) -> List[str]:
        """Fallback regex extraction."""
        import re

        # Remove comments
        query = re.sub(r'--[^\n]*', '', query)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)

        pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)?)'
        matches = re.finditer(pattern, query, re.IGNORECASE)

        tables = set()
        for match in matches:
            table = match.group(1)
            if table.lower() not in ['select', 'where', 'group', 'order']:
                tables.add(table)

        return sorted(tables)
```

**Update Engine**:
```python
# engine.py
from .sql.parser import SQLTableExtractor

class OptimizationEngine:
    def analyze_query(self, query: str, ...) -> AnalysisResult:
        # Use new parser
        table_names = SQLTableExtractor.extract_tables(query)
        self.logger.debug("Extracted tables", tables=table_names)
        ...
```

### Fix 6: Extract Magic Numbers

**Create Configuration Classes**:
```python
# config/thresholds.py
from dataclasses import dataclass

@dataclass
class CostThresholds:
    """Cost severity thresholds."""
    high_cost_usd: float = 1.0
    medium_cost_usd: float = 0.1
    high_savings_percentage: float = 70.0
    medium_savings_percentage: float = 50.0

@dataclass
class AnalyzerThresholds:
    """Thresholds for various analyzers."""
    cost: CostThresholds = CostThresholds()

    # Add more as needed
    max_join_count_warning: int = 3
    max_fragment_count_warning: int = 5
    max_distinct_columns: int = 5
```

**Update OptimizerConfig**:
```python
# models/types.py
class OptimizerConfig(BaseModel):
    # ... existing fields ...
    thresholds: AnalyzerThresholds = AnalyzerThresholds()
```

**Update CostAnalyzer**:
```python
# analyzers/cost_analyzer.py
class CostAnalyzer(BaseAnalyzer):
    def analyze(self, context: dict[str, Any]) -> list[Recommendation]:
        thresholds = self.config.thresholds.cost

        if current_cost > thresholds.high_cost_usd:
            severity = Severity.HIGH
        elif current_cost > thresholds.medium_cost_usd:
            severity = Severity.MEDIUM
        else:
            severity = Severity.INFO
```

### Fix 7: Implement Caching

**Create Caching Layer**:
```python
# cache.py
from datetime import datetime, timedelta
from typing import Optional, Callable, TypeVar, Generic, Dict, Tuple
from dataclasses import dataclass
import threading

T = TypeVar('T')

@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry."""
    value: T
    timestamp: datetime
    hits: int = 0

class TTLCache(Generic[T]):
    """Thread-safe TTL cache."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if datetime.now() - entry.timestamp > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None

            entry.hits += 1
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T) -> None:
        """Set value in cache."""
        with self._lock:
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=datetime.now()
            )

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
            }
```

**Update GlueCollector**:
```python
# collectors/glue_collector.py
from ..cache import TTLCache
from ..logging_config import get_logger

class GlueCollector(BaseCollector):
    def __init__(self, config: OptimizerConfig):
        super().__init__(config)
        self.logger = get_logger(self.__class__.__name__)
        self._metadata_cache = TTLCache[TableMetadata](ttl_seconds=300)  # 5 min

    def get_table_metadata(
        self,
        database: str,
        table: str,
        use_cache: bool = True
    ) -> TableMetadata:
        """Get table metadata with caching."""
        cache_key = f"{database}.{table}"

        # Try cache first
        if use_cache:
            cached = self._metadata_cache.get(cache_key)
            if cached:
                self.logger.debug("Cache hit", table=cache_key)
                return cached

        # Fetch from AWS
        self.logger.debug("Cache miss, fetching from AWS", table=cache_key)
        metadata = self._fetch_table_metadata(database, table)

        # Store in cache
        self._metadata_cache.set(cache_key, metadata)

        return metadata

    def _fetch_table_metadata(self, database: str, table: str) -> TableMetadata:
        """Actually fetch from AWS (extracted for clarity)."""
        # ... existing get_table_metadata code ...

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return self._metadata_cache.stats()
```

### Fix 8: Refactor Cost Calculation

**Create Cost Module**:
```python
# cost/calculator.py
from dataclasses import dataclass
from typing import List, Optional
from ..models import Recommendation, QueryMetrics, OptimizerConfig

@dataclass
class CostSummary:
    """Aggregated cost information."""
    current_cost_usd: float
    optimized_cost_usd: float
    savings_usd: float
    savings_percentage: float

    @property
    def has_savings(self) -> bool:
        return self.savings_usd > 0

class CostCalculator:
    """Handles all cost calculation logic."""

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.cost_per_tb = config.athena_cost_per_tb

    def calculate(
        self,
        recommendations: List[Recommendation],
        query_metrics: Optional[QueryMetrics]
    ) -> CostSummary:
        """Calculate cost summary from all available data."""

        # Try recommendation-based costs first
        rec_costs = self._from_recommendations(recommendations)
        if rec_costs.current_cost_usd > 0:
            return rec_costs

        # Fall back to metrics-based costs
        if query_metrics:
            return self._from_metrics(query_metrics, recommendations)

        return CostSummary(0.0, 0.0, 0.0, 0.0)

    def _from_recommendations(self, recommendations: List[Recommendation]) -> CostSummary:
        """Calculate from recommendation costs."""
        current = sum(r.current_cost_usd for r in recommendations if r.current_cost_usd)
        optimized = sum(r.optimized_cost_usd for r in recommendations if r.optimized_cost_usd)
        savings = sum(r.savings_usd for r in recommendations if r.savings_usd)

        percentage = (savings / current * 100) if current > 0 else 0.0

        return CostSummary(current, optimized, savings, percentage)

    def _from_metrics(
        self,
        metrics: QueryMetrics,
        recommendations: List[Recommendation]
    ) -> CostSummary:
        """Calculate from query metrics + recommendation percentages."""
        data_scanned_tb = metrics.data_scanned_bytes / (1024 ** 4)
        current_cost = data_scanned_tb * self.cost_per_tb

        # Find maximum savings percentage from recommendations
        max_savings_pct = max(
            (r.savings_percentage for r in recommendations if r.savings_percentage),
            default=0.0
        )

        if max_savings_pct > 0:
            optimized_cost = current_cost * (1 - max_savings_pct / 100)
            savings = current_cost - optimized_cost
            return CostSummary(current_cost, optimized_cost, savings, max_savings_pct)

        return CostSummary(current_cost, current_cost, 0.0, 0.0)
```

**Update Engine**:
```python
# engine.py
from .cost.calculator import CostCalculator

class OptimizationEngine:
    def analyze_query(self, ...) -> AnalysisResult:
        # ... run analyzers ...

        # Calculate costs
        cost_calc = CostCalculator(self.config)
        cost_summary = cost_calc.calculate(sorted_recommendations, context.get("query_metrics"))

        return AnalysisResult(
            query=query,
            recommendations=sorted_recommendations,
            total_current_cost_usd=cost_summary.current_cost_usd,
            total_optimized_cost_usd=cost_summary.optimized_cost_usd,
            total_savings_usd=cost_summary.savings_usd,
            total_savings_percentage=cost_summary.savings_percentage,
            ...
        )
```

---

## Phase 3: Medium Priority (Week 3)

### Fix 9: Add Async Support

**Make Methods Async**:
```python
# engine.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

class OptimizationEngine:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=5)
        # ... rest of init ...

    async def analyze_query_async(
        self,
        query: str,
        database: Optional[str] = None,
        run_explain_analyze: Optional[bool] = None
    ) -> AnalysisResult:
        """Async version of analyze_query."""
        db = database or self.config.database
        table_names = SQLTableExtractor.extract_tables(query)

        # Fetch metadata in parallel
        metadata_tasks = [
            self._fetch_table_metadata_async(name, db)
            for name in table_names
        ]

        metadata_results = await asyncio.gather(*metadata_tasks, return_exceptions=True)

        # Build context
        table_metadata = {}
        for name, result in zip(table_names, metadata_results):
            if not isinstance(result, Exception):
                table_metadata[name] = result

        context = {
            "query": query,
            "database": db,
            "table_metadata": table_metadata,
            ...
        }

        # Continue with synchronous analysis
        # (analyzers are CPU-bound, not I/O-bound)
        ...

    async def _fetch_table_metadata_async(
        self,
        table_name: str,
        database: str
    ) -> Optional[TableMetadata]:
        """Async table metadata fetching."""
        loop = asyncio.get_event_loop()

        try:
            if "." in table_name:
                db, table = table_name.split(".", 1)
            else:
                db, table = database, table_name

            # Run sync method in thread pool
            metadata = await loop.run_in_executor(
                self.executor,
                self.glue.get_table_metadata,
                db,
                table
            )

            return metadata
        except Exception as e:
            self.logger.warning(f"Failed to fetch metadata for {table_name}: {e}")
            return None

    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        super().close()
```

### Fix 10-15: See DESIGN_EVALUATION.md for detailed implementations

---

## Testing Strategy for Refactored Code

### Unit Tests
```python
# tests/unit/test_cache.py
def test_cache_basic():
    cache = TTLCache[str](ttl_seconds=1)
    cache.set("key", "value")
    assert cache.get("key") == "value"

    time.sleep(2)
    assert cache.get("key") is None

def test_cache_stats():
    cache = TTLCache[str]()
    cache.set("k1", "v1")
    cache.get("k1")  # hit
    cache.get("k2")  # miss

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5

# tests/unit/test_sql_parser.py
def test_extract_simple_table():
    query = "SELECT * FROM users"
    tables = SQLTableExtractor.extract_tables(query)
    assert tables == ["users"]

def test_extract_with_joins():
    query = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
    tables = SQLTableExtractor.extract_tables(query)
    assert set(tables) == {"users", "orders"}

def test_extract_with_cte():
    query = """
    WITH cte AS (SELECT * FROM table1)
    SELECT * FROM cte JOIN table2
    """
    tables = SQLTableExtractor.extract_tables(query)
    assert "table1" in tables
    assert "table2" in tables

# tests/unit/test_cost_calculator.py
def test_cost_from_metrics():
    config = OptimizerConfig(...)
    calc = CostCalculator(config)

    metrics = QueryMetrics(data_scanned_bytes=1024**4)  # 1 TB
    recommendations = [
        Recommendation(savings_percentage=80.0, ...)
    ]

    summary = calc.calculate(recommendations, metrics)
    assert summary.current_cost_usd == 5.0  # 1 TB * $5
    assert summary.savings_percentage == 80.0
```

### Integration Tests
```python
# tests/integration/test_async_analysis.py
@pytest.mark.asyncio
async def test_parallel_metadata_fetching():
    """Test that metadata is fetched in parallel."""
    engine = OptimizationEngine(config)

    query = """
    SELECT * FROM table1
    JOIN table2 ON table1.id = table2.id
    JOIN table3 ON table2.id = table3.id
    """

    start = time.time()
    result = await engine.analyze_query_async(query)
    duration = time.time() - start

    # With 3 tables and parallel fetching,
    # should be faster than 3x sequential
    assert duration < 3.0  # Adjust based on your AWS latency
```

---

## Migration Path

### Step 1: Parallel Development
- Create new modules alongside old ones
- Use feature flags to switch between implementations

### Step 2: Gradual Migration
- Migrate one component at a time
- Keep tests passing at each step
- Update integration tests

### Step 3: Cleanup
- Remove old implementations
- Update documentation
- Final test pass

---

## Success Metrics

Track these metrics before and after refactoring:

1. **Performance**
   - Query analysis time (should decrease with async)
   - Cache hit rate (should be >70%)

2. **Reliability**
   - Error rate
   - Resource leak detection (memory profiling)

3. **Code Quality**
   - Lines of code (should decrease)
   - Cyclomatic complexity (should decrease)
   - Test coverage (should stay >70%)

4. **Maintainability**
   - Time to add new analyzer (should decrease)
   - Number of failing tests when making changes (should decrease)
