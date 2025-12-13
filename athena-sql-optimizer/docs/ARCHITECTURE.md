# Architecture Documentation

## Overview

The Athena SQL Optimizer is built with a modular, plugin-based architecture that separates concerns and enables easy extensibility. The system follows a data flow pattern where query analysis progresses through multiple specialized analyzers that each focus on a specific aspect of optimization.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Client (Claude)                       │
└──────────────────────────┬──────────────────────────────────────┘
                          │ MCP Protocol
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastMCP Server                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  MCP Tools                                                 │  │
│  │  • analyze_sql_query()                                     │  │
│  │  • estimate_query_cost()                                   │  │
│  │  • check_table_health()                                    │  │
│  └──────────────────────┬─────────────────────────────────────┘  │
└─────────────────────────┼─────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OptimizationEngine                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Orchestration Layer                                      │   │
│  │  • Query parsing                                          │   │
│  │  • Analyzer coordination                                  │   │
│  │  • Result aggregation                                     │   │
│  │  • Cost calculation                                       │   │
│  └───────────┬──────────────────────────┬───────────────────┘   │
└──────────────┼──────────────────────────┼───────────────────────┘
               │                          │
               ▼                          ▼
    ┌──────────────────┐      ┌──────────────────────┐
    │   Collectors      │      │     Analyzers        │
    │                   │      │                      │
    │  ┌────────────┐   │      │  ┌────────────────┐ │
    │  │  Athena    │   │      │  │  Explain       │ │
    │  │  Collector │   │      │  │  Analyzer      │ │
    │  └────────────┘   │      │  └────────────────┘ │
    │                   │      │                      │
    │  ┌────────────┐   │      │  ┌────────────────┐ │
    │  │   Glue     │   │      │  │  Partition     │ │
    │  │  Collector │   │      │  │  Analyzer      │ │
    │  └────────────┘   │      │  └────────────────┘ │
    │                   │      │                      │
    └─────────┬─────────┘      │  ┌────────────────┐ │
              │                │  │  Format        │ │
              │                │  │  Analyzer      │ │
              │                │  └────────────────┘ │
              │                │                      │
              │                │  ┌────────────────┐ │
              │                │  │  JOIN          │ │
              │                │  │  Analyzer      │ │
              │                │  └────────────────┘ │
              │                │                      │
              │                │  ┌────────────────┐ │
              │                │  │  Projection    │ │
              │                │  │  Analyzer      │ │
              │                │  └────────────────┘ │
              │                │                      │
              │                │  ┌────────────────┐ │
              │                │  │  Cost          │ │
              │                │  │  Analyzer      │ │
              │                │  └────────────────┘ │
              │                └──────────────────────┘
              ▼
┌─────────────────────────────────────┐
│          AWS Services                │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ Athena   │  │ Glue Data Catalog│ │
│  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────┘
```

## Core Components

### 1. FastMCP Server Layer (`server.py`)

**Responsibilities:**
- Expose MCP tools to clients
- Handle configuration initialization
- Convert tool results to JSON
- Error handling and user-friendly messages

**Key Functions:**
- `initialize_engine()`: Sets up OptimizationEngine with configuration
- `analyze_sql_query()`: Main analysis endpoint
- `estimate_query_cost()`: Cost estimation without execution
- `check_table_health()`: Table metadata and recommendations

### 2. Optimization Engine (`engine.py`)

**Responsibilities:**
- Query parsing and table name extraction
- Orchestrate data collection and analysis
- Aggregate recommendations from all analyzers
- Calculate total costs and savings
- Sort recommendations by severity and confidence

**Key Methods:**
- `analyze_query()`: Main analysis workflow
- `estimate_cost()`: Cost estimation from metadata
- `check_table_health()`: Table structure analysis
- `_extract_table_names()`: SQL parsing for table discovery
- `_sort_recommendations()`: Priority-based recommendation ordering

**Data Flow:**
1. Parse query to extract table names
2. Collect table metadata from Glue
3. Execute EXPLAIN (and optionally EXPLAIN ANALYZE)
4. Pass context to all enabled analyzers
5. Aggregate recommendations
6. Calculate costs and savings
7. Return structured result

### 3. Collectors Package

#### AthenaCollector (`collectors/athena_collector.py`)

**Responsibilities:**
- Execute queries in Athena
- Retrieve query execution metadata and metrics
- Get EXPLAIN plans
- Poll query status until completion

**Key Methods:**
- `execute_query()`: Run query and wait for completion
- `get_explain_plan()`: Execute EXPLAIN query
- `get_explain_analyze_plan()`: Execute EXPLAIN ANALYZE (incurs cost)
- `get_query_results()`: Fetch paginated query results

**AWS APIs Used:**
- `start_query_execution`
- `get_query_execution`
- `get_query_results`

#### GlueCollector (`collectors/glue_collector.py`)

**Responsibilities:**
- Fetch table metadata from Glue Data Catalog
- Retrieve partition information
- Get table statistics
- List databases and tables

**Key Methods:**
- `get_table_metadata()`: Comprehensive table information
- `get_partitions()`: Partition listing with metadata
- `get_table_statistics()`: Row counts, sizes, etc.
- `list_databases()`: Catalog exploration
- `list_tables()`: Database exploration

**AWS APIs Used:**
- `get_table`
- `get_partitions`
- `get_databases`
- `get_tables`

### 4. Analyzers Package

All analyzers inherit from `BaseAnalyzer` and implement the `analyze()` method.

#### ExplainAnalyzer (`analyzers/explain_analyzer.py`)

**Focus:** EXPLAIN plan analysis

**Detects:**
- Full table scans without filters
- CROSS JOINs
- Complex distributed queries (many fragments)
- Expensive operations (SORT without LIMIT, DISTINCT, Window functions)

**Output:** Recommendations for query restructuring

#### PartitionAnalyzer (`analyzers/partition_analyzer.py`)

**Focus:** Partition usage and optimization

**Detects:**
- Tables without partitions
- Missing partition filters on partitioned tables
- Inefficient partition filters (LIKE, range operators)
- Unused partition columns

**Output:** Partition-related recommendations with CTAS examples

#### FormatAnalyzer (`analyzers/format_analyzer.py`)

**Focus:** Table storage format

**Detects:**
- Non-columnar formats (CSV, JSON, TEXT)
- Uncompressed data
- Suboptimal column ordering

**Output:** Format conversion recommendations with SQL examples

#### JoinAnalyzer (`analyzers/join_analyzer.py`)

**Focus:** JOIN operation optimization

**Detects:**
- Multiple JOINs (complexity warning)
- OUTER JOINs (may be avoidable)
- Non-equality JOIN conditions
- JOIN order issues

**Output:** JOIN optimization suggestions

#### ProjectionAnalyzer (`analyzers/projection_analyzer.py`)

**Focus:** Column selection optimization

**Detects:**
- SELECT * usage
- COUNT(*) vs COUNT(1)
- Complex expressions in SELECT
- DISTINCT on many columns

**Output:** Column selection recommendations

#### CostAnalyzer (`analyzers/cost_analyzer.py`)

**Focus:** Query cost calculation and warnings

**Detects:**
- High-cost queries (> $1.00)
- Moderate-cost queries (> $0.10)
- Partitioned tables without filters

**Output:** Cost warnings with savings potential

### 5. Models Package (`models/types.py`)

**Pydantic Models:**
- `OptimizerConfig`: Configuration parameters
- `Recommendation`: Single optimization recommendation
- `TableMetadata`: Glue table information
- `QueryMetrics`: Query execution statistics
- `ExplainPlan`: Parsed EXPLAIN output
- `AnalysisResult`: Complete analysis result

**Enums:**
- `Severity`: CRITICAL, HIGH, MEDIUM, LOW, INFO
- `Category`: PARTITION, FORMAT, JOIN, PROJECTION, COST, etc.
- `Effort`: LOW, MEDIUM, HIGH

## Data Flow

### Query Analysis Flow

```
1. User calls analyze_sql_query(query)
                ↓
2. Engine extracts table names from SQL
                ↓
3. For each table, collect metadata from Glue
                ↓
4. Execute EXPLAIN query in Athena
                ↓
5. (Optional) Execute EXPLAIN ANALYZE
                ↓
6. Build context object with all collected data
                ↓
7. Pass context to each analyzer
                ↓
8. Each analyzer returns list of recommendations
                ↓
9. Engine aggregates recommendations
                ↓
10. Calculate costs from metrics + percentages
                ↓
11. Sort recommendations by severity + confidence
                ↓
12. Return AnalysisResult as JSON
```

### Context Object Structure

```python
context = {
    "query": str,                      # Original SQL query
    "database": str,                   # Database name
    "table_metadata": {                # Table info from Glue
        "table_name": TableMetadata
    },
    "explain_plan": str,               # Raw EXPLAIN output
    "parsed_explain_plan": ExplainPlan,# Parsed plan
    "explain_analyze_plan": str,       # EXPLAIN ANALYZE output
    "query_metrics": QueryMetrics      # Execution metrics
}
```

## Extensibility Points

### Adding a New Analyzer

1. Create class inheriting from `BaseAnalyzer`
2. Implement `analyze(context) -> list[Recommendation]`
3. Implement `name` property
4. Register in `OptimizationEngine.__init__()`

```python
from .analyzers.base import BaseAnalyzer
from .models import Recommendation

class MyAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "MyAnalyzer"

    def analyze(self, context: dict) -> list[Recommendation]:
        # Analysis logic
        return recommendations
```

### Adding a New Collector

1. Create class with AWS client initialization
2. Implement data collection methods
3. Use in `OptimizationEngine`

```python
class MyCollector:
    def __init__(self, config: OptimizerConfig):
        session = boto3.Session(...)
        self.client = session.client("service")

    def collect_data(self, params):
        # AWS API calls
        return data
```

### Adding a New MCP Tool

1. Define function in `server.py`
2. Decorate with `@mcp.tool()`
3. Call `OptimizationEngine` methods
4. Return JSON string

```python
@mcp.tool()
def my_tool(param: str) -> str:
    engine = get_engine()
    result = engine.my_method(param)
    return json.dumps(result, indent=2)
```

## Configuration Management

Configuration flows through the system:

```
MCP Config → OptimizerConfig → Engine → Collectors/Analyzers
```

Critical parameters:
- `workgroup`: Athena workgroup (required)
- `s3_output_location`: Query results location (required)
- `aws_profile`: AWS credentials profile
- `region`: AWS region
- `database`: Default database
- `run_explain_analyze`: Execute queries for real metrics

## Error Handling Strategy

**Graceful Degradation:**
- If metadata collection fails → Continue without table info
- If EXPLAIN fails → Skip explain analyzer
- If one analyzer fails → Other analyzers continue
- Errors logged, not raised to user

**User-Facing Errors:**
- Missing required config → Clear error message
- AWS permission errors → Suggest permissions needed
- Query syntax errors → Pass through Athena error

## Performance Considerations

**Optimization Strategies:**
1. Parallel data collection where possible
2. Lazy loading of expensive operations (EXPLAIN ANALYZE)
3. Pagination for large result sets
4. Timeouts to prevent hanging
5. Caching potential (not yet implemented)

**Cost Awareness:**
- EXPLAIN is cheap (metadata only)
- EXPLAIN ANALYZE costs money (executes query)
- Default: Don't run EXPLAIN ANALYZE unless requested

## Security Considerations

**AWS Credentials:**
- Uses standard AWS credential chain
- Supports named profiles
- No credentials stored in code

**Required IAM Permissions:**
- `athena:StartQueryExecution`
- `athena:GetQueryExecution`
- `athena:GetQueryResults`
- `glue:GetDatabase`
- `glue:GetTable`
- `glue:GetPartitions`
- `s3:GetObject`
- `s3:PutObject`
- `s3:ListBucket`

## Testing Architecture

**Test Structure:**
```
tests/
├── unit/              # Unit tests with mocking
│   ├── test_models.py
│   ├── test_analyzers.py
│   ├── test_collectors.py
│   └── test_engine.py
└── integration/       # End-to-end tests
    └── test_end_to_end.py
```

**Testing Strategy:**
- Unit tests mock AWS clients
- Integration tests mock at AWS API level
- Fixtures provide reusable test data
- 73 tests covering all components

## Future Architecture Enhancements

**Planned:**
1. Query history tracking (database/cache layer)
2. Materialized view recommendations (new analyzer)
3. Cross-query optimization (batch analysis)
4. CloudWatch metrics integration (new collector)
5. Caching layer for table metadata
6. Async collectors for parallel data fetching
7. Plugin system for custom analyzers
8. Web UI for visualization

**Scalability:**
- Current: Single-query synchronous analysis
- Future: Batch processing, async operations
- Potential: Distributed analysis for large workloads
