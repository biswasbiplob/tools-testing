# Athena SQL Optimizer MCP Server

A comprehensive Model Context Protocol (MCP) server that analyzes AWS Athena SQL queries for performance issues, cost optimization opportunities, and provides actionable recommendations based on AWS best practices.

## Features

### 🔍 Comprehensive Analysis
- **EXPLAIN Plan Analysis**: Parses query execution plans to identify expensive operations
- **EXPLAIN ANALYZE Support**: Optional actual query execution for real metrics (configurable)
- **Partition Analysis**: Detects missing or inefficient partition filters
- **Format Optimization**: Recommends columnar formats (Parquet/ORC) for cost savings
- **JOIN Optimization**: Identifies problematic JOIN patterns and ordering issues
- **Projection Analysis**: Detects SELECT * and unnecessary column scanning
- **Cost Calculation**: Shows current costs and potential savings with optimizations

### 💰 Cost Optimization
- Calculates actual query costs based on data scanned
- Estimates optimized costs after applying recommendations
- Provides percentage-based savings projections
- Identifies queries that could benefit most from optimization

### 🎯 Actionable Recommendations
- Structured JSON output with severity levels (CRITICAL, HIGH, MEDIUM, LOW, INFO)
- Confidence scores for each recommendation
- Implementation effort estimates
- Step-by-step action plans
- SQL code examples for fixes
- AWS documentation references

### 🔌 Extensible Architecture
- Plugin-based analyzer system
- Easy to add new analysis types
- Modular collectors for AWS services
- Configurable recommendation scoring

## 📚 Documentation

- **[Getting Started Guide](docs/guides/GETTING_STARTED.md)** - Quick start and usage examples
- **[Architecture Documentation](docs/ARCHITECTURE.md)** - System design and components
- **[Flow Diagrams](docs/FLOW_DIAGRAM.md)** - Visual workflow documentation
- **[Contributing Guide](CONTRIBUTING.md)** - How to extend the optimizer

## Installation

### Prerequisites

- **Python 3.10+**
- **uv** (recommended) - Modern, fast Python package manager
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Quick Install with uv

```bash
# Clone the repository
cd athena-sql-optimizer

# Install all dependencies (creates venv automatically)
uv sync

# The project is now ready to use!
```

### Alternative: Install with pip

```bash
pip install -e .
```

## Configuration

### MCP Server Configuration

Add to your MCP settings file (e.g., `mcp_config.json`):

```json
{
  "mcpServers": {
    "athena-optimizer": {
      "command": "python",
      "args": [
        "-m",
        "athena_optimizer.server"
      ],
      "env": {
        "AWS_PROFILE": "your-profile-name",
        "AWS_REGION": "eu-west-1",
        "ATHENA_WORKGROUP": "primary",
        "ATHENA_S3_OUTPUT": "s3://your-bucket/athena-results/",
        "ATHENA_CATALOG": "AwsDataCatalog",
        "ATHENA_DATABASE": "default",
        "RUN_EXPLAIN_ANALYZE": "false",
        "ATHENA_COST_PER_TB": "5.0",
        "TIMEOUT_SECONDS": "300"
      }
    }
  }
}
```

### Configuration Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `AWS_PROFILE` | No | Default profile | AWS profile for authentication |
| `AWS_REGION` | No | `eu-west-1` | AWS region |
| `ATHENA_WORKGROUP` | **Yes** | - | Athena workgroup name |
| `ATHENA_S3_OUTPUT` | **Yes** | - | S3 location for query results |
| `ATHENA_CATALOG` | No | `AwsDataCatalog` | Glue Data Catalog name |
| `ATHENA_DATABASE` | No | - | Default database |
| `RUN_EXPLAIN_ANALYZE` | No | `false` | Run EXPLAIN ANALYZE (executes query) |
| `ATHENA_COST_PER_TB` | No | `5.0` | Cost per TB scanned in USD |
| `TIMEOUT_SECONDS` | No | `300` | Query timeout in seconds |

## Usage

### Tool: analyze_sql_query

Performs comprehensive analysis of a SQL query.

```python
# Example query
query = """
SELECT *
FROM my_database.large_table
WHERE created_date > '2024-01-01'
"""

# Analyze the query
result = analyze_sql_query(
    query=query,
    database="my_database",
    run_explain_analyze=False  # Set to True to execute query
)
```

**Output Structure:**
```json
{
  "query": "SELECT * FROM ...",
  "recommendations": [
    {
      "severity": "CRITICAL",
      "category": "PARTITION",
      "title": "Missing Partition Filters",
      "description": "...",
      "current_cost_usd": 10.50,
      "optimized_cost_usd": 2.10,
      "savings_usd": 8.40,
      "savings_percentage": 80.0,
      "confidence": 0.9,
      "effort": "LOW",
      "action_plan": ["Step 1", "Step 2"],
      "code_example": "SELECT ... WHERE partition_key = 'value'",
      "references": ["https://docs.aws.amazon.com/..."]
    }
  ],
  "total_current_cost_usd": 10.50,
  "total_optimized_cost_usd": 2.10,
  "total_savings_usd": 8.40,
  "total_savings_percentage": 80.0,
  "analysis_timestamp": "2024-01-15T10:30:00"
}
```

### Tool: estimate_query_cost

Estimates query cost without execution using table metadata.

```python
result = estimate_query_cost(
    query="SELECT * FROM large_table",
    database="my_database"
)
```

**Output:**
```json
{
  "estimated_cost_usd": 5.25,
  "estimated_scan_tb": 1.05,
  "cost_range_usd": {
    "min": 0.525,
    "max": 5.25
  },
  "table_sizes_bytes": {
    "large_table": 1099511627776
  },
  "note": "This is an estimate..."
}
```

### Tool: check_table_health

Analyzes table structure and provides optimization recommendations.

```python
result = check_table_health(
    database="my_database",
    table="large_table"
)
```

**Output:**
```json
{
  "database": "my_database",
  "table": "large_table",
  "format": "CSV",
  "location": "s3://bucket/path/",
  "columns": [...],
  "partition_info": {
    "partition_keys": ["year", "month"],
    "partition_count": 24,
    "sample_partitions": [["2024", "01"], ["2024", "02"]]
  },
  "recommendations": [
    {
      "severity": "HIGH",
      "category": "FORMAT",
      "message": "Table uses CSV format. Consider Parquet or ORC..."
    }
  ]
}
```

## Best Practices Checked

### Partitioning
- ✅ Tables are partitioned appropriately
- ✅ Partition filters are used in queries
- ✅ Efficient partition filter patterns (equality vs LIKE)
- ✅ Partition projection opportunities

### Table Format
- ✅ Columnar formats (Parquet/ORC) instead of CSV/JSON
- ✅ Compression is enabled
- ✅ Optimal column ordering
- ✅ File sizes and counts

### Query Patterns
- ✅ SELECT * usage
- ✅ Unnecessary column scanning
- ✅ Full table scans
- ✅ Complex expressions in SELECT

### JOIN Optimization
- ✅ JOIN order (small tables first)
- ✅ CROSS JOIN detection
- ✅ Non-equality JOIN conditions
- ✅ Multiple JOIN complexity

### Cost Optimization
- ✅ Data scan volume
- ✅ Query execution time
- ✅ Potential cost savings
- ✅ Cost-effectiveness of operations

## Architecture

```
athena-sql-optimizer/
├── src/athena_optimizer/
│   ├── models/           # Data models and types
│   │   └── types.py
│   ├── collectors/       # AWS service data collectors
│   │   ├── athena_collector.py
│   │   └── glue_collector.py
│   ├── analyzers/        # Analysis plugins
│   │   ├── base.py
│   │   ├── cost_analyzer.py
│   │   ├── explain_analyzer.py
│   │   ├── partition_analyzer.py
│   │   ├── format_analyzer.py
│   │   ├── join_analyzer.py
│   │   └── projection_analyzer.py
│   ├── engine.py         # Main optimization engine
│   └── server.py         # FastMCP server
└── pyproject.toml
```

## Extending the Analyzer

### Adding a New Analyzer

1. Create a new analyzer class:

```python
from athena_optimizer.analyzers.base import BaseAnalyzer
from athena_optimizer.models import Recommendation, Severity, Category

class CustomAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "CustomAnalyzer"

    def analyze(self, context: dict) -> list[Recommendation]:
        recommendations = []

        # Your analysis logic here

        return recommendations
```

2. Register in `engine.py`:

```python
from .analyzers import CustomAnalyzer

self.analyzers = [
    # ... existing analyzers
    CustomAnalyzer(config),
]
```

### Adding a New Collector

Follow the same pattern as `AthenaCollector` or `GlueCollector` to add collectors for CloudWatch, S3, etc.

## AWS Permissions Required

The AWS profile/credentials must have the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "athena:StartQueryExecution",
        "athena:GetQueryExecution",
        "athena:GetQueryResults",
        "glue:GetDatabase",
        "glue:GetTable",
        "glue:GetPartitions",
        "s3:GetBucketLocation",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:PutObject"
      ],
      "Resource": "*"
    }
  ]
}
```

## Testing

The project includes comprehensive test coverage with 73 tests covering all components.

### Run All Tests

```bash
# Using uv (recommended)
uv run pytest tests/ -v

# Quick run
uv run pytest tests/ -q

# With coverage
uv run pytest tests/ --cov=athena_optimizer --cov-report=html
```

### Run Specific Test Suites

```bash
# Unit tests only
uv run pytest tests/unit/ -v

# Integration tests only
uv run pytest tests/integration/ -v

# Specific test file
uv run pytest tests/unit/test_analyzers.py -v

# Specific test
uv run pytest tests/unit/test_analyzers.py::TestCostAnalyzer::test_high_cost_detection -v
```

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Unit tests with mocking
│   ├── test_models.py
│   ├── test_analyzers.py
│   ├── test_collectors.py
│   └── test_engine.py
└── integration/         # End-to-end tests
    └── test_end_to_end.py
```

## Troubleshooting

### "Optimizer not initialized" error
- Ensure `ATHENA_WORKGROUP` and `ATHENA_S3_OUTPUT` are set in configuration
- Check that environment variables are properly loaded

### "Failed to execute query" error
- Verify AWS credentials are configured
- Check workgroup exists and is accessible
- Ensure S3 output location is writable

### "Table not found" error
- Verify database and table names are correct
- Check Glue Data Catalog permissions
- Ensure catalog name is correct (default: AwsDataCatalog)

## Future Enhancements

- [ ] Query history analysis and comparison
- [ ] Automatic query rewriting suggestions
- [ ] Materialized view recommendations
- [ ] Workload-based optimization
- [ ] Schema evolution tracking
- [ ] Data freshness analysis
- [ ] Cross-query pattern detection
- [ ] CloudWatch metrics integration
- [ ] Cost tracking over time
- [ ] Bucketing recommendations

## Contributing

Contributions are welcome! Areas for improvement:
- Additional analyzers
- Better cost estimation algorithms
- Query rewriting capabilities
- Performance benchmarking
- Documentation improvements

## License

MIT License

## References

- [AWS Athena Performance Tuning](https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html)
- [Athena Best Practices](https://docs.aws.amazon.com/athena/latest/ug/best-practices.html)
- [Columnar Storage Formats](https://docs.aws.amazon.com/athena/latest/ug/columnar-storage.html)
- [Partitioning Data](https://docs.aws.amazon.com/athena/latest/ug/partitions.html)
- [Model Context Protocol](https://modelcontextprotocol.io/)
