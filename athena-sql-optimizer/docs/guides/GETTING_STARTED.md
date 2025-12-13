# Getting Started Guide

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.10 or higher**
   ```bash
   python --version  # Should be >= 3.10
   ```

2. **uv installed** (modern Python package manager)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or
   pip install uv
   ```

3. **AWS Account** with:
   - Access to Athena
   - Glue Data Catalog with tables
   - S3 bucket for query results

4. **AWS Credentials** configured
   ```bash
   aws configure
   # or use AWS_PROFILE environment variable
   ```

## Quick Start (5 minutes)

### 1. Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd athena-sql-optimizer

# Install dependencies with uv (fast!)
uv sync

# This creates a virtual environment and installs all dependencies
```

### 2. Configure MCP Server

Create or edit your MCP configuration file (e.g., `~/.config/claude/mcp_config.json`):

```json
{
  "mcpServers": {
    "athena-optimizer": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/athena-sql-optimizer",
        "python",
        "-m",
        "athena_optimizer.server"
      ],
      "env": {
        "AWS_PROFILE": "default",
        "AWS_REGION": "us-east-1",
        "ATHENA_WORKGROUP": "primary",
        "ATHENA_S3_OUTPUT": "s3://your-bucket/athena-results/",
        "ATHENA_CATALOG": "AwsDataCatalog",
        "ATHENA_DATABASE": "your_database",
        "RUN_EXPLAIN_ANALYZE": "false",
        "ATHENA_COST_PER_TB": "5.0"
      }
    }
  }
}
```

**Required Configuration:**
- `ATHENA_WORKGROUP`: Your Athena workgroup name
- `ATHENA_S3_OUTPUT`: S3 location for query results

**Optional Configuration:**
- `AWS_PROFILE`: AWS credentials profile (default: uses default profile)
- `AWS_REGION`: AWS region (default: `eu-west-1`)
- `ATHENA_DATABASE`: Default database (default: none)
- `RUN_EXPLAIN_ANALYZE`: Execute queries for real metrics (default: `false`)
- `ATHENA_COST_PER_TB`: Cost per TB scanned (default: `5.0` USD)

### 3. Test the Installation

#### Option A: Using Claude Desktop

1. Restart Claude Desktop to load the new MCP server
2. Ask Claude: "Can you analyze this SQL query: SELECT * FROM my_table"
3. Claude will use the `analyze_sql_query` tool automatically

#### Option B: Direct Testing (Without MCP)

```bash
# Create a test script
cat > test_optimizer.py << 'EOF'
from athena_optimizer import OptimizationEngine, OptimizerConfig

config = OptimizerConfig(
    workgroup="primary",
    s3_output_location="s3://your-bucket/results/",
    region="us-east-1"
)

engine = OptimizationEngine(config)

query = """
SELECT *
FROM my_database.large_table
WHERE created_date > '2024-01-01'
"""

# Analyze without executing (uses EXPLAIN only)
result = engine.analyze_query(query, database="my_database")

print(f"Found {len(result.recommendations)} recommendations")
for rec in result.recommendations[:3]:
    print(f"[{rec.severity}] {rec.title}")
EOF

# Run the test
uv run python test_optimizer.py
```

## Usage Examples

### Example 1: Analyze a Simple Query

```sql
SELECT *
FROM sales_data
WHERE region = 'US'
```

**Expected Recommendations:**
- **[HIGH]** SELECT * Detected: Select only needed columns (50% savings)
- **[MEDIUM]** Table Not Partitioned: Consider partitioning
- **[HIGH]** Non-Columnar Format: Convert to Parquet (70% savings)

### Example 2: Query with Missing Partition Filters

```sql
SELECT customer_id, SUM(amount) as total
FROM transactions
WHERE amount > 100
GROUP BY customer_id
```

If `transactions` is partitioned by `transaction_date`:

**Expected Recommendations:**
- **[CRITICAL]** Missing Partition Filters: Not filtering on transaction_date
- **[INFO]** Partitioned Table: Reminder to use filters

### Example 3: Complex JOIN Query

```sql
SELECT *
FROM orders o
CROSS JOIN customers c
LEFT JOIN products p ON o.product_id = p.id
```

**Expected Recommendations:**
- **[CRITICAL]** Cross Join Detected: Will create cartesian product
- **[HIGH]** SELECT * Detected: Scanning all columns
- **[MEDIUM]** Multiple JOINs: Consider optimization
- **[LOW]** OUTER JOIN: Verify LEFT JOIN is necessary

### Example 4: Cost Estimation

```python
# Without executing the query
result = engine.estimate_cost(
    "SELECT * FROM large_table",
    database="my_db"
)

print(f"Estimated cost: ${result['estimated_cost_usd']:.2f}")
print(f"Data to scan: {result['estimated_scan_tb']:.2f} TB")
```

### Example 5: Table Health Check

```python
health = engine.check_table_health("my_database", "my_table")

print(f"Format: {health['format']}")
print(f"Partitions: {health['partition_info']['partition_count']}")
print(f"Recommendations: {len(health['recommendations'])}")

for rec in health['recommendations']:
    print(f"- [{rec['severity']}] {rec['message']}")
```

## Understanding Recommendations

### Severity Levels

| Severity | Meaning | Example |
|----------|---------|---------|
| **CRITICAL** | Severe performance/cost issue | CROSS JOIN, missing partition filters |
| **HIGH** | Significant optimization opportunity | SELECT *, wrong format (CSV) |
| **MEDIUM** | Moderate improvement possible | Multiple JOINs, uncompressed |
| **LOW** | Minor optimization | OUTER JOIN usage |
| **INFO** | Informational, good practices | Using Parquet, has partitions |

### Cost Savings

Recommendations include potential savings:

```json
{
  "severity": "CRITICAL",
  "title": "Missing Partition Filters",
  "savings_percentage": 80.0,
  "current_cost_usd": 25.00,
  "optimized_cost_usd": 5.00,
  "savings_usd": 20.00
}
```

**Typical Savings:**
- Partition filters: 70-90%
- Format change (CSV → Parquet): 70-80%
- Column selection (SELECT * → specific): 50-70%
- Compression: 50%

## Common Workflows

### Workflow 1: Optimize an Expensive Query

1. **Analyze** the query
   ```python
   result = engine.analyze_query(query)
   ```

2. **Review** recommendations sorted by severity
   ```python
   for rec in result.recommendations:
       if rec.severity in ["CRITICAL", "HIGH"]:
           print(rec.title)
           print(rec.description)
           print(rec.action_plan)
           print(rec.code_example)
   ```

3. **Apply** fixes based on recommendations

4. **Re-analyze** to verify improvements

### Workflow 2: Audit a Database

```python
# List all tables
tables = engine.glue.list_tables("my_database")

# Check health of each
for table in tables:
    health = engine.check_table_health("my_database", table)

    if health['recommendations']:
        print(f"\n{table}:")
        for rec in health['recommendations']:
            print(f"  - {rec['message']}")
```

### Workflow 3: Cost Analysis Before Execution

```python
# Before running an expensive query
cost_est = engine.estimate_cost(query)

if cost_est['estimated_cost_usd'] > 10.0:
    print(f"⚠️  This query may cost ${cost_est['estimated_cost_usd']:.2f}")
    print("Consider optimization before running")

    # Analyze for optimization opportunities
    result = engine.analyze_query(query)
    # ... show recommendations
```

## Troubleshooting

### Issue: "Optimizer not initialized"

**Cause:** Missing required configuration parameters

**Solution:** Ensure `ATHENA_WORKGROUP` and `ATHENA_S3_OUTPUT` are set

```bash
export ATHENA_WORKGROUP=primary
export ATHENA_S3_OUTPUT=s3://my-bucket/results/
```

### Issue: "Failed to execute query"

**Possible Causes:**
1. AWS credentials not configured
2. Workgroup doesn't exist
3. S3 location not writable
4. Insufficient permissions

**Solution:**
```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify workgroup
aws athena get-work-group --work-group-name primary

# Check S3 permissions
aws s3 ls s3://my-bucket/results/
```

### Issue: "Table not found"

**Cause:** Table doesn't exist in Glue catalog

**Solution:**
- Verify table exists: `aws glue get-table --database-name db --name table`
- Check database name is correct
- Ensure catalog name is correct (default: `AwsDataCatalog`)

### Issue: High memory usage

**Cause:** Analyzing very complex queries or many tables

**Solution:**
- Use `estimate_cost()` instead of full analysis for bulk operations
- Increase timeout for complex queries
- Break down analysis into smaller chunks

## Best Practices

### 1. Start with EXPLAIN (not EXPLAIN ANALYZE)

```python
# Default: Only runs EXPLAIN (fast, free)
result = engine.analyze_query(query)

# Only use EXPLAIN ANALYZE when you need real metrics
result = engine.analyze_query(query, run_explain_analyze=True)
```

### 2. Focus on CRITICAL and HIGH Recommendations First

```python
critical_high = [r for r in result.recommendations
                 if r.severity in ["CRITICAL", "HIGH"]]

for rec in critical_high:
    print(f"{rec.title}: {rec.savings_percentage:.0f}% savings")
```

### 3. Use Table Health Checks Proactively

Regular audits prevent issues:

```python
# Weekly audit
for table in important_tables:
    health = engine.check_table_health(db, table)
    if health['recommendations']:
        send_alert(f"{table} needs attention")
```

### 4. Track Savings Over Time

```python
# Before optimization
before = engine.analyze_query(original_query, run_explain_analyze=True)

# After applying fixes
after = engine.analyze_query(optimized_query, run_explain_analyze=True)

savings = before.total_current_cost_usd - after.total_current_cost_usd
print(f"Saved: ${savings:.2f} ({savings/before.total_current_cost_usd*100:.0f}%)")
```

### 5. Document Optimization Decisions

```python
# Create optimization log
log = {
    "date": datetime.now(),
    "query_id": "Q123",
    "before_cost": before.total_current_cost_usd,
    "after_cost": after.total_current_cost_usd,
    "recommendations_applied": [r.title for r in before.recommendations[:3]],
    "savings": savings
}
```

## Next Steps

1. **Explore the Architecture**: Read [ARCHITECTURE.md](../ARCHITECTURE.md)
2. **Understand Flows**: Review [FLOW_DIAGRAM.md](../FLOW_DIAGRAM.md)
3. **Extend the Tool**: See [CONTRIBUTING.md](../../CONTRIBUTING.md)
4. **Run Tests**: `uv run pytest tests/`
5. **Read Example Queries**: See [examples/query_examples.md](../../examples/query_examples.md)

## Getting Help

- **Documentation**: Check the `docs/` folder
- **Examples**: See `examples/` for sample code
- **Issues**: Report bugs or request features in the GitHub issues
- **Tests**: Run `uv run pytest tests/ -v` to see all test cases

## Advanced Configuration

### Custom Cost Per TB

If your Athena pricing is different:

```python
config = OptimizerConfig(
    workgroup="primary",
    s3_output_location="s3://bucket/",
    athena_cost_per_tb=10.0  # Custom rate
)
```

### Longer Timeouts

For very complex queries:

```python
config = OptimizerConfig(
    workgroup="primary",
    s3_output_location="s3://bucket/",
    timeout_seconds=600  # 10 minutes
)
```

### Multiple AWS Profiles

```python
# Profile for production
prod_config = OptimizerConfig(
    aws_profile="prod",
    workgroup="prod-workgroup",
    s3_output_location="s3://prod-bucket/"
)

# Profile for development
dev_config = OptimizerConfig(
    aws_profile="dev",
    workgroup="dev-workgroup",
    s3_output_location="s3://dev-bucket/"
)
```

## Resources

- [AWS Athena Best Practices](https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html)
- [Partitioning Data](https://docs.aws.amazon.com/athena/latest/ug/partitions.html)
- [Columnar Storage](https://docs.aws.amazon.com/athena/latest/ug/columnar-storage.html)
- [Model Context Protocol](https://modelcontextprotocol.io/)

---

**Ready to optimize?** Start by running `uv run pytest tests/` to ensure everything works, then try analyzing your first query!
