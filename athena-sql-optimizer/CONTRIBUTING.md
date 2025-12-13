# Contributing to Athena SQL Optimizer

Thank you for your interest in contributing! This document provides guidelines and information for extending the optimizer.

## Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd athena-sql-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt
```

## Architecture Overview

### Core Components

1. **Models** (`models/types.py`)
   - Pydantic models for type safety
   - Data structures for recommendations, metrics, etc.

2. **Collectors** (`collectors/`)
   - Interface with AWS services
   - `AthenaCollector`: Query execution and metrics
   - `GlueCollector`: Table metadata and partitions

3. **Analyzers** (`analyzers/`)
   - Plugin-based analysis system
   - Each analyzer focuses on specific optimization area
   - Inherit from `BaseAnalyzer`

4. **Engine** (`engine.py`)
   - Orchestrates all analyzers
   - Manages data flow between collectors and analyzers
   - Calculates aggregate metrics

5. **Server** (`server.py`)
   - FastMCP server implementation
   - Exposes tools to MCP clients

## Adding a New Analyzer

### Step 1: Create Analyzer Class

Create a new file in `src/athena_optimizer/analyzers/`:

```python
"""Description of what this analyzer does."""

from typing import Any
from ..models import Recommendation, Severity, Category, Effort
from .base import BaseAnalyzer


class MyAnalyzer(BaseAnalyzer):
    """Analyzes XYZ for optimization opportunities."""

    @property
    def name(self) -> str:
        return "MyAnalyzer"

    def analyze(self, context: dict[str, Any]) -> list[Recommendation]:
        """Analyze XYZ in the query."""
        recommendations = []

        # Access query and metadata
        query = context.get("query", "")
        table_metadata = context.get("table_metadata", {})
        explain_plan = context.get("parsed_explain_plan")
        query_metrics = context.get("query_metrics")

        # Your analysis logic here
        if self._detect_issue(query):
            recommendations.append(Recommendation(
                severity=Severity.HIGH,
                category=Category.QUERY_PATTERN,
                title="Issue Detected",
                description="Detailed description of the issue",
                confidence=0.9,
                effort=Effort.LOW,
                action_plan=[
                    "Step 1 to fix",
                    "Step 2 to fix"
                ],
                code_example="FIXED SQL HERE",
                references=[
                    "https://docs.aws.amazon.com/..."
                ]
            ))

        return recommendations

    def _detect_issue(self, query: str) -> bool:
        """Helper method to detect specific issue."""
        # Implementation
        return False
```

### Step 2: Register Analyzer

Add to `analyzers/__init__.py`:

```python
from .my_analyzer import MyAnalyzer

__all__ = [
    # ... existing
    "MyAnalyzer",
]
```

### Step 3: Add to Engine

In `engine.py`, add to the `__init__` method:

```python
self.analyzers = [
    ExplainAnalyzer(config),
    PartitionAnalyzer(config),
    # ... existing analyzers
    MyAnalyzer(config),  # Add your analyzer
    CostAnalyzer(config),  # Cost should stay last
]
```

## Adding a New Collector

### Step 1: Create Collector Class

Create a new file in `src/athena_optimizer/collectors/`:

```python
"""Collector for AWS Service XYZ."""

from typing import Optional
import boto3
from botocore.exceptions import ClientError

from ..models import OptimizerConfig


class XYZCollector:
    """Collects data from AWS XYZ service."""

    def __init__(self, config: OptimizerConfig):
        """Initialize XYZ collector."""
        self.config = config

        session_kwargs = {"region_name": config.region}
        if config.aws_profile:
            session_kwargs["profile_name"] = config.aws_profile

        session = boto3.Session(**session_kwargs)
        self.client = session.client("xyz")

    def get_data(self, resource_id: str) -> dict:
        """Get data from XYZ service."""
        try:
            response = self.client.describe_resource(
                ResourceId=resource_id
            )
            return response
        except ClientError as e:
            raise RuntimeError(f"Failed to get data: {e}") from e
```

### Step 2: Use in Engine

Add to `engine.py`:

```python
from .collectors import XYZCollector

class OptimizationEngine:
    def __init__(self, config: OptimizerConfig):
        # ... existing collectors
        self.xyz = XYZCollector(config)
```

## Adding a New Model

Add to `models/types.py`:

```python
class NewModel(BaseModel):
    """Description of the model."""
    field1: str
    field2: int
    optional_field: Optional[float] = None
```

Export in `models/__init__.py`:

```python
from .types import NewModel

__all__ = [
    # ... existing
    "NewModel",
]
```

## Adding a New MCP Tool

In `server.py`:

```python
@mcp.tool()
def my_new_tool(
    param1: str,
    param2: Optional[int] = None
) -> str:
    """
    Description of what this tool does.

    Args:
        param1: Description of param1
        param2: Description of param2 (optional)

    Returns:
        JSON string with results
    """
    try:
        engine = get_engine()

        # Your tool logic here
        result = engine.some_method(param1, param2)

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "failed"
        }, indent=2)
```

## Testing

### Manual Testing

1. Create a test script in `examples/`:

```python
from athena_optimizer import OptimizationEngine, OptimizerConfig

config = OptimizerConfig(
    workgroup="test",
    s3_output_location="s3://test/",
    # ... other config
)

engine = OptimizationEngine(config)
result = engine.analyze_query("SELECT * FROM test")
print(result)
```

2. Run the script:

```bash
python examples/my_test.py
```

### Unit Testing

Create tests in `tests/` directory:

```python
import pytest
from athena_optimizer.analyzers import MyAnalyzer
from athena_optimizer.models import OptimizerConfig

def test_my_analyzer():
    config = OptimizerConfig(
        workgroup="test",
        s3_output_location="s3://test/"
    )

    analyzer = MyAnalyzer(config)

    context = {
        "query": "SELECT * FROM test",
        "table_metadata": {}
    }

    recommendations = analyzer.analyze(context)

    assert len(recommendations) > 0
    assert recommendations[0].severity == "HIGH"
```

Run tests:

```bash
pytest tests/
```

## Code Style

- Use Black for formatting: `black src/`
- Use Ruff for linting: `ruff check src/`
- Use type hints for all function parameters and returns
- Follow PEP 8 naming conventions
- Write docstrings for all public functions and classes

## Best Practices

### Analyzer Design

1. **Single Responsibility**: Each analyzer should focus on one aspect
2. **No Side Effects**: Analyzers should only read data, not modify
3. **Confidence Scores**: Assign realistic confidence scores (0.0-1.0)
4. **Actionable Recommendations**: Include clear action plans
5. **Code Examples**: Provide SQL examples when possible

### Error Handling

1. **Graceful Degradation**: If one analyzer fails, others should continue
2. **Informative Messages**: Log warnings for debugging
3. **User-Friendly Errors**: Return clear error messages to users

### Performance

1. **Lazy Loading**: Only fetch data when needed
2. **Caching**: Cache table metadata when analyzing multiple queries
3. **Timeouts**: Respect timeout configurations
4. **Pagination**: Handle large result sets properly

## Recommendation Categories

Use these categories for consistency:

- `PARTITION`: Partitioning issues
- `FORMAT`: File format and storage
- `JOIN`: JOIN operation issues
- `PROJECTION`: Column selection issues
- `COST`: Cost-related recommendations
- `QUERY_PATTERN`: Query structure issues
- `STATISTICS`: Table statistics issues
- `COMPRESSION`: Compression recommendations

## Severity Levels

- `CRITICAL`: Severe issues (e.g., CROSS JOIN, missing partition filters)
- `HIGH`: Significant optimization opportunities (e.g., SELECT *, wrong format)
- `MEDIUM`: Moderate improvements (e.g., suboptimal patterns)
- `LOW`: Minor optimizations (e.g., style improvements)
- `INFO`: Informational (e.g., confirmation of good practices)

## Effort Levels

- `LOW`: < 1 hour (e.g., add WHERE clause)
- `MEDIUM`: 1-4 hours (e.g., query rewrite)
- `HIGH`: > 4 hours (e.g., table repartitioning)

## Documentation

- Update README.md for user-facing changes
- Update CONTRIBUTING.md for developer-facing changes
- Add examples to `examples/query_examples.md`
- Include docstrings in all code

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Add tests if applicable
4. Update documentation
5. Run linting and formatting
6. Submit PR with clear description
7. Address review comments

## Future Enhancement Ideas

- CloudWatch metrics integration
- Query history analysis
- Automatic query rewriting
- Cost tracking over time
- Benchmark suite
- Performance regression testing
- Web UI for visualization
- Slack/email notifications
- Multi-region support
- Custom cost models

## Questions?

- Check the README.md for usage questions
- Review existing analyzers for patterns
- Open an issue for bugs or feature requests

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
