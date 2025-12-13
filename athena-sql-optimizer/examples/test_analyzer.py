"""Example script demonstrating the Athena SQL Optimizer."""

import json
from athena_optimizer import OptimizationEngine, OptimizerConfig


def main():
    """Run example analyses."""

    # Configuration
    config = OptimizerConfig(
        aws_profile="default",  # Use your AWS profile
        region="eu-west-1",
        workgroup="primary",  # Your Athena workgroup
        s3_output_location="s3://your-bucket/athena-results/",  # Your S3 path
        catalog="AwsDataCatalog",
        database="my_database",  # Your database
        run_explain_analyze=False,  # Set to True to execute queries
        athena_cost_per_tb=5.0,
        timeout_seconds=300
    )

    # Initialize engine
    engine = OptimizationEngine(config)

    # Example 1: Analyze a query with SELECT *
    print("=" * 80)
    print("Example 1: Query with SELECT *")
    print("=" * 80)

    query1 = """
    SELECT *
    FROM sales_data
    WHERE region = 'US'
    """

    try:
        result1 = engine.analyze_query(query1)
        print(f"\nQuery: {result1.query}")
        print(f"\nTotal Recommendations: {len(result1.recommendations)}")

        for i, rec in enumerate(result1.recommendations, 1):
            print(f"\n{i}. [{rec.severity}] {rec.title}")
            print(f"   {rec.description}")
            if rec.savings_percentage:
                print(f"   Potential Savings: {rec.savings_percentage:.1f}%")

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Analyze a query on partitioned table
    print("\n" + "=" * 80)
    print("Example 2: Query on Partitioned Table")
    print("=" * 80)

    query2 = """
    SELECT customer_id, SUM(amount) as total
    FROM transactions
    WHERE amount > 100
    GROUP BY customer_id
    """

    try:
        result2 = engine.analyze_query(query2)
        print(f"\nQuery: {result2.query}")
        print(f"\nRecommendations: {len(result2.recommendations)}")

        # Show CRITICAL and HIGH severity only
        critical_recs = [r for r in result2.recommendations
                        if r.severity in ["CRITICAL", "HIGH"]]

        for rec in critical_recs:
            print(f"\n[{rec.severity}] {rec.title}")
            print(f"{rec.description}")
            print(f"Action Plan:")
            for step in rec.action_plan:
                print(f"  - {step}")

            if rec.code_example:
                print(f"\nExample:\n{rec.code_example}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Estimate cost without execution
    print("\n" + "=" * 80)
    print("Example 3: Cost Estimation")
    print("=" * 80)

    query3 = """
    SELECT *
    FROM large_table
    JOIN another_table ON large_table.id = another_table.id
    """

    try:
        cost_estimate = engine.estimate_cost(query3)
        print(json.dumps(cost_estimate, indent=2))

    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Check table health
    print("\n" + "=" * 80)
    print("Example 4: Table Health Check")
    print("=" * 80)

    try:
        health = engine.check_table_health("my_database", "sales_data")

        print(f"Table: {health['database']}.{health['table']}")
        print(f"Format: {health['format']}")
        print(f"Location: {health['location']}")
        print(f"\nColumns ({len(health['columns'])}):")
        for col in health['columns'][:5]:  # Show first 5
            print(f"  - {col['name']}: {col['type']}")

        if health['partition_info']:
            print(f"\nPartitions:")
            print(f"  Keys: {health['partition_info']['partition_keys']}")
            print(f"  Count: {health['partition_info']['partition_count']}")

        print(f"\nRecommendations: {len(health['recommendations'])}")
        for rec in health['recommendations']:
            print(f"  [{rec['severity']}] {rec['message']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
