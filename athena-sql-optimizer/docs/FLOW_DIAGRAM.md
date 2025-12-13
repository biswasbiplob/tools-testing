# Flow Diagrams

## 1. Complete Analysis Flow

```
┌────────────┐
│   User     │
│  (Claude)  │
└─────┬──────┘
      │
      │ 1. analyze_sql_query(query)
      ▼
┌──────────────────────────────────────────┐
│         FastMCP Server                    │
│  • Validates input                        │
│  • Initializes engine if needed           │
└────────────┬─────────────────────────────┘
             │
             │ 2. Call engine.analyze_query()
             ▼
┌──────────────────────────────────────────┐
│      OptimizationEngine                   │
│  Step 1: Parse Query                      │
│  ┌────────────────────────────────────┐   │
│  │ Extract table names via regex      │   │
│  │ Example: "FROM users" → ["users"]  │   │
│  └────────────────────────────────────┘   │
└────────────┬─────────────────────────────┘
             │
             │ 3. For each table
             ▼
┌──────────────────────────────────────────┐
│       GlueCollector                       │
│  Step 2: Collect Table Metadata          │
│  ┌────────────────────────────────────┐   │
│  │ AWS Glue API: get_table()          │   │
│  │ Returns:                           │   │
│  │  - Column definitions              │   │
│  │  - Partition keys                  │   │
│  │  - Storage format                  │   │
│  │  - Compression                     │   │
│  │  - Location                        │   │
│  └────────────────────────────────────┘   │
└────────────┬─────────────────────────────┘
             │
             │ 4. Build context
             ▼
┌──────────────────────────────────────────┐
│       AthenaCollector                     │
│  Step 3: Get EXPLAIN Plan                │
│  ┌────────────────────────────────────┐   │
│  │ Execute: "EXPLAIN <query>"         │   │
│  │ Returns: Query execution plan      │   │
│  └────────────────────────────────────┘   │
│                                           │
│  Step 4: (Optional) EXPLAIN ANALYZE       │
│  ┌────────────────────────────────────┐   │
│  │ Execute: "EXPLAIN ANALYZE <query>" │   │
│  │ Returns:                           │   │
│  │  - Actual execution metrics        │   │
│  │  - Data scanned                    │   │
│  │  - Execution time                  │   │
│  └────────────────────────────────────┘   │
└────────────┬─────────────────────────────┘
             │
             │ 5. Context = {query, metadata, explain}
             ▼
┌──────────────────────────────────────────┐
│     Analyzer Pipeline                     │
│  Step 5: Run All Analyzers                │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │ 1. ExplainAnalyzer                  │  │
│  │    • Parse EXPLAIN plan             │  │
│  │    • Detect table scans             │  │
│  │    • Find CROSS JOINs               │  │
│  │    • Check complexity               │  │
│  └─────────────────────────────────────┘  │
│           │                                │
│           │ Recommendations                │
│           ▼                                │
│  ┌─────────────────────────────────────┐  │
│  │ 2. PartitionAnalyzer                │  │
│  │    • Check if partitioned           │  │
│  │    • Verify filters used            │  │
│  │    • Validate filter efficiency     │  │
│  └─────────────────────────────────────┘  │
│           │                                │
│           │ Recommendations                │
│           ▼                                │
│  ┌─────────────────────────────────────┐  │
│  │ 3. FormatAnalyzer                   │  │
│  │    • Check table format             │  │
│  │    • Verify compression             │  │
│  │    • Recommend columnar             │  │
│  └─────────────────────────────────────┘  │
│           │                                │
│           │ Recommendations                │
│           ▼                                │
│  ┌─────────────────────────────────────┐  │
│  │ 4. JoinAnalyzer                     │  │
│  │    • Count JOINs                    │  │
│  │    • Check JOIN types               │  │
│  │    • Validate conditions            │  │
│  └─────────────────────────────────────┘  │
│           │                                │
│           │ Recommendations                │
│           ▼                                │
│  ┌─────────────────────────────────────┐  │
│  │ 5. ProjectionAnalyzer               │  │
│  │    • Detect SELECT *                │  │
│  │    • Check COUNT(*)                 │  │
│  │    • Find complex expressions       │  │
│  └─────────────────────────────────────┘  │
│           │                                │
│           │ Recommendations                │
│           ▼                                │
│  ┌─────────────────────────────────────┐  │
│  │ 6. CostAnalyzer                     │  │
│  │    • Calculate costs                │  │
│  │    • Classify by severity           │  │
│  │    • Add cost recommendations       │  │
│  └─────────────────────────────────────┘  │
│           │                                │
│           │ All Recommendations            │
│           ▼                                │
└───────────┬─────────────────────────────────┘
            │
            │ 6. Aggregate results
            ▼
┌──────────────────────────────────────────┐
│      OptimizationEngine                   │
│  Step 6: Aggregate & Calculate            │
│  ┌────────────────────────────────────┐   │
│  │ • Collect all recommendations      │   │
│  │ • Calculate total costs            │   │
│  │ • Calculate savings                │   │
│  │ • Sort by severity + confidence    │   │
│  └────────────────────────────────────┘   │
└────────────┬─────────────────────────────┘
             │
             │ 7. Return AnalysisResult
             ▼
┌──────────────────────────────────────────┐
│         FastMCP Server                    │
│  Step 7: Format Response                  │
│  ┌────────────────────────────────────┐   │
│  │ • Convert to JSON                  │   │
│  │ • Add metadata                     │   │
│  │ • Handle errors                    │   │
│  └────────────────────────────────────┘   │
└────────────┬─────────────────────────────┘
             │
             │ 8. JSON Response
             ▼
┌────────────┴──────┐
│   User (Claude)   │
│  • Display results│
│  • Show savings   │
│  • List actions   │
└───────────────────┘
```

## 2. Analyzer Decision Tree

```
                    ┌──────────────────┐
                    │  Context Object  │
                    │  (Query + Meta)  │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Has EXPLAIN │   │  Has Metadata │   │  Has Metrics  │
│     Plan?     │   │   for Tables? │   │  from Query?  │
└───┬───────────┘   └───┬───────────┘   └───┬───────────┘
    │                   │                   │
    │ YES               │ YES               │ YES
    ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ ExplainAnalyzer│  │ PartitionAnal. │  │  CostAnalyzer  │
│ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐ │
│ │Parse plan  │ │  │ │Check parts │ │  │ │Calc costs  │ │
│ │Find scans  │ │  │ │Check filter│ │  │ │Classify    │ │
│ │Detect XJoin│ │  │ │Recommend   │ │  │ │Warn        │ │
│ └────────────┘ │  │ └────────────┘ │  │ └────────────┘ │
└────────────────┘  │                │  └────────────────┘
                    │ FormatAnalyzer │
                    │ ┌────────────┐ │
                    │ │Check format│ │
                    │ │Check comp. │ │
                    │ │Recommend   │ │
                    │ └────────────┘ │
                    └────────────────┘

        ┌────────────────────────────────┐
        │    Query Text Analysis         │
        │    (Always Available)          │
        └────────┬───────────────────────┘
                 │
        ┌────────┼────────┐
        │                 │
        ▼                 ▼
┌────────────────┐  ┌────────────────┐
│  JoinAnalyzer  │  │ProjectionAnal. │
│ ┌────────────┐ │  │ ┌────────────┐ │
│ │Parse JOINs │ │  │ │Find SELECT*│ │
│ │Check types │ │  │ │Find COUNT* │ │
│ │Check conds │ │  │ │Check DIST. │ │
│ └────────────┘ │  │ └────────────┘ │
└────────────────┘  └────────────────┘
```

## 3. Recommendation Severity Flow

```
Analyzer Generates Recommendation
         │
         ▼
┌────────────────────┐
│  Assess Severity   │
└────────┬───────────┘
         │
    ┌────┴────┐
    │  Issue  │
    │  Type?  │
    └────┬────┘
         │
    ┌────┴────┬─────────┬─────────┬─────────┐
    │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼
┌─────────┐ ┌───────┐ ┌────────┐ ┌─────┐ ┌──────┐
│CRITICAL │ │ HIGH  │ │ MEDIUM │ │ LOW │ │ INFO │
└─────────┘ └───────┘ └────────┘ └─────┘ └──────┘
    │         │         │         │         │
    │         │         │         │         │
Examples:   │         │         │         │
- CROSS     │         │         │         │
  JOIN      │         │         │         │
- Missing   │         │         │         │
  partition │         │         │         │
  filters   │         │         │         │
            │         │         │         │
        Examples:     │         │         │
        - SELECT *    │         │         │
        - CSV format  │         │         │
        - Non-equal   │         │         │
          JOINs       │         │         │
                  Examples:     │         │
                  - Multiple    │         │
                    JOINs       │         │
                  - Uncompressed│         │
                  - No parts    │         │
                            Examples:     │
                            - OUTER JOINs │
                            - Complex expr│
                                      Examples:
                                      - Good format
                                      - Has partitions
                                      - Using filters
```

## 4. Cost Calculation Flow

```
┌────────────────────┐
│  Query Executed?   │
└────────┬───────────┘
         │
    ┌────┴────┐
    │  YES    │  NO
    ▼         ▼
┌──────────┐ ┌────────────────┐
│Has       │ │Estimate from   │
│Metrics   │ │table metadata  │
└────┬─────┘ └────┬───────────┘
     │            │
     │            │ get_table_statistics()
     │            │ total_size → estimated_scan
     │            │
     ▼            ▼
┌─────────────────────────────────────┐
│  Calculate Base Cost                │
│  cost = data_scanned_tb * $5/TB     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Collect Savings % from Recs        │
│  • Partition filter: 80%            │
│  • Format change: 70%               │
│  • Column selection: 50%            │
│  • Take MAX savings                 │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Calculate Optimized Cost           │
│  optimized = current * (1 - max%)   │
│  savings = current - optimized      │
│  savings_% = (savings/current)*100  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Return Cost Breakdown              │
│  {                                  │
│    current: $25.00,                 │
│    optimized: $5.00,                │
│    savings: $20.00,                 │
│    savings_pct: 80%                 │
│  }                                  │
└─────────────────────────────────────┘
```

## 5. Error Handling Flow

```
┌────────────────────┐
│  Operation Starts  │
└────────┬───────────┘
         │
         ▼
     try {
         │
    ┌────┴───────────────┐
    │   AWS API Call     │
    │   or Analysis      │
    └────┬───────────────┘
         │
    } catch {
         │
         ▼
    ┌─────────────────┐
    │  Error Type?    │
    └────┬────────────┘
         │
    ┌────┴────┬───────────┬──────────┐
    │         │           │          │
    ▼         ▼           ▼          ▼
┌─────────┐ ┌────────┐ ┌───────┐ ┌────────┐
│Table    │ │EXPLAIN │ │Timeout│ │Unknown │
│Not Found│ │Failed  │ │       │ │Error   │
└────┬────┘ └────┬───┘ └───┬───┘ └───┬────┘
     │           │         │         │
     │           │         │         │
     ▼           ▼         ▼         ▼
┌────────────────────────────────────────┐
│  Log Warning                           │
│  print(f"Warning: {error}")            │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  Continue Gracefully                   │
│  • Skip this analyzer                  │
│  • Use partial data                    │
│  • Return what we have                 │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  Complete Analysis with Available Data │
│  • Some recommendations may be missing │
│  • User still gets value               │
└────────────────────────────────────────┘
```

## 6. Configuration Flow

```
┌────────────────────────────────────────┐
│  MCP Config JSON                       │
│  {                                     │
│    "env": {                            │
│      "AWS_PROFILE": "my-profile",      │
│      "AWS_REGION": "us-east-1",        │
│      "ATHENA_WORKGROUP": "primary",    │
│      "ATHENA_S3_OUTPUT": "s3://...",   │
│      ...                               │
│    }                                   │
│  }                                     │
└────────────┬───────────────────────────┘
             │
             │ Environment variables
             ▼
┌────────────────────────────────────────┐
│  server.py: main()                     │
│  • Read env vars                       │
│  • Validate required params            │
│  • Call initialize_engine()            │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  OptimizerConfig (Pydantic)            │
│  • Type validation                     │
│  • Default values                      │
│  • Validation errors                   │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  OptimizationEngine                    │
│  • Store config                        │
│  • Pass to collectors                  │
│  • Pass to analyzers                   │
└────────────┬───────────────────────────┘
             │
        ┌────┴────┐
        │         │
        ▼         ▼
┌─────────────┐ ┌──────────┐
│ Collectors  │ │Analyzers │
│ • Use AWS   │ │• Use cost│
│   profile   │ │  per TB  │
│ • Use region│ │• Use     │
│ • Use       │ │  timeouts│
│   workgroup │ │          │
└─────────────┘ └──────────┘
```

## 7. Table Metadata Collection Flow

```
Query: "SELECT * FROM db.table1 JOIN db.table2"
         │
         ▼
┌────────────────────────────────────────┐
│  Extract Table Names                   │
│  • Regex: FROM|JOIN (\w+\.?\w+)        │
│  • Result: ["db.table1", "db.table2"]  │
└────────────┬───────────────────────────┘
             │
             │ For each table
             ▼
┌────────────────────────────────────────┐
│  GlueCollector.get_table_metadata()    │
│  ┌──────────────────────────────────┐  │
│  │ AWS Glue: get_table()            │  │
│  │ Input: database="db",            │  │
│  │        table="table1"            │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐  │
│  │ Parse Response                   │  │
│  │ • StorageDescriptor → format     │  │
│  │ • PartitionKeys → partition_keys │  │
│  │ • Columns → columns              │  │
│  │ • Compressed → compressed        │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐  │
│  │ Create TableMetadata Object      │  │
│  │ • All fields populated           │  │
│  │ • Type-safe (Pydantic)           │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
         │
         │ Store in context
         ▼
context["table_metadata"]["db.table1"] = metadata
         │
         │ Repeat for table2
         ▼
context["table_metadata"]["db.table2"] = metadata
         │
         │ Pass to analyzers
         ▼
   Analyzers use metadata
```

## 8. Recommendation Sorting Flow

```
All Recommendations Collected
         │
         ▼
┌────────────────────────────────────────┐
│  Sort by Severity (Primary)            │
│  Order: CRITICAL → HIGH → MEDIUM       │
│         → LOW → INFO                   │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  Sort by Confidence (Secondary)        │
│  Within same severity, higher          │
│  confidence comes first                │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  Sorted Recommendations                │
│  1. [CRITICAL, 0.9] Missing partitions │
│  2. [CRITICAL, 0.8] CROSS JOIN         │
│  3. [HIGH, 0.9] SELECT *               │
│  4. [HIGH, 0.8] CSV format             │
│  5. [MEDIUM, 0.7] Multiple JOINs       │
│  6. [LOW, 0.6] OUTER JOIN              │
│  7. [INFO, 0.8] Using Parquet          │
└────────────────────────────────────────┘
```

This sorted order ensures users see the most critical, high-confidence recommendations first.
