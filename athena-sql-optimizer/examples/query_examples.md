# Query Examples and Expected Recommendations

This document shows example queries and the recommendations the analyzer should provide.

## Example 1: SELECT * on Large Table

### Query
```sql
SELECT *
FROM sales_data
WHERE region = 'US'
```

### Expected Recommendations
- **[HIGH] SELECT * Detected**: Replace with specific columns (50%+ cost savings)
- **[MEDIUM] Table Not Partitioned**: Consider partitioning by date/region
- **[HIGH] Non-Columnar Format**: Convert to Parquet/ORC (70%+ savings)

## Example 2: Missing Partition Filters

### Query
```sql
SELECT customer_id, SUM(amount)
FROM transactions
WHERE amount > 100
GROUP BY customer_id
```

### Expected Recommendations (if table is partitioned by `transaction_date`)
- **[CRITICAL] Missing Partition Filters**: Not filtering on `transaction_date` causes full table scan
- **[HIGH] SELECT * Detected**: If using SELECT *
- **[INFO] Partitioned Table**: Reminder to use partition filters

## Example 3: Complex JOINs

### Query
```sql
SELECT *
FROM large_table l
CROSS JOIN medium_table m
LEFT JOIN small_table s ON l.id = s.id
WHERE l.date > '2024-01-01'
```

### Expected Recommendations
- **[CRITICAL] Cross Join Detected**: Will create cartesian product
- **[HIGH] SELECT * Detected**: Select only needed columns
- **[MEDIUM] Multiple JOINs Detected**: Consider join order optimization
- **[LOW] OUTER JOINs May Be Expensive**: Verify LEFT JOIN is necessary
- **[INFO] JOIN Order Optimization**: Place smaller tables on right side

## Example 4: Inefficient Aggregation

### Query
```sql
SELECT DISTINCT *
FROM large_table
WHERE date LIKE '2024-%'
```

### Expected Recommendations
- **[HIGH] SELECT * Detected**: Scanning all columns
- **[MEDIUM] DISTINCT on Many Columns**: Expensive deduplication
- **[HIGH] Inefficient Partition Filters**: LIKE pattern on partition column

## Example 5: Non-Equality JOIN

### Query
```sql
SELECT a.*, b.*
FROM table_a a
JOIN table_b b ON a.id != b.id
```

### Expected Recommendations
- **[CRITICAL] Non-Equality JOIN Conditions**: Can cause cartesian products
- **[HIGH] SELECT * Detected**: Multiple tables with SELECT *
- **[MEDIUM] Expensive Operation**: Non-equality joins are very expensive

## Example 6: Optimized Query

### Query
```sql
SELECT
    customer_id,
    transaction_date,
    SUM(amount) as total_amount
FROM transactions
WHERE
    transaction_date >= DATE '2024-01-01'
    AND transaction_date < DATE '2024-02-01'
    AND region = 'US'
GROUP BY
    customer_id,
    transaction_date
```

### Expected Recommendations (for well-optimized query)
- **[INFO] Optimal Format**: If using Parquet/ORC
- **[INFO] Partitioned Table**: Partition filters properly used
- **[LOW] Query Cost**: Minimal recommendations if well-optimized

## Example 7: Sorting Without Limit

### Query
```sql
SELECT customer_id, transaction_date, amount
FROM transactions
ORDER BY amount DESC
```

### Expected Recommendations
- **[MEDIUM] Expensive Operation: Sort without LIMIT**: Sorting entire result set
- Suggest adding LIMIT if not all rows needed

## Example 8: Window Functions

### Query
```sql
SELECT
    customer_id,
    amount,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY transaction_date) as rn
FROM transactions
```

### Expected Recommendations
- **[MEDIUM] Expensive Operation: Window function**: Can be expensive with large partitions
- Ensure PARTITION BY uses appropriate columns

## Example 9: COUNT(*)

### Query
```sql
SELECT COUNT(*) FROM large_table
```

### Expected Recommendations
- **[MEDIUM] COUNT(*) Usage**: Replace with COUNT(1) or COUNT(partition_column)

## Example 10: Format Conversion Needed

### Query
```sql
SELECT col1, col2, col3
FROM csv_table
WHERE date_col = '2024-01-01'
```

### Expected Recommendations (for CSV table)
- **[HIGH] Non-Columnar Format**: Convert CSV to Parquet
- **[MEDIUM] Uncompressed Data**: Enable compression
- Code example provided for CTAS conversion

## Cost Analysis Examples

### High Cost Query
```sql
SELECT * FROM multi_tb_table  -- Scans 10 TB
```

**Expected Cost Analysis:**
- Current cost: $50.00 (10 TB × $5/TB)
- Optimized cost: $5.00 (1 TB with partitions + column selection)
- Savings: $45.00 (90%)

### Medium Cost Query
```sql
SELECT specific_cols FROM partitioned_table
WHERE partition_date = '2024-01-01'  -- Scans 0.5 TB
```

**Expected Cost Analysis:**
- Current cost: $2.50
- Optimized cost: $1.25 (better format)
- Savings: $1.25 (50%)

### Low Cost Query
```sql
SELECT id, name FROM small_parquet_table
WHERE partition_key = '2024-01-01'
LIMIT 100
```

**Expected Cost Analysis:**
- Current cost: $0.05
- Already well-optimized
- Minimal recommendations

## Table Health Check Examples

### CSV Table
```sql
-- check_table_health("my_db", "csv_table")
```

**Expected Recommendations:**
- Format: CSV
- Recommendation: Convert to Parquet (70% savings)
- Recommendation: Add partitioning
- Recommendation: Enable compression

### Well-Optimized Parquet Table
```sql
-- check_table_health("my_db", "optimized_table")
```

**Expected Output:**
- Format: PARQUET
- Partitions: year, month (24 partitions)
- Compressed: Yes
- Recommendations: Minimal, mostly INFO level

### Unpartitioned Large Table
```sql
-- check_table_health("my_db", "large_unpartitioned")
```

**Expected Recommendations:**
- Table size: 5 TB
- Recommendation: Partition by date column
- Recommendation: Consider partition projection
- Example CTAS query provided
