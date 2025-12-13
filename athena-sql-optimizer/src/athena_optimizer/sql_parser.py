"""SQL parsing utilities using sqlparse for robust query analysis."""

from typing import List, Set
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Function, Parenthesis
from sqlparse.tokens import Keyword, DML


def extract_table_names(query: str) -> List[str]:
    """
    Extract table names from SQL query using proper SQL parsing.

    This function uses sqlparse to robustly extract table names from:
    - FROM clauses
    - JOIN clauses (INNER, LEFT, RIGHT, FULL, CROSS)
    - Subqueries and CTEs

    Args:
        query: SQL query string

    Returns:
        List of unique table names (may include database.table notation)
    """
    tables: Set[str] = set()

    # Parse the SQL query
    try:
        parsed = sqlparse.parse(query)
    except Exception:
        # If parsing fails, return empty list
        return []

    for statement in parsed:
        # Extract tables from this statement
        tables.update(_extract_from_statement(statement))

    return sorted(list(tables))


def _extract_from_statement(statement) -> Set[str]:
    """Extract table names from a single SQL statement."""
    tables: Set[str] = set()
    from_seen = False

    for token in statement.tokens:
        # Skip whitespace and comments
        if token.is_whitespace:
            continue

        # Check for FROM keyword
        if token.ttype is Keyword and token.value.upper() in ("FROM", "JOIN", "INNER JOIN",
                                                                "LEFT JOIN", "RIGHT JOIN",
                                                                "FULL JOIN", "CROSS JOIN"):
            from_seen = True
            continue

        # After FROM or JOIN, extract table names
        if from_seen:
            if isinstance(token, IdentifierList):
                # Multiple tables/identifiers
                for identifier in token.get_identifiers():
                    table_name = _extract_table_from_identifier(identifier)
                    if table_name:
                        tables.add(table_name)
                from_seen = False

            elif isinstance(token, Identifier):
                # Single table
                table_name = _extract_table_from_identifier(token)
                if table_name:
                    tables.add(table_name)
                from_seen = False

            elif isinstance(token, Function):
                # Skip functions (e.g., table-valued functions)
                from_seen = False

            elif isinstance(token, Parenthesis):
                # Subquery - recursively extract
                subquery = token.value[1:-1]  # Remove parentheses
                sub_tables = extract_table_names(subquery)
                tables.update(sub_tables)
                from_seen = False

            elif token.ttype is Keyword:
                # Another keyword, stop looking for tables
                from_seen = False

    return tables


def _extract_table_from_identifier(identifier: Identifier) -> str:
    """
    Extract table name from an identifier token.

    Handles:
    - Simple table names: table
    - Database qualified: database.table
    - Tables with aliases: table AS t or table t

    Args:
        identifier: sqlparse Identifier token

    Returns:
        Table name (possibly database.table) or empty string
    """
    # Check if identifier contains a dot (database.table notation)
    identifier_str = str(identifier).split()[0]  # Remove alias if present

    # Check for database.table pattern
    if '.' in identifier_str:
        # Clean up any quotes or backticks
        parts = identifier_str.split('.')
        cleaned_parts = [p.strip('`"[]') for p in parts]
        if len(cleaned_parts) >= 2:
            return f"{cleaned_parts[0]}.{cleaned_parts[1]}"

    # Otherwise, get the real name (handles simple tables and aliases)
    name = identifier.get_real_name()
    if name:
        return name.strip('`"[]')

    return ""


def _extract_tables_from_where(statement) -> Set[str]:
    """
    Extract tables from WHERE clauses (for subqueries).

    Args:
        statement: Parsed SQL statement

    Returns:
        Set of table names found in WHERE clause subqueries
    """
    tables: Set[str] = set()

    # Look for subqueries in WHERE clause
    for token in statement.tokens:
        if isinstance(token, Parenthesis):
            # Check if this contains a SELECT (subquery)
            subquery_str = token.value[1:-1]
            if 'SELECT' in subquery_str.upper():
                sub_tables = extract_table_names(subquery_str)
                tables.update(sub_tables)

    return tables
