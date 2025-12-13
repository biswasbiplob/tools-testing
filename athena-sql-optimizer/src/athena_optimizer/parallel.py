"""Parallel execution utilities for AWS operations."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any, Optional, Dict, List
import os

from .logging import get_logger

logger = get_logger(__name__)


def get_default_workers() -> int:
    """
    Get default number of worker threads for parallel execution.

    Returns:
        Number of workers based on CPU count (min 2, max 10)
    """
    cpu_count = os.cpu_count() or 4
    # Use 2x CPU count for I/O-bound AWS API calls, but cap at 10
    return min(cpu_count * 2, 10)


def execute_parallel(
    tasks: Dict[str, Callable[[], Any]],
    max_workers: Optional[int] = None,
    fail_fast: bool = False
) -> Dict[str, Any]:
    """
    Execute multiple tasks in parallel using a thread pool.

    Args:
        tasks: Dictionary mapping task names to callables
        max_workers: Maximum number of worker threads (defaults to 2x CPU count)
        fail_fast: If True, cancel remaining tasks on first error

    Returns:
        Dictionary mapping task names to results (or exceptions if they failed)

    Example:
        results = execute_parallel({
            'task1': lambda: fetch_data_1(),
            'task2': lambda: fetch_data_2(),
        })
    """
    if not tasks:
        return {}

    # Use default workers if not specified
    workers = max_workers or get_default_workers()

    # Cap workers to number of tasks (no point having more threads than tasks)
    workers = min(workers, len(tasks))

    logger.debug(
        "parallel_execution_started",
        num_tasks=len(tasks),
        max_workers=workers,
        task_names=list(tasks.keys())
    )

    results = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks and keep track of future to task name mapping
        future_to_name = {
            executor.submit(func): name
            for name, func in tasks.items()
        }

        # Process completed tasks
        for future in as_completed(future_to_name):
            task_name = future_to_name[future]

            try:
                result = future.result()
                results[task_name] = result
                logger.debug(
                    "parallel_task_completed",
                    task_name=task_name,
                    success=True
                )
            except Exception as e:
                logger.warning(
                    "parallel_task_failed",
                    task_name=task_name,
                    error=str(e)
                )

                if fail_fast:
                    # Cancel remaining futures
                    for f in future_to_name:
                        f.cancel()
                    raise

                # Store exception in results
                results[task_name] = e

    logger.debug(
        "parallel_execution_completed",
        num_tasks=len(tasks),
        num_succeeded=sum(1 for r in results.values() if not isinstance(r, Exception)),
        num_failed=sum(1 for r in results.values() if isinstance(r, Exception))
    )

    return results


def execute_parallel_map(
    items: List[Any],
    func: Callable[[Any], Any],
    max_workers: Optional[int] = None,
    fail_fast: bool = False
) -> List[Any]:
    """
    Apply a function to items in parallel using a thread pool.

    This is a simpler, more Pythonic implementation using ThreadPoolExecutor
    directly instead of routing through execute_parallel.

    Args:
        items: List of items to process
        func: Function to apply to each item
        max_workers: Maximum number of worker threads
        fail_fast: If True, stop on first error

    Returns:
        List of results in the same order as input items.
        If fail_fast=False, exceptions are returned in place of results.

    Example:
        tables = ['table1', 'table2', 'table3']
        results = execute_parallel_map(
            tables,
            lambda t: fetch_metadata(t)
        )
    """
    if not items:
        return []

    workers = max_workers or get_default_workers()
    workers = min(workers, len(items))

    logger.debug(
        "parallel_map_started",
        num_items=len(items),
        max_workers=workers
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        if fail_fast:
            # Use executor.map for immediate exception propagation
            try:
                results = list(executor.map(func, items))
                logger.debug("parallel_map_completed", num_items=len(results))
                return results
            except Exception as e:
                logger.warning("parallel_map_failed_fast", error=str(e))
                raise
        else:
            # Submit all items and collect results/exceptions gracefully
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
            results = [None] * len(items)

            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = e
                    logger.debug("parallel_map_item_failed", index=index, error=str(e))

            num_succeeded = sum(1 for r in results if not isinstance(r, Exception))
            logger.debug(
                "parallel_map_completed",
                total=len(results),
                succeeded=num_succeeded,
                failed=len(results) - num_succeeded
            )

            return results
