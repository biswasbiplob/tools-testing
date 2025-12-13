"""Tests for parallel execution utilities."""

import time
import pytest
from athena_optimizer.parallel import (
    execute_parallel,
    execute_parallel_map,
    get_default_workers
)


class TestParallelExecution:
    """Test parallel execution functions."""

    def test_get_default_workers(self):
        """Test default worker count calculation."""
        workers = get_default_workers()
        assert workers >= 2
        assert workers <= 10

    def test_execute_parallel_basic(self):
        """Test basic parallel execution."""
        tasks = {
            'task1': lambda: 'result1',
            'task2': lambda: 'result2',
            'task3': lambda: 'result3',
        }

        results = execute_parallel(tasks)

        assert len(results) == 3
        assert results['task1'] == 'result1'
        assert results['task2'] == 'result2'
        assert results['task3'] == 'result3'

    def test_execute_parallel_empty_tasks(self):
        """Test parallel execution with empty task list."""
        results = execute_parallel({})
        assert results == {}

    def test_execute_parallel_with_exception(self):
        """Test parallel execution handles exceptions."""
        def failing_task():
            raise ValueError("Task failed")

        tasks = {
            'success': lambda: 'ok',
            'failure': failing_task,
        }

        results = execute_parallel(tasks, fail_fast=False)

        assert len(results) == 2
        assert results['success'] == 'ok'
        assert isinstance(results['failure'], ValueError)

    def test_execute_parallel_fail_fast(self):
        """Test parallel execution with fail_fast enabled."""
        def failing_task():
            raise ValueError("Task failed")

        tasks = {
            'task1': lambda: time.sleep(0.1) or 'result1',
            'task2': failing_task,
        }

        with pytest.raises(ValueError):
            execute_parallel(tasks, fail_fast=True)

    def test_execute_parallel_with_custom_workers(self):
        """Test parallel execution with custom worker count."""
        tasks = {f'task{i}': lambda i=i: i for i in range(5)}

        results = execute_parallel(tasks, max_workers=2)

        assert len(results) == 5
        assert all(i in results.values() for i in range(5))

    def test_execute_parallel_performance(self):
        """Test that parallel execution is actually faster."""
        def slow_task(duration):
            time.sleep(duration)
            return "done"

        # Sequential would take ~0.3s (3 * 0.1s)
        # Parallel should take ~0.1s (all at once)
        tasks = {
            f'task{i}': lambda d=0.1: slow_task(d)
            for i in range(3)
        }

        start = time.time()
        results = execute_parallel(tasks, max_workers=3)
        elapsed = time.time() - start

        assert len(results) == 3
        # Should be much faster than sequential (0.3s)
        assert elapsed < 0.2  # Allow some overhead

    def test_execute_parallel_map_basic(self):
        """Test parallel map execution."""
        items = [1, 2, 3, 4, 5]

        results = execute_parallel_map(items, lambda x: x * 2)

        assert results == [2, 4, 6, 8, 10]

    def test_execute_parallel_map_empty(self):
        """Test parallel map with empty list."""
        results = execute_parallel_map([], lambda x: x)
        assert results == []

    def test_execute_parallel_map_with_exception(self):
        """Test parallel map handles exceptions."""
        def process_item(x):
            if x == 2:
                raise ValueError("Bad value")
            return x * 2

        items = [1, 2, 3]

        results = execute_parallel_map(items, process_item, fail_fast=False)

        assert len(results) == 3
        assert results[0] == 2
        assert isinstance(results[1], ValueError)
        assert results[2] == 6

    def test_execute_parallel_map_fail_fast(self):
        """Test parallel map with fail_fast enabled."""
        def process_item(x):
            if x == 2:
                raise ValueError("Bad value")
            time.sleep(0.1)
            return x * 2

        items = [1, 2, 3]

        with pytest.raises(ValueError):
            execute_parallel_map(items, process_item, fail_fast=True)

    def test_execute_parallel_map_preserves_order(self):
        """Test that parallel map preserves input order."""
        items = list(range(10))

        # Process items with random sleep to ensure order is preserved
        def process_with_delay(x):
            time.sleep(0.01 * (10 - x))  # Later items finish first
            return x * 2

        results = execute_parallel_map(items, process_with_delay, max_workers=5)

        # Results should match input order
        assert results == [x * 2 for x in items]

    def test_execute_parallel_complex_results(self):
        """Test parallel execution with complex result types."""
        tasks = {
            'dict': lambda: {'key': 'value'},
            'list': lambda: [1, 2, 3],
            'tuple': lambda: (1, 2, 3),
            'none': lambda: None,
        }

        results = execute_parallel(tasks)

        assert results['dict'] == {'key': 'value'}
        assert results['list'] == [1, 2, 3]
        assert results['tuple'] == (1, 2, 3)
        assert results['none'] is None
