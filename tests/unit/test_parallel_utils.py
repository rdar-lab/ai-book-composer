"""Unit tests for parallel execution utilities."""
import time

import pytest

from src.ai_book_composer.config import Settings
from src.ai_book_composer.parallel_utils import (
    is_parallel_enabled,
    get_worker_count,
    execute_parallel
)


class TestParallelUtilsBasics:
    """Test basic utility functions."""

    def test_is_parallel_enabled_true(self):
        """Test parallel execution enabled check."""
        settings = Settings()
        settings.parallel.parallel_execution = True
        assert is_parallel_enabled(settings) is True

    def test_is_parallel_enabled_false(self):
        """Test parallel execution disabled check."""
        settings = Settings()
        settings.parallel.parallel_execution = False
        assert is_parallel_enabled(settings) is False

    def test_get_worker_count_parallel_enabled(self):
        """Test worker count when parallel is enabled."""
        settings = Settings()
        settings.parallel.parallel_execution = True
        settings.parallel.parallel_workers = 4
        assert get_worker_count(settings) == 4

    def test_get_worker_count_parallel_disabled(self):
        """Test worker count when parallel is disabled."""
        settings = Settings()
        settings.parallel.parallel_execution = False
        assert get_worker_count(settings) == 1


class TestExecuteParallelSequential:
    """Test sequential execution (parallel disabled)."""

    def test_execute_parallel_empty_list(self):
        """Test execution with empty list."""
        settings = Settings()
        settings.parallel.parallel_execution = False

        def dummy_func(item):
            return item * 2

        results = execute_parallel(settings, dummy_func, [])
        assert results == []

    def test_execute_parallel_sequential_success(self):
        """Test sequential execution with successful items."""
        settings = Settings()
        settings.parallel.parallel_execution = False

        def multiply_by_two(item):
            return item * 2

        items = [1, 2, 3, 4, 5]
        results = execute_parallel(settings, multiply_by_two, items)

        assert results == [2, 4, 6, 8, 10]

    def test_execute_parallel_sequential_with_error(self):
        """Test sequential execution with errors."""
        settings = Settings()
        settings.parallel.parallel_execution = False

        def divide_by_value(item):
            if item == 0:
                raise ValueError("Cannot divide by zero")
            return 100 / item

        items = [5, 0, 10]
        results = execute_parallel(settings, divide_by_value, items)

        assert results[0] == 20.0
        assert "error" in results[1]
        assert results[2] == 10.0

    def test_execute_parallel_sequential_with_extra_args(self):
        """Test sequential execution with extra arguments."""
        settings = Settings()
        settings.parallel.parallel_execution = False

        def add_values(item, offset, multiplier=1):
            return (item + offset) * multiplier

        items = [1, 2, 3]
        results = execute_parallel(settings, add_values, items, False, 10, multiplier=2)

        assert results == [22, 24, 26]


class TestExecuteParallelThreaded:
    """Test parallel execution with multiple workers."""

    def test_execute_parallel_threaded_success(self):
        """Test parallel execution with successful items."""
        settings = Settings()
        settings.parallel.parallel_execution = True
        settings.parallel.parallel_workers = 2

        def process_item(item):
            return item ** 2

        items = [1, 2, 3, 4, 5]
        results = execute_parallel(settings, process_item, items)

        assert results == [1, 4, 9, 16, 25]

    def test_execute_parallel_threaded_with_error(self):
        """Test parallel execution with errors."""
        settings = Settings()
        settings.parallel.parallel_execution = True
        settings.parallel.parallel_workers = 3

        def risky_operation(item):
            if item == "error":
                raise RuntimeError("Expected error")
            return f"processed_{item}"

        items = ["a", "error", "b", "c"]
        results = execute_parallel(settings, risky_operation, items)

        assert results[0] == "processed_a"
        assert "error" in results[1]
        assert results[2] == "processed_b"
        assert results[3] == "processed_c"

    def test_execute_parallel_threaded_with_error_fail(self):
        """Test parallel execution with errors."""
        settings = Settings()
        settings.parallel.parallel_execution = True
        settings.parallel.parallel_workers = 3

        def risky_operation(item):
            if item == "error":
                raise RuntimeError("Expected error")
            return f"processed_{item}"

        items = ["a", "error", "b", "c"]
        with pytest.raises(RuntimeError, match="Expected error"):
            results = execute_parallel(settings, risky_operation, items, fail_on_error=True)

    def test_execute_parallel_maintains_order(self):
        """Test that parallel execution maintains item order."""
        settings = Settings()
        settings.parallel.parallel_execution = True
        settings.parallel.parallel_workers = 4

        def slow_process(item):
            # Simulate varying processing times
            time.sleep(0.01 * (5 - item))  # Earlier items take longer
            return item * 10

        items = [1, 2, 3, 4, 5]
        results = execute_parallel(settings, slow_process, items)

        # Results should be in order despite different processing times
        assert results == [10, 20, 30, 40, 50]

    def test_execute_parallel_single_worker(self):
        """Test parallel execution with single worker (acts as sequential)."""
        settings = Settings()
        settings.parallel.parallel_execution = True
        settings.parallel.parallel_workers = 1

        def process(item):
            return item + 1

        items = [10, 20, 30]
        results = execute_parallel(settings, process, items)

        assert results == [11, 21, 31]


class TestExecuteParallelEdgeCases:
    """Test edge cases and error handling."""

    def test_execute_parallel_long_error_message(self):
        """Test that error messages are truncated to 100 chars."""
        settings = Settings()
        settings.parallel.parallel_execution = False

        def failing_func(item):
            raise ValueError(f"Very long error message: {item}")

        long_item = "x" * 200
        results = execute_parallel(settings, failing_func, [long_item])

        assert "error" in results[0]
        assert len(results[0]["item"]) <= 100

    def test_execute_parallel_with_none_items(self):
        """Test execution with None values."""
        settings = Settings()
        settings.parallel.parallel_execution = False

        def process(item):
            if item is None:
                return "none"
            return item

        items = [1, None, 3]
        results = execute_parallel(settings, process, items)

        assert results == [1, "none", 3]

    def test_execute_parallel_preserves_result_types(self):
        """Test that different result types are preserved."""
        settings = Settings()
        settings.parallel.parallel_execution = True
        settings.parallel.parallel_workers = 2

        def varied_results(item):
            if item == 1:
                return {"result": "dict"}
            elif item == 2:
                return ["list"]
            elif item == 3:
                return 42
            return None

        items = [1, 2, 3, 4]
        results = execute_parallel(settings, varied_results, items)

        assert results[0] == {"result": "dict"}
        assert results[1] == ["list"]
        assert results[2] == 42
        assert results[3] is None
