"""Utilities for parallel execution of tasks."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Callable, List, Any, Dict
import logging

from .config import Settings

logger = logging.getLogger(__name__)


def is_parallel_enabled(settings: Settings) -> bool:
    """Check if parallel execution is enabled.
    
    Returns:
        True if parallel execution is enabled, False otherwise
    """
    return settings.parallel.parallel_execution


def get_worker_count(settings: Settings) -> int:
    """Get the number of parallel workers.
    
    Returns:
        Number of workers (1 for sequential, configured value for parallel)
    """
    if is_parallel_enabled(settings):
        return settings.parallel.parallel_workers
    return 1


def execute_parallel(
        settings: Settings,
        func: Callable,
        items: List[Any],
        *args,
        **kwargs
) -> List[Any]:
    """Execute a function on a list of items in parallel (if enabled).
    
    Args:
        settings: The application settings
        func: Function to execute. Should accept an item as first argument.
        items: List of items to process
        *args: Additional positional arguments to pass to func
        **kwargs: Additional keyword arguments to pass to func
        
    Returns:
        List of results in the same order as input items. If an item fails to process,
        the corresponding result will be a dictionary with 'error' and 'item' keys
        instead of the normal return value.
    """
    if not items:
        return []

    worker_count = get_worker_count(settings)

    # If parallel execution is disabled or only 1 worker, execute sequentially
    if worker_count == 1:
        logger.debug(f"Executing {len(items)} items sequentially")
        results = []
        for item in items:
            try:
                result = func(item, *args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.exception(f"Error processing item: {e}")
                results.append({"error": str(e), "item": str(item)[:100]})  # Limit item string to 100 chars
        return results

    # Execute in parallel with ThreadPoolExecutor
    logger.info(f"Executing {len(items)} items in parallel with {worker_count} workers")
    results: list[Any] = [None] * len(items)  # Pre-allocate list to maintain order

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        # Submit all tasks and track their indices
        future_to_index = {}
        for i, item in enumerate(items):
            future = executor.submit(func, item, *args, **kwargs)
            future_to_index[future] = i

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                logger.exception(f"Error processing item at index {index}: {e}")
                results[index] = {"error": str(e), "item": str(items[index])[:100]}  # Limit item string to 100 chars

    return results


def execute_parallel_with_context(
        settings: Settings,
        func: Callable,
        items: List[Any],
        context: Dict[str, Any],
        *args,
        **kwargs
) -> List[Any]:
    """Execute a function on items in parallel with a shared context.
    
    This is useful when you need to pass the same context to all function calls.
    
    Args:
        settings: The application settings
        func: Function to execute. Should accept (item, context, *args, **kwargs)
        items: List of items to process
        context: Shared context dictionary to pass to each function call
        *args: Additional positional arguments to pass to func
        **kwargs: Additional keyword arguments to pass to func
        
    Returns:
        List of results in the same order as input items
    """
    # Create a partial function that includes the context
    func_with_context = partial(func, context=context)
    return execute_parallel(settings, func_with_context, items, *args, **kwargs)
