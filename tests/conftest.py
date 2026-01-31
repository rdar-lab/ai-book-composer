"""Pytest configuration for test suite."""
import shutil
from pathlib import Path

import pytest

from src.ai_book_composer import logging_config
from src.ai_book_composer.config import Settings

settings = Settings()
settings.logging.level = "INFO"
settings.logging.file = None
settings.logging.console_output = True

logging_config.setup_logging(settings)


@pytest.fixture(scope="session", autouse=True)
def after_all_tests():
    yield

    # Remove the cache directory after all tests
    cache_dir = Path(settings.general.cache_dir)
    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir)
