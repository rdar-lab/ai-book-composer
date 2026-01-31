"""Pytest configuration for test suite."""

import os
import sys

# Ensure the src/ directory is on the path so tests can import the package
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# noinspection PyUnresolvedReferences
from ai_book_composer import logging_config
# noinspection PyUnresolvedReferences
from ai_book_composer.config import Settings

logging_config.setup_logging(Settings())
