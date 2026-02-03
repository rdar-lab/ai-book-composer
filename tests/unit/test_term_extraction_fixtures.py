"""Unit test for term extraction using fixture files.

This test validates term extraction against real content from fixture files,
comparing results with expected terms to identify potential stop words that
should be added to the exclusion list.
"""

import json
from pathlib import Path

from src.ai_book_composer.config import Settings
from src.ai_book_composer.utils import file_utils
from src.ai_book_composer.utils.term_extraction import extract_key_terms


class TestTermExtractionWithFixtures:
    """Test term extraction using actual fixture files."""

    def test_extract_terms_from_fixture_files(self):
        """Test term extraction from fixture text files and compare with expected list."""
        # Load fixture files
        fixtures_dir = Path(__file__).parent.parent / "fixtures"

        fixture_files = [
            fixtures_dir / "article1_ai_intro.txt",
            fixtures_dir / "article2_ml_fundamentals.txt",
            fixtures_dir / "article3_deep_learning.txt",
            fixtures_dir / "7.pdf",
            fixtures_dir / "2307.06435v10.pdf"
        ]

        settings = Settings()

        # Create documents structure
        documents = []
        for file_path in fixture_files:
            if file_path.exists():
                content = file_utils.read_text_file(settings, str(file_path.resolve()))["content"]
                documents.append(content)

        # Extract terms
        extracted_terms = extract_key_terms(documents, max_terms=10000, min_term_length=3)

        # Load expected terms
        expected_terms_path = fixtures_dir / "expected_terms.json"
        with open(expected_terms_path, 'r') as f:
            expected_terms = json.load(f)

        assert expected_terms == extracted_terms
