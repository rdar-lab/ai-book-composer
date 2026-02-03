"""Unit tests for term extraction utilities."""

from src.ai_book_composer.utils.term_extraction import extract_key_terms


class TestTermExtraction:
    """Test key term extraction functionality."""

    def test_extract_key_terms_success(self):
        """Test extracting key terms successfully."""
        documents = [
            "Python programming language is great for machine learning and artificial intelligence.",
            "Machine learning algorithms require data science knowledge.",
        ]
        terms = extract_key_terms(documents, max_terms=10)

        assert len(terms) > 0
        assert "machine" in terms or "learning" in terms or "python" in terms

    def test_extract_key_terms_filters_stop_words(self):
        """Test that stop words are filtered out."""
        terms = extract_key_terms(["The quick brown fox jumps over the lazy dog."], max_terms=10)

        # Stop words should be filtered out
        for term in terms:
            assert term not in ["the", "over", "a", "an", "and"]

    def test_extract_key_terms_respects_max_terms(self):
        """Test that max_terms is respected."""
        # Create content with many different terms
        content_text = " ".join([f"term{i}" for i in range(100)])

        terms = extract_key_terms([content_text], max_terms=10)

        assert len(terms) <= 10

    def test_extract_key_terms_no_documents(self):
        """Test with empty content."""
        gathered_content = []

        terms = extract_key_terms(gathered_content, max_terms=10)

        assert len(terms) == 0

    def test_extract_key_terms_min_length(self):
        """Test that min_term_length is respected."""
        # With min_term_length=3, only 'abc', 'abcd', 'abcde' should be included
        terms = extract_key_terms(["a a a a a ab ab ab ab abc abcd abcde"], max_terms=10, min_term_length=3)

        for term in terms:
            assert len(term) >= 3
