"""Term extraction utilities for AI Book Composer.

This module provides utilities for extracting key terms and subjects from document collections.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Set, Optional

logger = logging.getLogger(__name__)

# Cache for stop words to avoid repeated file reads
_IGNORE_WORDS_CACHE: Optional[Set[str]] = None
_WORDS_EXTRACTION_REGEX = re.compile(r'\b[a-z][\w\-]*\b', re.IGNORECASE)


def _get_ignore_words() -> Set[str]:
    """Load stop words from JSON file.
    
    Returns:
        Set of stop words loaded from ignore_terms.json
    """
    global _IGNORE_WORDS_CACHE

    if _IGNORE_WORDS_CACHE is None:
        # Try to find ignore_terms.json in project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        ignore_words_path = project_root / "ignore_terms.json"

        if not ignore_words_path.exists():
            raise Exception(f"Stop words file not found at {ignore_words_path}, using default set")

        try:
            with open(ignore_words_path, 'r') as f:
                words = json.load(f)
                words = [word.strip().lower() for word in words if word.strip()]
                _IGNORE_WORDS_CACHE = set(words)
                logger.info(f"Loaded {len(_IGNORE_WORDS_CACHE)} stop words from {ignore_words_path}")
        except Exception as e:
            raise Exception(f"Error loading stop words from JSON: {e}, using default set", e)

    return _IGNORE_WORDS_CACHE


def extract_key_terms(
        documents,
        max_terms: int = 50,
        min_term_length: int = 3
) -> List[str]:
    """Extract key terms/subjects from document collection efficiently.
    
    This function extracts important terms from document summaries and content
    using frequency analysis and filtering heuristics (no LLM needed).
    
    Args:
        documents: The documents to analyze
        max_terms: Maximum number of terms to return
        min_term_length: Minimum length of terms to consider
        
    Returns:
        List of key terms sorted by importance
    """
    logger.info("Extracting key terms from document collection...")

    # Load stop words from JSON file
    ignore_words = _get_ignore_words()

    # Collect all terms from summaries and content
    term_counter = Counter()

    for content in documents:
        # Extract terms using simple heuristics
        terms = _extract_terms_from_text(content, ignore_words, min_term_length)
        term_counter.update(terms)

    # Get most common terms
    key_terms = [term for term, _ in term_counter.most_common(max_terms)][:max_terms]

    logger.info(f"Extracted {len(key_terms)} key terms from {len(documents)} documents")
    return key_terms


def _extract_terms_from_text(
        text: str,
        ignore_words: Set[str],
        min_length: int
) -> List[str]:
    """Extract meaningful terms from text.
    
    Args:
        text: Text to extract terms from
        ignore_words: Set of stop words to filter
        min_length: Minimum term length
        
    Returns:
        List of extracted terms
    """
    # Convert to lowercase and split into words
    # Keep alphanumeric and hyphens
    words = _WORDS_EXTRACTION_REGEX.findall(text.lower())

    # Filter terms
    terms = []
    for word in words:
        word = word.strip()
        # Skip stop words, short words, and numbers
        if (word and word not in ignore_words and
                len(word) >= min_length and
                not word.isdigit()):
            terms.append(word)

    return terms
