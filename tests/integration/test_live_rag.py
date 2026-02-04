"""Unit tests for RAG module."""
import random
from pathlib import Path

import numpy as np

from src.ai_book_composer.config import Settings
from src.ai_book_composer.rag import RAGManager
from src.ai_book_composer.utils import file_utils


class TestLiveRAGManager:
    def test_real_ingest(self):
        random.seed(42)
        np.random.seed(42)

        settings = Settings()

        fixtures_dir = Path(__file__).parent.parent / "fixtures"

        fixture_files = [
            fixtures_dir / "article1_ai_intro.txt",
            fixtures_dir / "article2_ml_fundamentals.txt",
            fixtures_dir / "article3_deep_learning.txt",
            fixtures_dir / "7.pdf",
            fixtures_dir / "2307.06435v10.pdf"
        ]

        gathered_content = {}
        for file_path in fixture_files:
            if file_path.exists():
                content = file_utils.read_text_file(settings, str(file_path.resolve()))["content"]
                gathered_content[str(file_path)] = {
                    "name": file_path.name,
                    "type": "text",
                    "content": content
                }

        manager = RAGManager(settings)
        ingest_result = manager.ingest_documents(gathered_content)
        assert ingest_result == {'status': 'success', 'total_chunks': 479, 'total_documents': 5}
        relevant_documents = manager.retrieve_relevant_documents("Key concepts of machine learning", k=3)
        assert len(relevant_documents) > 0
        for doc in relevant_documents:
            assert "content" in doc
            assert "metadata" in doc
            assert "similarity_score" in doc
            assert doc["similarity_score"] >= 0.0
            assert doc["similarity_score"] <= 0.9
