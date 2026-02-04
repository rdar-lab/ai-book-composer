"""Unit tests for RAG module."""
from unittest.mock import MagicMock, patch

from src.ai_book_composer.config import Settings
from src.ai_book_composer.rag import RAGManager


class TestRAGManagerInitialization:
    """Test RAGManager initialization."""

    # noinspection PyUnusedLocal
    @patch('src.ai_book_composer.rag.HuggingFaceEmbeddings')
    @patch('src.ai_book_composer.rag.chromadb.Client')
    @patch('src.ai_book_composer.rag.Chroma')
    def test_init_creates_manager(self, mock_chroma, mock_client, mock_embeddings):
        """Test that RAGManager can be initialized."""
        settings = Settings()

        # Mock the embeddings
        mock_embeddings.return_value = MagicMock()

        manager = RAGManager(settings)

        assert manager.settings == settings
        assert manager.embeddings is not None
        assert manager.text_splitter is not None
        assert manager.vectorstore is not None


class TestRAGManagerIngestDocuments:
    """Test document ingestion functionality."""

    # noinspection PyUnusedLocal
    @patch('src.ai_book_composer.rag.HuggingFaceEmbeddings')
    @patch('src.ai_book_composer.rag.chromadb.Client')
    @patch('src.ai_book_composer.rag.Chroma')
    def test_ingest_documents_success(self, mock_chroma, mock_client, mock_embeddings):
        """Test ingesting documents successfully."""
        settings = Settings()

        # Mock components
        mock_embeddings.return_value = MagicMock()
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore

        manager = RAGManager(settings)

        # Prepare test data
        gathered_content = {
            "/path/file1.txt": {
                "name": "file1.txt",
                "type": "text",
                "content": "This is a test document with some content that should be long enough to not be skipped.",
                "summary": "Test document"
            },
            "/path/file2.txt": {
                "name": "file2.txt",
                "type": "text",
                "content": "Another test document with different content that is also sufficiently long for testing.",
                "summary": "Another test"
            }
        }

        result = manager.ingest_documents(gathered_content)

        assert result["status"] == "success"
        assert result["total_documents"] == 2
        assert result["total_chunks"] > 0
        assert mock_vectorstore.add_texts.called

    # noinspection PyUnusedLocal
    @patch('src.ai_book_composer.rag.HuggingFaceEmbeddings')
    @patch('src.ai_book_composer.rag.chromadb.Client')
    @patch('src.ai_book_composer.rag.Chroma')
    def test_ingest_documents_empty(self, mock_chroma, mock_client, mock_embeddings):
        """Test ingesting empty document collection."""
        settings = Settings()

        mock_embeddings.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()

        manager = RAGManager(settings)

        result = manager.ingest_documents({})

        assert result["status"] == "no_content"
        assert result["total_documents"] == 0

    # noinspection PyUnusedLocal
    @patch('src.ai_book_composer.rag.HuggingFaceEmbeddings')
    @patch('src.ai_book_composer.rag.chromadb.Client')
    @patch('src.ai_book_composer.rag.Chroma')
    def test_ingest_documents_includes_short_content(self, mock_chroma, mock_client, mock_embeddings):
        """Test that short content is included (not skipped)."""
        settings = Settings()

        mock_embeddings.return_value = MagicMock()
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore

        manager = RAGManager(settings)

        # Short content should now be included
        gathered_content = {
            "/path/short.txt": {
                "name": "short.txt",
                "type": "text",
                "content": "Short file content",
                "summary": "Short"
            }
        }

        result = manager.ingest_documents(gathered_content)

        # Should be successful since we no longer skip short files
        assert result["status"] == "success"
        assert result["total_documents"] == 1
        assert mock_vectorstore.add_texts.called


class TestRAGManagerRetrieveDocuments:
    """Test document retrieval functionality."""

    # noinspection PyUnusedLocal
    @patch('src.ai_book_composer.rag.HuggingFaceEmbeddings')
    @patch('src.ai_book_composer.rag.chromadb.Client')
    @patch('src.ai_book_composer.rag.Chroma')
    def test_retrieve_documents_success(self, mock_chroma, mock_client, mock_embeddings):
        """Test retrieving documents successfully."""
        settings = Settings()

        mock_embeddings.return_value = MagicMock()
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore

        # Mock search results
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"file_name": "test.txt", "file_type": "text"}
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_doc, 0.85)
        ]

        manager = RAGManager(settings)

        results = manager.retrieve_relevant_documents("test query", k=5)

        assert len(results) == 1
        assert results[0]["content"] == "Test content"
        assert results[0]["metadata"]["file_name"] == "test.txt"
        assert results[0]["similarity_score"] == 0.85

    # noinspection PyUnusedLocal
    @patch('src.ai_book_composer.rag.HuggingFaceEmbeddings')
    @patch('src.ai_book_composer.rag.chromadb.Client')
    @patch('src.ai_book_composer.rag.Chroma')
    def test_retrieve_documents_empty_results(self, mock_chroma, mock_client, mock_embeddings):
        """Test retrieving with no results."""
        settings = Settings()

        mock_embeddings.return_value = MagicMock()
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        mock_vectorstore.similarity_search_with_score.return_value = []

        manager = RAGManager(settings)

        results = manager.retrieve_relevant_documents("test query")

        assert len(results) == 0
