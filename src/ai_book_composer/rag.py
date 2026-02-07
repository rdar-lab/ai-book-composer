"""RAG (Retrieval Augmented Generation) module for AI Book Composer.

This module provides vector database functionality for efficient document retrieval.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings

logger = logging.getLogger(__name__)

# Default chunk size for document splitting
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_ALLOWED_DISTANCE = 0.9


class RAGManager:
    """Manages vector database operations for document retrieval."""

    def __init__(self, settings: Settings):
        """Initialize RAG manager with settings.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = None
        self._init_components()

    def _init_components(self):
        """Initialize embedding model, text splitter, and vector store."""
        logger.info("Initializing RAG components...")

        self.max_allowed_distance = getattr(self.settings.rag, 'max_allowed_distance', DEFAULT_MAX_ALLOWED_DISTANCE)

        # Initialize embeddings model (using local sentence-transformers)
        model_name = getattr(self.settings.rag, 'embedding_model', 'all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize text splitter for chunking documents
        chunk_size = getattr(self.settings.rag, 'chunk_size', DEFAULT_CHUNK_SIZE)
        chunk_overlap = getattr(self.settings.rag, 'chunk_overlap', DEFAULT_CHUNK_OVERLAP)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialize in-memory ChromaDB
        logger.info("Initializing in-memory vector database...")
        client = chromadb.Client(
            ChromaSettings(
                is_persistent=False,
                anonymized_telemetry=False
            )
        )

        self.vectorstore = Chroma(
            client=client,
            collection_name="document_collection",
            embedding_function=self.embeddings
        )

        logger.info("RAG components initialized successfully")

    def ingest_documents(self, gathered_content: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest documents into the vector database.
        
        Args:
            gathered_content: Dictionary of file content keyed by file path
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Ingesting {len(gathered_content)} documents into vector database...")

        texts = []
        metadatas = []

        for file_path, content_info in gathered_content.items():
            content = content_info.get("content", "")
            file_name = content_info.get("name", Path(file_path).name)
            file_type = content_info.get("type", "unknown")

            # Skip only completely empty content
            if not content:
                logger.warning(f"Skipping {file_name} - no content")
                continue

            # Split content into chunks (even if small, it will be processed)
            chunks = self.text_splitter.split_text(content)

            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })

        if not texts:
            logger.warning("No valid content to ingest")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "status": "no_content"
            }

        # Add documents to vector store
        logger.info(f"Adding {len(texts)} text chunks to vector database...")
        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )

        logger.info(f"Successfully ingested {len(gathered_content)} documents ({len(texts)} chunks)")

        return {
            "total_documents": len(gathered_content),
            "total_chunks": len(texts),
            "status": "success"
        }

    def retrieve_relevant_documents(
            self,
            query: str,
            k: int = 5,
            filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (default: 5)
            filter_dict: Optional metadata filters
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []

        logger.info(f"Retrieving relevant documents for query: {query[:100]}...")

        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )

            # Format results
            documents = []
            for doc, score in results:
                if score <= self.max_allowed_distance:
                    documents.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": float(score)
                    })

            logger.info(f"Retrieved {len(documents)} relevant documents")
            return documents

        except Exception as e:
            logger.exception(f"Error retrieving documents: {str(e)}")
            return []
