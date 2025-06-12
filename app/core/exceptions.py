"""Custom exceptions for the RAG application."""

class RAGServiceError(Exception):
    """Base exception for RAG service errors."""
    pass

class ConfigurationError(RAGServiceError):
    """Raised when configuration is invalid or incomplete."""
    pass

class EmbeddingModelError(RAGServiceError):
    """Raised when embedding model operations fail."""
    pass

class VectorStoreError(RAGServiceError):
    """Raised when vector store operations fail."""
    pass

class LLMProviderError(RAGServiceError):
    """Raised when LLM provider operations fail."""
    pass

class RetrieverError(RAGServiceError):
    """Raised when retrieval operations fail."""
    pass

class QueryProcessingError(RAGServiceError):
    """Raised when query processing fails."""
    pass
