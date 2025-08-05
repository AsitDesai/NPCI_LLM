"""
Settings configuration for the RAG System.

This module handles all application settings, loading from environment
variables using Pydantic's BaseSettings for type safety and validation.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = Field(default="localhost", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    debug: bool = Field(default=True, env="DEBUG")
    
    # LLM Configuration (Mistral for generation)
    mistral_api_key: Optional[str] = Field(default=None, env="MISTRAL_API_KEY")
    mistral_model: str = Field(default="mistral-small-latest", env="MISTRAL_MODEL")
    # Fallback OpenAI (optional)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    
    # Embedding Configuration (LlamaIndex HuggingFace integration)
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    embedding_model_dimension: int = Field(default=384, env="EMBEDDING_MODEL_DIMENSION")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    
    # Qdrant Database Settings
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    vector_db_name: str = Field(default="rag_embeddings", env="VECTOR_DB_NAME")
    vector_db_dimension: int = Field(default=384, env="VECTOR_DB_DIMENSION")
    vector_db_metric: str = Field(default="cosine", env="VECTOR_DB_METRIC")
    
    # Document Storage Settings
    reference_documents_dir: str = Field(default="./reference_documents", env="REFERENCE_DOCUMENTS_DIR")
    
    # Local Storage Settings
    storage_dir: str = Field(default="./storage", env="STORAGE_DIR")
    embeddings_dir: str = Field(default="./storage/embeddings", env="EMBEDDINGS_DIR")
    upload_dir: str = Field(default="./storage/uploads", env="UPLOAD_DIR")
    
    # LlamaIndex Settings
    llama_index_cache_dir: str = Field(default="./storage/cache", env="LLAMA_INDEX_CACHE_DIR")
    llama_index_persist_dir: str = Field(default="./storage/persist", env="LLAMA_INDEX_PERSIST_DIR")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="./logs/rag_system.log", env="LOG_FILE")
    
    # LlamaIndex Chunking Settings (Simple and Stable)
    chunk_size: int = Field(default=1024, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    chunk_separator: str = Field(default="\n\n", env="CHUNK_SEPARATOR")
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")
    retrieval_score_threshold: float = Field(default=0.3, env="RETRIEVAL_SCORE_THRESHOLD")
    context_max_tokens: int = Field(default=4000, env="CONTEXT_MAX_TOKENS")
    context_overlap: int = Field(default=200, env="CONTEXT_OVERLAP")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        directories = [
            self.reference_documents_dir,
            self.storage_dir,
            self.embeddings_dir,
            self.upload_dir,
            os.path.dirname(self.log_file),
            self.llama_index_cache_dir,
            self.llama_index_persist_dir,
        ]
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        # Handle both local and cloud URLs
        if self.qdrant_host.startswith(('http://', 'https://')):
            return self.qdrant_host
        else:
            return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    @property
    def api_url(self) -> str:
        """Get API base URL."""
        return f"http://{self.api_host}:{self.api_port}"


# Global settings instance
settings = Settings() 