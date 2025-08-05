"""
Embedding model configurations for the RAG System.

This module defines embedding model configurations and settings
for consistent embedding generation across the system.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models."""
    
    model_name: str
    model_dimension: int
    batch_size: int
    max_length: int
    device: str = "cpu"
    normalize: bool = True
    
    # Model-specific parameters
    model_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_dimension <= 0:
            raise ValueError("Model dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_length <= 0:
            raise ValueError("Max length must be positive")
        
        logger.info(f"Embedding model config initialized: {self.model_name} ({self.model_dimension}d)")


def get_default_embedding_config() -> EmbeddingModelConfig:
    """
    Get the default embedding configuration from settings.
    
    Returns:
        EmbeddingModelConfig with default settings
    """
    return EmbeddingModelConfig(
        model_name=settings.embedding_model_name,
        model_dimension=settings.embedding_model_dimension,
        batch_size=settings.embedding_batch_size,
        max_length=512,  # Standard max length for sentence transformers
        device="cpu",  # Can be changed to "cuda" if GPU is available
        normalize=True,  # Normalize embeddings for cosine similarity
        model_params={
            "trust_remote_code": True,
            "cache_folder": settings.llama_index_cache_dir
        }
    )


def get_embedding_config_for_model(model_name: str) -> EmbeddingModelConfig:
    """
    Get embedding configuration for a specific model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        EmbeddingModelConfig for the specified model
    """
    # Model-specific configurations
    model_configs = {
        "sentence-transformers/all-MiniLM-L6-v2": EmbeddingModelConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_dimension=384,
            batch_size=32,
            max_length=256,  # Optimized for this model
            device="cpu",
            normalize=True
        ),
        "sentence-transformers/all-mpnet-base-v2": EmbeddingModelConfig(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_dimension=768,
            batch_size=16,  # Larger model, smaller batch size
            max_length=384,
            device="cpu",
            normalize=True
        ),
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": EmbeddingModelConfig(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_dimension=384,
            batch_size=32,
            max_length=256,
            device="cpu",
            normalize=True
        )
    }
    
    if model_name in model_configs:
        return model_configs[model_name]
    else:
        logger.warning(f"Unknown model {model_name}, using default config")
        return get_default_embedding_config()


# Global default configuration
default_config = get_default_embedding_config() 