"""
LlamaIndex embedding wrapper for text-to-vector conversion.

This module provides a wrapper around LlamaIndex's embedding capabilities
for consistent embedding generation across the RAG system.
"""

import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .models import EmbeddingModelConfig, get_default_embedding_config
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics about embedding operations."""
    total_texts: int
    total_embeddings: int
    total_tokens: int
    processing_time: float
    avg_time_per_text: float
    model_name: str
    model_dimension: int


class LlamaIndexEmbedder:
    """
    LlamaIndex wrapper for embedding generation.
    
    This class provides a simple interface for generating embeddings
    using LlamaIndex's HuggingFace integration.
    """
    
    def __init__(self, config: Optional[EmbeddingModelConfig] = None):
        """
        Initialize the embedder.
        
        Args:
            config: Embedding model configuration
        """
        self.config = config or get_default_embedding_config()
        
        # Initialize LlamaIndex HuggingFace embedding
        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.config.model_name,
            cache_folder=self.config.model_params.get("cache_folder") if self.config.model_params else None,
            trust_remote_code=self.config.model_params.get("trust_remote_code", True) if self.config.model_params else True,
            device=self.config.device
        )
        
        logger.info(f"LlamaIndex embedder initialized with model: {self.config.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            start_time = time.time()
            
            # Generate embedding using LlamaIndex
            embedding = self.embedding_model.get_text_embedding(text)
            
            processing_time = time.time() - start_time
            
            logger.debug(f"Generated embedding for text ({len(text)} chars) in {processing_time:.3f}s")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            start_time = time.time()
            
            # Generate embeddings using LlamaIndex
            embeddings = self.embedding_model.get_text_embedding_batch(texts)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated {len(embeddings)} embeddings in {processing_time:.3f}s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings for {len(texts)} texts: {e}")
            raise
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for LlamaIndex documents.
        
        Args:
            documents: List of LlamaIndex Document objects
            
        Returns:
            List of embedding vectors
        """
        texts = [doc.text for doc in documents]
        return self.embed_texts(texts)
    
    def embed_nodes(self, nodes: List[TextNode]) -> List[List[float]]:
        """
        Generate embeddings for LlamaIndex text nodes.
        
        Args:
            nodes: List of LlamaIndex TextNode objects
            
        Returns:
            List of embedding vectors
        """
        texts = [node.text for node in nodes]
        return self.embed_texts(texts)
    
    def get_embedding_stats(self, texts: List[str], processing_time: float) -> EmbeddingStats:
        """
        Calculate embedding statistics.
        
        Args:
            texts: List of texts that were embedded
            processing_time: Total processing time
            
        Returns:
            EmbeddingStats object with statistics
        """
        total_texts = len(texts)
        total_embeddings = total_texts
        total_tokens = sum(len(text.split()) for text in texts)  # Approximate token count
        avg_time_per_text = processing_time / total_texts if total_texts > 0 else 0
        
        return EmbeddingStats(
            total_texts=total_texts,
            total_embeddings=total_embeddings,
            total_tokens=total_tokens,
            processing_time=processing_time,
            avg_time_per_text=avg_time_per_text,
            model_name=self.config.model_name,
            model_dimension=self.config.model_dimension
        )
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding is correct.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if embedding is valid, False otherwise
        """
        if not embedding:
            return False
        
        if len(embedding) != self.config.model_dimension:
            logger.warning(f"Embedding dimension mismatch: expected {self.config.model_dimension}, got {len(embedding)}")
            return False
        
        # Check for NaN or infinite values
        if any(not isinstance(x, (int, float)) or not (x == x) for x in embedding):
            logger.warning("Embedding contains invalid values (NaN or infinite)")
            return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.config.model_name,
            "model_dimension": self.config.model_dimension,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "device": self.config.device,
            "normalize": self.config.normalize
        } 