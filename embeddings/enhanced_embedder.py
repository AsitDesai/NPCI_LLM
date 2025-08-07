#!/usr/bin/env python3
"""
Enhanced Embedder

This module implements enhanced embedding generation with token restrictions and mean pooling.
Uses sentence-transformers/all-MiniLM-L6-v2 with optimized settings for RAG performance.
"""

import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 256
    device: str = "auto"
    normalize_embeddings: bool = True
    use_mean_pooling: bool = True


@dataclass
class EmbeddingStats:
    """Statistics for embedding generation."""
    input_length: int
    token_count: int
    embedding_dimension: int
    embedding_variance: float
    quality_score: float


class EnhancedEmbedder:
    """
    Enhanced embedder with token restrictions and mean pooling.
    
    Implements the architecture requirements:
    - Uses sentence-transformers/all-MiniLM-L6-v2
    - Token restrictions (max 256 tokens)
    - Mean pooling for embedding generation
    - Quality assessment for embeddings
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the enhanced embedder."""
        self.config = config or EmbeddingConfig()
        
        # Set device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        # Load model
        self.model = SentenceTransformer(self.config.model_name, device=self.device)
        self.tokenizer = self.model.tokenizer
        
        logger.info(f"Enhanced embedder initialized with {self.config.model_name} on {self.device}")
    
    def truncate_text(self, text: str) -> str:
        """
        Truncate text to respect token limit.
        
        Args:
            text: Input text
            
        Returns:
            Truncated text within token limit
        """
        # Tokenize to check length
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) <= self.config.max_tokens:
            return text
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:self.config.max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        logger.debug(f"Truncated text from {len(tokens)} to {len(truncated_tokens)} tokens")
        return truncated_text
    
    def generate_embedding(self, text: str) -> tuple:
        """
        Generate embedding with token restrictions and mean pooling.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (embedding, stats)
        """
        # Truncate text to respect token limit
        truncated_text = self.truncate_text(text)
        
        # Tokenize for statistics
        tokens = self.tokenizer.encode(truncated_text, add_special_tokens=True)
        
        # Generate embedding using mean pooling
        if self.config.use_mean_pooling:
            embedding = self._generate_mean_pooled_embedding(truncated_text)
        else:
            embedding = self.model.encode(truncated_text, convert_to_tensor=True)
        
        # Calculate statistics
        stats = self._calculate_embedding_stats(truncated_text, tokens, embedding)
        
        return embedding, stats
    
    def _generate_mean_pooled_embedding(self, text: str) -> torch.Tensor:
        """
        Generate embedding using mean pooling of token embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Mean pooled embedding tensor
        """
        # Use the SentenceTransformer's encode method with mean pooling
        embedding = self.model.encode(text, convert_to_tensor=True, device=self.device)
        return embedding
    
    def _calculate_embedding_stats(self, text: str, tokens: List[int], 
                                 embedding: torch.Tensor) -> EmbeddingStats:
        """
        Calculate embedding statistics for quality assessment.
        
        Args:
            text: Input text
            tokens: Token list
            embedding: Generated embedding
            
        Returns:
            Embedding statistics
        """
        # Basic stats
        input_length = len(text)
        token_count = len(tokens)
        embedding_dimension = embedding.shape[-1]
        
        # Calculate embedding variance (quality indicator)
        embedding_array = embedding.cpu().numpy()
        embedding_variance = np.var(embedding_array)
        
        # Calculate quality score (0.0 to 1.0)
        # Higher score for appropriate token count and embedding variance
        token_score = min(token_count / self.config.max_tokens, 1.0)
        variance_score = min(embedding_variance / 0.1, 1.0)  # Normalize variance
        quality_score = (token_score + variance_score) / 2.0
        
        return EmbeddingStats(
            input_length=input_length,
            token_count=token_count,
            embedding_dimension=embedding_dimension,
            embedding_variance=embedding_variance,
            quality_score=quality_score
        )
    
    def generate_embeddings_batch(self, texts: List[str]) -> tuple:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (embeddings, stats_list)
        """
        embeddings = []
        stats_list = []
        
        for text in texts:
            embedding, stats = self.generate_embedding(text)
            embeddings.append(embedding)
            stats_list.append(stats)
        
        # Stack embeddings
        if embeddings:
            stacked_embeddings = torch.stack(embeddings)
        else:
            stacked_embeddings = torch.empty(0)
        
        return stacked_embeddings, stats_list
    
    def calculate_similarity(self, embedding1: torch.Tensor, 
                           embedding2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        return cos_sim(embedding1, embedding2).item()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of generated embeddings."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "use_mean_pooling": self.config.use_mean_pooling,
            "normalize_embeddings": self.config.normalize_embeddings
        }
