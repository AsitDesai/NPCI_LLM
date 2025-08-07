"""
Context builder for the RAG System.

This module handles assembling retrieved document chunks into
context for LLM generation, including token management and
metadata preservation.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .enhanced_retriever import RetrievalResult
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ContextInfo:
    """Information about the built context."""
    context: str
    total_tokens: int
    num_chunks: int
    sources: List[str]
    chunk_scores: List[float]
    metadata: Dict[str, Any]


class ContextBuilder:
    """
    Context builder for assembling retrieved chunks.
    
    This class handles building context from retrieval results,
    including token management, chunk ordering, and metadata preservation.
    """
    
    def __init__(self, 
                 max_tokens: Optional[int] = None,
                 overlap_tokens: Optional[int] = None):
        """
        Initialize the context builder.
        
        Args:
            max_tokens: Maximum tokens for context
            overlap_tokens: Token overlap between chunks
        """
        self.max_tokens = max_tokens or getattr(settings, 'context_max_tokens', 4000)
        self.overlap_tokens = overlap_tokens or getattr(settings, 'context_overlap', 200)
        
        # Simple token estimation (4 chars per token is a rough estimate)
        self.chars_per_token = 4
        
        logger.info(f"Context builder initialized with max_tokens={self.max_tokens}, overlap={self.overlap_tokens}")
    
    def build_context(self, 
                     results: List[RetrievalResult],
                     query: str = "",
                     include_sources: bool = True) -> ContextInfo:
        """
        Build context from retrieval results.
        
        Args:
            results: List of retrieval results
            query: Original query (for context)
            include_sources: Whether to include source information
            
        Returns:
            ContextInfo with assembled context and metadata
        """
        try:
            if not results:
                logger.warning("No retrieval results to build context from")
                return ContextInfo(
                    context="",
                    total_tokens=0,
                    num_chunks=0,
                    sources=[],
                    chunk_scores=[],
                    metadata={}
                )
            
            # Sort results by score (descending)
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            
            # Build context with token management
            context_parts = []
            total_tokens = 0
            included_chunks = []
            sources = set()
            chunk_scores = []
            
            for result in sorted_results:
                # Estimate tokens for this chunk
                chunk_tokens = self._estimate_tokens(result.text)
                
                # Check if adding this chunk would exceed limit
                if total_tokens + chunk_tokens > self.max_tokens:
                    logger.debug(f"Stopping context building at {total_tokens} tokens (limit: {self.max_tokens})")
                    break
                
                # Add chunk to context
                chunk_text = self._format_chunk(result, include_sources)
                context_parts.append(chunk_text)
                
                # Update tracking
                total_tokens += chunk_tokens
                included_chunks.append(result)
                source_file = result.metadata.get('source_file', 'Unknown')
                sources.add(source_file)
                chunk_scores.append(result.score)
            
            # Join context parts
            context = "\n\n".join(context_parts)
            
            # Add query context if provided
            if query and total_tokens < self.max_tokens:
                query_context = f"Query: {query}\n\n"
                query_tokens = self._estimate_tokens(query_context)
                if total_tokens + query_tokens <= self.max_tokens:
                    context = query_context + context
                    total_tokens += query_tokens
            
            # Create context info
            context_info = ContextInfo(
                context=context,
                total_tokens=total_tokens,
                num_chunks=len(included_chunks),
                sources=list(sources),
                chunk_scores=chunk_scores,
                metadata={
                    "max_tokens": self.max_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "query": query,
                    "included_chunks": len(included_chunks),
                    "total_results": len(results)
                }
            )
            
            logger.info(f"Built context with {total_tokens} tokens from {len(included_chunks)} chunks")
            logger.debug(f"Sources: {list(sources)}")
            
            return context_info
            
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return ContextInfo(
                context="",
                total_tokens=0,
                num_chunks=0,
                sources=[],
                chunk_scores=[],
                metadata={"error": str(e)}
            )
    
    def _format_chunk(self, result: RetrievalResult, include_sources: bool) -> str:
        """
        Format a retrieval result chunk for context.
        
        Args:
            result: Retrieval result to format
            include_sources: Whether to include source information
            
        Returns:
            Formatted chunk text
        """
        text = result.text.strip()
        
        if include_sources:
            # Extract file name from source document
            source_file = result.metadata.get('source_file', 'Unknown')
            source_name = source_file.split('/')[-1] if source_file else "Unknown"
            
            # Add source information
            formatted_text = f"[Source: {source_name}, Score: {result.score:.3f}]\n{text}"
        else:
            formatted_text = text
        
        return formatted_text
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation: 4 characters per token
        return len(text) // self.chars_per_token
    
    def build_context_with_metadata(self, 
                                  results: List[RetrievalResult],
                                  query: str = "") -> Dict[str, Any]:
        """
        Build context with detailed metadata.
        
        Args:
            results: List of retrieval results
            query: Original query
            
        Returns:
            Dictionary with context and detailed metadata
        """
        context_info = self.build_context(results, query, include_sources=True)
        
        # Add detailed metadata
        metadata = {
            "context": context_info.context,
            "total_tokens": context_info.total_tokens,
            "num_chunks": context_info.num_chunks,
            "sources": context_info.sources,
            "chunk_scores": context_info.chunk_scores,
            "avg_score": sum(context_info.chunk_scores) / len(context_info.chunk_scores) if context_info.chunk_scores else 0.0,
            "max_score": max(context_info.chunk_scores) if context_info.chunk_scores else 0.0,
            "min_score": min(context_info.chunk_scores) if context_info.chunk_scores else 0.0,
            "query": query,
            "query_tokens": self._estimate_tokens(query),
            "context_utilization": context_info.total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0
        }
        
        return metadata
    
    def validate_context(self, context_info: ContextInfo) -> bool:
        """
        Validate the built context.
        
        Args:
            context_info: Context information to validate
            
        Returns:
            True if context is valid, False otherwise
        """
        if not context_info.context.strip():
            logger.warning("Context is empty")
            return False
        
        if context_info.total_tokens > self.max_tokens:
            logger.warning(f"Context exceeds token limit: {context_info.total_tokens} > {self.max_tokens}")
            return False
        
        if context_info.num_chunks == 0:
            logger.warning("No chunks included in context")
            return False
        
        logger.debug(f"Context validation passed: {context_info.total_tokens} tokens, {context_info.num_chunks} chunks")
        return True
    
    def get_context_stats(self, context_info: ContextInfo) -> Dict[str, Any]:
        """
        Get statistics about the built context.
        
        Args:
            context_info: Context information
            
        Returns:
            Dictionary with context statistics
        """
        return {
            "total_tokens": context_info.total_tokens,
            "num_chunks": context_info.num_chunks,
            "num_sources": len(context_info.sources),
            "avg_score": sum(context_info.chunk_scores) / len(context_info.chunk_scores) if context_info.chunk_scores else 0.0,
            "max_score": max(context_info.chunk_scores) if context_info.chunk_scores else 0.0,
            "min_score": min(context_info.chunk_scores) if context_info.chunk_scores else 0.0,
            "context_length": len(context_info.context),
            "token_utilization": context_info.total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            "sources": context_info.sources
        } 