"""
Document chunking module using LlamaIndex for text segmentation.

This module handles intelligent text chunking using LlamaIndex's
built-in chunking capabilities for optimal embedding generation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import TextNode

from config.settings import settings
from config.logging import get_logger
from .semantic_chunker import SemanticChunker

logger = get_logger(__name__)


@dataclass
class ChunkingStats:
    """Statistics about chunking operations."""
    original_documents: int
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    chunk_overlap: int


class DocumentChunker:
    """
    Chunks documents using LlamaIndex for optimal embedding generation.
    
    Uses simple and stable chunking methods with configurable
    chunk size and overlap for consistent results.
    """
    
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = 50  # Minimum chunk size for semantic chunking
        self.max_chunk_size = 1000  # Maximum chunk size for semantic chunking
        
        # Initialize LlamaIndex text splitters
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=settings.chunk_separator
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=settings.chunk_separator
        )
        
        logger.info(f"Document chunker initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_document(self, document: Document, method: str = "semantic") -> List[TextNode]:
        """
        Chunk a single document into smaller pieces.
        
        Args:
            document: LlamaIndex Document object
            method: Chunking method ("semantic", "sentence", or "token")
            
        Returns:
            List of TextNode objects representing chunks
        """
        if not document.text:
            logger.warning(f"Document {document.doc_id} has no text content")
            return []
        
        try:
            # Use semantic chunking as primary method
            if method == "semantic":
                semantic_chunker = SemanticChunker(
                    sentence_overlap=1,
                    min_chunk_size=self.min_chunk_size,
                    max_chunk_size=self.max_chunk_size
                )
                nodes = semantic_chunker.chunk_document(document)
                
                # If semantic chunking fails or produces no chunks, fallback to sentence
                if not nodes:
                    logger.warning(f"Semantic chunking produced no chunks for {document.doc_id}, falling back to sentence")
                    nodes = self.sentence_splitter.get_nodes_from_documents([document])
                    for node in nodes:
                        node.metadata.update({
                            'source_document': document.doc_id,
                            'chunking_method': 'sentence_fallback',
                            'original_length': len(document.text),
                            **document.metadata
                        })
                
                logger.info(f"Chunked document {document.doc_id} into {len(nodes)} chunks using semantic method")
                return nodes
            
            # Fallback to original methods
            elif method == "sentence":
                nodes = self.sentence_splitter.get_nodes_from_documents([document])
            elif method == "token":
                nodes = self.token_splitter.get_nodes_from_documents([document])
            else:
                logger.warning(f"Unknown chunking method: {method}, using sentence")
                nodes = self.sentence_splitter.get_nodes_from_documents([document])
            
            # Add document metadata to each node
            for node in nodes:
                node.metadata.update({
                    'source_document': document.doc_id,
                    'chunking_method': method,
                    'original_length': len(document.text),
                    **document.metadata
                })
            
            logger.info(f"Chunked document {document.doc_id} into {len(nodes)} chunks using {method} method")
            return nodes
            
        except Exception as e:
            logger.error(f"Error chunking document {document.doc_id}: {e}")
            return []
    
    def chunk_documents(self, documents: List[Document], method: str = "sentence") -> List[TextNode]:
        """
        Chunk multiple documents into smaller pieces.
        
        Args:
            documents: List of LlamaIndex Document objects
            method: Chunking method ("sentence" or "token")
            
        Returns:
            List of TextNode objects representing all chunks
        """
        all_nodes = []
        
        for document in documents:
            try:
                nodes = self.chunk_document(document, method)
                all_nodes.extend(nodes)
            except Exception as e:
                logger.error(f"Error chunking document {document.doc_id}: {e}")
                continue
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_nodes)} total chunks")
        return all_nodes
    
    def get_chunk_texts(self, nodes: List[TextNode]) -> List[str]:
        """
        Extract text content from chunk nodes.
        
        Args:
            nodes: List of TextNode objects
            
        Returns:
            List of text strings from chunks
        """
        return [node.text for node in nodes if node.text.strip()]
    
    def get_chunk_metadata(self, nodes: List[TextNode]) -> List[Dict[str, Any]]:
        """
        Extract metadata from chunk nodes.
        
        Args:
            nodes: List of TextNode objects
            
        Returns:
            List of metadata dictionaries
        """
        return [node.metadata for node in nodes]
    
    def filter_chunks_by_size(self, nodes: List[TextNode], min_size: int = 50, max_size: int = None) -> List[TextNode]:
        """
        Filter chunks based on size constraints.
        
        Args:
            nodes: List of TextNode objects
            min_size: Minimum chunk size in characters
            max_size: Maximum chunk size in characters (None for no limit)
            
        Returns:
            Filtered list of TextNode objects
        """
        filtered_nodes = []
        
        for node in nodes:
            chunk_size = len(node.text)
            
            if chunk_size < min_size:
                logger.debug(f"Filtering out chunk with {chunk_size} chars (below minimum {min_size})")
                continue
            
            if max_size and chunk_size > max_size:
                logger.debug(f"Filtering out chunk with {chunk_size} chars (above maximum {max_size})")
                continue
            
            filtered_nodes.append(node)
        
        logger.info(f"Filtered chunks: {len(nodes)} -> {len(filtered_nodes)}")
        return filtered_nodes
    
    def get_chunking_stats(self, original_documents: List[Document], chunks: List[TextNode]) -> ChunkingStats:
        """
        Get statistics about chunking operations.
        
        Args:
            original_documents: Original documents before chunking
            chunks: Resulting chunks
            
        Returns:
            ChunkingStats object
        """
        if not chunks:
            return ChunkingStats(
                original_documents=len(original_documents),
                total_chunks=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                chunk_overlap=self.chunk_overlap
            )
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        
        return ChunkingStats(
            original_documents=len(original_documents),
            total_chunks=len(chunks),
            avg_chunk_size=sum(chunk_sizes) / len(chunk_sizes),
            min_chunk_size=min(chunk_sizes),
            max_chunk_size=max(chunk_sizes),
            chunk_overlap=self.chunk_overlap
        )
    
    def merge_small_chunks(self, nodes: List[TextNode], min_size: int = 100) -> List[TextNode]:
        """
        Merge small chunks with adjacent chunks to meet minimum size requirements.
        
        Args:
            nodes: List of TextNode objects
            min_size: Minimum size for chunks
            
        Returns:
            List of merged TextNode objects
        """
        if not nodes:
            return []
        
        merged_nodes = []
        current_chunk = nodes[0].text
        current_metadata = nodes[0].metadata.copy()
        
        for i in range(1, len(nodes)):
            node = nodes[i]
            
            # If current chunk is small and nodes are from same document, merge
            if (len(current_chunk) < min_size and 
                current_metadata.get('source_document') == node.metadata.get('source_document')):
                
                current_chunk += f"\n\n{node.text}"
                # Merge metadata (keep first chunk's metadata as base)
                current_metadata.update({
                    'merged_chunks': current_metadata.get('merged_chunks', 1) + 1
                })
            else:
                # Save current chunk and start new one
                merged_nodes.append(TextNode(
                    text=current_chunk,
                    metadata=current_metadata
                ))
                current_chunk = node.text
                current_metadata = node.metadata.copy()
        
        # Add the last chunk
        merged_nodes.append(TextNode(
            text=current_chunk,
            metadata=current_metadata
        ))
        
        logger.info(f"Merged chunks: {len(nodes)} -> {len(merged_nodes)}")
        return merged_nodes
    
    def validate_chunks(self, nodes: List[TextNode]) -> bool:
        """
        Validate that chunks are suitable for embedding.
        
        Args:
            nodes: List of TextNode objects
            
        Returns:
            True if chunks are valid, False otherwise
        """
        if not nodes:
            return False
        
        for i, node in enumerate(nodes):
            if not node.text or not node.text.strip():
                logger.warning(f"Chunk {i} has empty text")
                return False
            
            if len(node.text.strip()) < 10:
                logger.warning(f"Chunk {i} is too short: {len(node.text)} chars")
                return False
        
        return True 