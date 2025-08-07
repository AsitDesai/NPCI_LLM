#!/usr/bin/env python3
"""
Enhanced Document Chunker

This module implements intelligent document chunking based on document format detection.
Supports Q&A, list-based, and paragraphic document types with optimized chunking strategies.
"""

import re
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.schema import BaseNode
import structlog

logger = structlog.get_logger(__name__)


class DocumentType(Enum):
    """Document type classification."""
    QA = "qa"
    LIST = "list"
    PARAGRAPHIC = "paragraphic"
    UNKNOWN = "unknown"


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    max_tokens: int = 256
    sentence_overlap: int = 1
    min_chunk_size: int = 50
    max_chunk_size: int = 1000
    include_adjacent_items: bool = True


@dataclass
class ChunkMetadata:
    """Metadata for document chunks."""
    document_type: DocumentType
    chunk_index: int
    total_chunks: int
    source_file: str
    structural_tags: List[str]
    qa_pair: Optional[Dict[str, str]] = None
    list_item_index: Optional[int] = None


class EnhancedChunker:
    """
    Enhanced document chunker with format detection and type-specific strategies.
    
    Implements the architecture for handling Q&A, list-based, and paragraphic documents
    with optimized chunking for each format.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize the enhanced chunker."""
        self.config = config or ChunkingConfig()
        self._setup_patterns()
        logger.info("Enhanced chunker initialized")
    
    def _setup_patterns(self):
        """Setup regex patterns for document type detection."""
        # Q&A patterns
        self.qa_patterns = [
            re.compile(r'^\*\*Q\d+:\s*(.+?)\*\*$', re.MULTILINE),
            re.compile(r'^Q\d+:\s*(.+?)$', re.MULTILINE),
            re.compile(r'^Question\s*\d*:\s*(.+?)$', re.MULTILINE),
            re.compile(r'^A:\s*(.+)$', re.MULTILINE),
            re.compile(r'^Answer:\s*(.+)$', re.MULTILINE),
        ]
        
        # List patterns
        self.list_patterns = [
            re.compile(r'^\d+\.\s', re.MULTILINE),  # Numbered lists
            re.compile(r'^[-*+]\s', re.MULTILINE),  # Bullet points
            re.compile(r'^DO\'s?:', re.MULTILINE),  # DO's
            re.compile(r'^DON\'T\'s?:', re.MULTILINE),  # DON'Ts
        ]
        
        # Paragraph patterns
        self.paragraph_patterns = [
            re.compile(r'\n\s*\n', re.MULTILINE),  # Double newlines
            re.compile(r'[.!?]\s+[A-Z]', re.MULTILINE),  # Sentence boundaries
        ]
    
    def detect_document_type(self, text: str) -> DocumentType:
        """
        Detect document type using rule-based checks and structural indicators.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Detected document type
        """
        # Check for Q&A patterns
        qa_matches = sum(1 for pattern in self.qa_patterns if pattern.search(text))
        if qa_matches >= 2:  # Need both Q and A patterns
            logger.info("Document classified as Q&A format")
            return DocumentType.QA
        
        # Check for list patterns
        list_matches = sum(1 for pattern in self.list_patterns if pattern.search(text))
        if list_matches >= 3:  # Multiple list items
            logger.info("Document classified as list format")
            return DocumentType.LIST
        
        # Check for paragraphic structure
        paragraph_matches = sum(1 for pattern in self.paragraph_patterns if pattern.search(text))
        if paragraph_matches >= 2:
            logger.info("Document classified as paragraphic format")
            return DocumentType.PARAGRAPHIC
        
        # Default to paragraphic for unknown formats
        logger.info("Document classified as paragraphic (default)")
        return DocumentType.PARAGRAPHIC
    
    def chunk_qa_document(self, text: str, source_file: str) -> List[TextNode]:
        """
        Chunk Q&A document keeping each Q&A pair as atomic unit.
        
        Args:
            text: Document text
            source_file: Source file name
            
        Returns:
            List of TextNode objects with Q&A pairs
        """
        chunks = []
        qa_pairs = self._extract_qa_pairs(text)
        
        for i, (question, answer) in enumerate(qa_pairs):
            # Combine Q&A into single chunk
            chunk_text = f"Q: {question}\nA: {answer}"
            
            # Create metadata
            metadata = ChunkMetadata(
                document_type=DocumentType.QA,
                chunk_index=i,
                total_chunks=len(qa_pairs),
                source_file=source_file,
                structural_tags=["qa_pair"],
                qa_pair={"question": question, "answer": answer}
            )
            
            # Create TextNode
            node = TextNode(
                text=chunk_text,
                metadata={
                    "document_type": metadata.document_type.value,
                    "chunk_index": metadata.chunk_index,
                    "total_chunks": metadata.total_chunks,
                    "source_file": metadata.source_file,
                    "structural_tags": metadata.structural_tags,
                    "qa_pair": metadata.qa_pair
                }
            )
            chunks.append(node)
        
        logger.info(f"Created {len(chunks)} Q&A chunks from {source_file}")
        return chunks
    
    def _extract_qa_pairs(self, text: str) -> List[tuple]:
        """Extract Q&A pairs from text."""
        qa_pairs = []
        lines = text.split('\n')
        current_question = None
        current_answer = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for question patterns
            question_match = re.match(r'^\*\*Q(\d+):\s*(.+?)\*\*$', line)
            if question_match:
                # Save previous pair if exists
                if current_question and current_answer:
                    qa_pairs.append((current_question, ' '.join(current_answer)))
                
                # Start new pair
                current_question = question_match.group(2)
                current_answer = []
                continue
            
            # Check for answer patterns
            answer_match = re.match(r'^A:\s*(.+)$', line)
            if answer_match and current_question:
                current_answer.append(answer_match.group(1))
                continue
            
            # If we have a question, accumulate answer lines
            if current_question:
                current_answer.append(line)
        
        # Add final pair
        if current_question and current_answer:
            qa_pairs.append((current_question, ' '.join(current_answer)))
        
        return qa_pairs
    
    def chunk_list_document(self, text: str, source_file: str) -> List[TextNode]:
        """
        Chunk list document by splitting on list markers.
        
        Args:
            text: Document text
            source_file: Source file name
            
        Returns:
            List of TextNode objects with list items
        """
        chunks = []
        list_items = self._extract_list_items(text)
        
        for i, item in enumerate(list_items):
            # Include adjacent item if configured
            if self.config.include_adjacent_items and i < len(list_items) - 1:
                chunk_text = f"{item}\n\n{list_items[i + 1]}"
                structural_tags = ["list_item", "with_adjacent"]
            else:
                chunk_text = item
                structural_tags = ["list_item"]
            
            # Create metadata
            metadata = ChunkMetadata(
                document_type=DocumentType.LIST,
                chunk_index=i,
                total_chunks=len(list_items),
                source_file=source_file,
                structural_tags=structural_tags,
                list_item_index=i
            )
            
            # Create TextNode
            node = TextNode(
                text=chunk_text,
                metadata={
                    "document_type": metadata.document_type.value,
                    "chunk_index": metadata.chunk_index,
                    "total_chunks": metadata.total_chunks,
                    "source_file": metadata.source_file,
                    "structural_tags": metadata.structural_tags,
                    "list_item_index": metadata.list_item_index
                }
            )
            chunks.append(node)
        
        logger.info(f"Created {len(chunks)} list chunks from {source_file}")
        return chunks
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from text."""
        items = []
        lines = text.split('\n')
        current_item = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for list markers
            if re.match(r'^\d+\.\s', line) or re.match(r'^[-*+]\s', line) or \
               re.match(r'^DO\'s?:', line) or re.match(r'^DON\'T\'s?:', line):
                # Save previous item
                if current_item:
                    items.append('\n'.join(current_item))
                
                # Start new item
                current_item = [line]
            else:
                # Continue current item
                if current_item:
                    current_item.append(line)
        
        # Add final item
        if current_item:
            items.append('\n'.join(current_item))
        
        return items
    
    def chunk_paragraphic_document(self, text: str, source_file: str) -> List[TextNode]:
        """
        Chunk paragraphic document by splitting at sentence boundaries and paragraphs.
        
        Args:
            text: Document text
            source_file: Source file name
            
        Returns:
            List of TextNode objects with paragraph chunks
        """
        chunks = []
        paragraphs = self._split_into_paragraphs(text)
        
        for i, paragraph in enumerate(paragraphs):
            # Split long paragraphs into sentences
            sentences = self._split_into_sentences(paragraph)
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                # Check if adding this sentence would exceed token limit
                if current_length + sentence_length > self.config.max_tokens and current_chunk:
                    # Create chunk from current sentences
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_text_node(
                        chunk_text, i, len(paragraphs), source_file, 
                        DocumentType.PARAGRAPHIC, ["paragraph"]
                    ))
                    
                    # Start new chunk with overlap
                    if self.config.sentence_overlap > 0 and current_chunk:
                        overlap_sentences = current_chunk[-self.config.sentence_overlap:]
                        current_chunk = overlap_sentences
                        current_length = sum(len(s.split()) for s in overlap_sentences)
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add remaining sentences as final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_text_node(
                    chunk_text, i, len(paragraphs), source_file,
                    DocumentType.PARAGRAPHIC, ["paragraph"]
                ))
        
        logger.info(f"Created {len(chunks)} paragraphic chunks from {source_file}")
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_text_node(self, text: str, chunk_index: int, total_chunks: int, 
                         source_file: str, doc_type: DocumentType, 
                         structural_tags: List[str]) -> TextNode:
        """Create a TextNode with metadata."""
        return TextNode(
            text=text,
            metadata={
                "document_type": doc_type.value,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "source_file": source_file,
                "structural_tags": structural_tags
            }
        )
    
    def chunk_document(self, document: Document) -> List[TextNode]:
        """
        Main chunking method that detects document type and applies appropriate strategy.
        
        Args:
            document: LlamaIndex Document object
            
        Returns:
            List of TextNode objects
        """
        text = document.text
        source_file = document.metadata.get("file_name", "unknown")
        
        # Detect document type
        doc_type = self.detect_document_type(text)
        
        # Apply type-specific chunking
        if doc_type == DocumentType.QA:
            return self.chunk_qa_document(text, source_file)
        elif doc_type == DocumentType.LIST:
            return self.chunk_list_document(text, source_file)
        elif doc_type == DocumentType.PARAGRAPHIC:
            return self.chunk_paragraphic_document(text, source_file)
        else:
            # Fallback to paragraphic
            return self.chunk_paragraphic_document(text, source_file)
