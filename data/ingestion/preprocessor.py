"""
Text preprocessing module for cleaning and normalizing document content.

This module handles text cleaning, normalization, and preparation
for optimal embedding generation and chunking.
"""

import re
import string
from typing import List, Optional
from dataclasses import dataclass

from llama_index.core import Document

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingStats:
    """Statistics about preprocessing operations."""
    original_length: int
    cleaned_length: int
    removed_chars: int
    removed_lines: int
    processing_time: float


class TextPreprocessor:
    """
    Preprocesses text content for optimal embedding and chunking.
    
    Handles text cleaning, normalization, and preparation using
    simple and stable methods for consistent results.
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.cleaning_patterns = [
            # Remove excessive whitespace
            (r'\s+', ' '),
            # Remove excessive newlines
            (r'\n\s*\n\s*\n+', '\n\n'),
            # Clean up bullet points and lists
            (r'^\s*[-•*]\s*', '• '),
            # Normalize quotes
            (r'["""]', '"'),
            (r"[''']", "'"),
            # Clean up punctuation
            (r'\.{2,}', '.'),
            (r'!{2,}', '!'),
            (r'\?{2,}', '?'),
        ]
        
        logger.info("Text preprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        original_length = len(text)
        
        # Apply cleaning patterns
        cleaned_text = text
        for pattern, replacement in self.cleaning_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.MULTILINE)
        
        # Remove leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        # Remove empty lines at the beginning and end
        lines = cleaned_text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        removed_chars = original_length - len(cleaned_text)
        logger.debug(f"Cleaned text: {original_length} -> {len(cleaned_text)} chars (removed {removed_chars})")
        
        return cleaned_text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent processing.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase for consistency
        normalized = text.lower()
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove excessive punctuation
        normalized = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', normalized)
        
        return normalized.strip()
    
    def extract_sections(self, text: str) -> List[str]:
        """
        Extract logical sections from text based on headers.
        
        Args:
            text: Text content to section
            
        Returns:
            List of text sections
        """
        if not text:
            return []
        
        # Split by common header patterns
        header_patterns = [
            r'^#{1,6}\s+',  # Markdown headers
            r'^[A-Z][A-Z\s]+\n',  # ALL CAPS headers
            r'^\d+\.\s+',  # Numbered sections
            r'^[A-Z][a-z]+[A-Z][a-z]*\s*$',  # TitleCase headers
        ]
        
        combined_pattern = '|'.join(header_patterns)
        sections = re.split(combined_pattern, text, flags=re.MULTILINE)
        
        # Filter out empty sections
        sections = [section.strip() for section in sections if section.strip()]
        
        logger.debug(f"Extracted {len(sections)} sections from text")
        return sections
    
    def preprocess_document(self, document: Document) -> Document:
        """
        Preprocess a single document.
        
        Args:
            document: LlamaIndex Document object
            
        Returns:
            Preprocessed Document object
        """
        if not document.text:
            logger.warning(f"Document {document.doc_id} has no text content")
            return document
        
        original_text = document.text
        cleaned_text = self.clean_text(original_text)
        
        # Create new document with cleaned text
        processed_document = Document(
            text=cleaned_text,
            doc_id=document.doc_id,
            metadata={
                **document.metadata,
                'original_length': len(original_text),
                'cleaned_length': len(cleaned_text),
                'preprocessed': True
            }
        )
        
        logger.info(f"Preprocessed document {document.doc_id}: {len(original_text)} -> {len(cleaned_text)} chars")
        return processed_document
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess multiple documents.
        
        Args:
            documents: List of LlamaIndex Document objects
            
        Returns:
            List of preprocessed Document objects
        """
        processed_documents = []
        
        for document in documents:
            try:
                processed_doc = self.preprocess_document(document)
                processed_documents.append(processed_doc)
            except Exception as e:
                logger.error(f"Error preprocessing document {document.doc_id}: {e}")
                # Keep original document if preprocessing fails
                processed_documents.append(document)
        
        logger.info(f"Preprocessed {len(processed_documents)} documents")
        return processed_documents
    
    def get_preprocessing_stats(self, original_text: str, cleaned_text: str) -> PreprocessingStats:
        """
        Get statistics about preprocessing operations.
        
        Args:
            original_text: Original text content
            cleaned_text: Cleaned text content
            
        Returns:
            PreprocessingStats object
        """
        original_lines = len(original_text.split('\n'))
        cleaned_lines = len(cleaned_text.split('\n'))
        
        return PreprocessingStats(
            original_length=len(original_text),
            cleaned_length=len(cleaned_text),
            removed_chars=len(original_text) - len(cleaned_text),
            removed_lines=original_lines - cleaned_lines,
            processing_time=0.0  # Could be calculated if timing is needed
        )
    
    def validate_text(self, text: str) -> bool:
        """
        Validate that text is suitable for processing.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text is valid, False otherwise
        """
        if not text or not text.strip():
            return False
        
        # Check for minimum content length
        if len(text.strip()) < 10:
            return False
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
        if special_char_ratio > 0.5:
            logger.warning(f"Text has high special character ratio: {special_char_ratio:.2f}")
            return False
        
        return True 