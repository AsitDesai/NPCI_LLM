"""
Document ingestion package for the RAG System.

This package handles document loading, preprocessing, and chunking
operations using LlamaIndex for optimal text processing.
"""

from .data_collector import DocumentCollector
from .preprocessor import TextPreprocessor
from .chunking import DocumentChunker

__all__ = ["DocumentCollector", "TextPreprocessor", "DocumentChunker"] 