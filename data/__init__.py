"""
Data management package for the RAG System.

This package handles all data-related operations including:
- Document ingestion and collection
- Text preprocessing and cleaning
- Chunking and segmentation
- Data storage and retrieval
"""

from .ingestion.data_collector import DocumentCollector
from .ingestion.preprocessor import TextPreprocessor
from .ingestion.chunking import DocumentChunker

__all__ = ["DocumentCollector", "TextPreprocessor", "DocumentChunker"] 