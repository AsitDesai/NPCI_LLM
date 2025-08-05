"""
Document collector for loading files from reference_documents directory.

This module handles loading various file formats (TXT, PDF, DOCX) using
LlamaIndex's document loaders for consistent processing.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentInfo:
    """Information about a loaded document."""
    file_path: str
    file_name: str
    file_type: str
    content_length: int
    metadata: Dict[str, Any]


class DocumentCollector:
    """
    Collects and loads documents from the reference_documents directory.
    
    Uses LlamaIndex document loaders for consistent processing across
    different file formats (TXT, PDF, DOCX).
    """
    
    def __init__(self, documents_dir: Optional[str] = None):
        """
        Initialize the document collector.
        
        Args:
            documents_dir: Directory containing documents to load
        """
        self.documents_dir = Path(documents_dir or settings.reference_documents_dir)
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.doc'}
        self.loaded_documents: List[Document] = []
        self.document_info: List[DocumentInfo] = []
        
        logger.info(f"Document collector initialized for directory: {self.documents_dir}")
    
    def scan_documents(self) -> List[Path]:
        """
        Scan the documents directory for supported file types.
        
        Returns:
            List of file paths to supported documents
        """
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory does not exist: {self.documents_dir}")
            return []
        
        document_files = []
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                document_files.append(file_path)
        
        logger.info(f"Found {len(document_files)} supported documents")
        return document_files
    
    def load_documents(self) -> List[Document]:
        """
        Load all documents from the reference_documents directory.
        
        Returns:
            List of LlamaIndex Document objects
        """
        document_files = self.scan_documents()
        
        if not document_files:
            logger.warning("No documents found to load")
            return []
        
        # Use LlamaIndex's SimpleDirectoryReader for consistent loading
        try:
            # Convert to absolute path
            abs_documents_dir = self.documents_dir.resolve()
            logger.info(f"Loading documents from: {abs_documents_dir}")
            logger.info(f"Required extensions: {[ext[1:] for ext in self.supported_extensions]}")
            
            # Try without required_exts first
            reader = SimpleDirectoryReader(
                input_dir=str(abs_documents_dir),
                recursive=False,  # Don't search subdirectories
                filename_as_id=True,
                exclude_hidden=True  # Exclude hidden files like .gitkeep
            )
            
            documents = reader.load_data()
            
            # Filter documents by supported extensions
            filtered_documents = []
            for doc in documents:
                file_path = doc.metadata.get('file_path', doc.doc_id)
                file_ext = Path(file_path).suffix.lower()
                if file_ext in self.supported_extensions:
                    filtered_documents.append(doc)
            
            # Process and store document information
            for doc in filtered_documents:
                self._process_document_info(doc)
            
            self.loaded_documents = filtered_documents
            logger.info(f"Successfully loaded {len(filtered_documents)} documents")
            
            return filtered_documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def _process_document_info(self, document: Document) -> None:
        """
        Process and store information about a loaded document.
        
        Args:
            document: LlamaIndex Document object
        """
        try:
            # Extract file path from metadata or document ID
            file_path = document.metadata.get('file_path', document.doc_id)
            file_name = Path(file_path).name
            file_type = Path(file_path).suffix.lower()
            
            doc_info = DocumentInfo(
                file_path=file_path,
                file_name=file_name,
                file_type=file_type,
                content_length=len(document.text),
                metadata=document.metadata
            )
            
            self.document_info.append(doc_info)
            logger.debug(f"Processed document: {file_name} ({doc_info.content_length} chars)")
            
        except Exception as e:
            logger.error(f"Error processing document info: {e}")
    
    def get_document_info(self) -> List[DocumentInfo]:
        """
        Get information about all loaded documents.
        
        Returns:
            List of DocumentInfo objects
        """
        return self.document_info
    
    def get_document_by_name(self, file_name: str) -> Optional[Document]:
        """
        Get a specific document by file name.
        
        Args:
            file_name: Name of the file to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        for doc in self.loaded_documents:
            if doc.metadata.get('file_name') == file_name or doc.doc_id == file_name:
                return doc
        return None
    
    def get_documents_by_type(self, file_type: str) -> List[Document]:
        """
        Get all documents of a specific file type.
        
        Args:
            file_type: File extension (e.g., '.txt', '.pdf')
            
        Returns:
            List of Document objects of the specified type
        """
        file_type = file_type.lower()
        return [doc for doc in self.loaded_documents 
                if doc.metadata.get('file_type', '').lower() == file_type]
    
    def get_total_content_length(self) -> int:
        """
        Get the total content length of all loaded documents.
        
        Returns:
            Total number of characters across all documents
        """
        return sum(len(doc.text) for doc in self.loaded_documents)
    
    def clear_documents(self) -> None:
        """Clear all loaded documents and document info."""
        self.loaded_documents.clear()
        self.document_info.clear()
        logger.info("Cleared all loaded documents") 