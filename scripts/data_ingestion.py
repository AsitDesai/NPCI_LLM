#!/usr/bin/env python3
"""
Document processing pipeline script using LlamaIndex.

This script handles the complete document ingestion pipeline:
- Document loading using LlamaIndex
- Text preprocessing and cleaning
- Document chunking with LlamaIndex text splitters
- Metadata extraction and preservation
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from data.ingestion.enhanced_chunker import EnhancedChunker
from llama_index.core.schema import TextNode

from data.ingestion.data_collector import DocumentCollector
from data.ingestion.preprocessor import TextPreprocessor
from data.ingestion.chunking import DocumentChunker
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


class LlamaIndexDataIngestion:
    """
    LlamaIndex-based document ingestion pipeline.
    
    This class provides a complete document processing pipeline
    using LlamaIndex components for loading, preprocessing, and chunking.
    """
    
    def __init__(self):
        """Initialize the LlamaIndex data ingestion pipeline."""
        self.data_collector = DocumentCollector()
        self.preprocessor = TextPreprocessor()
        self.enhanced_chunker = EnhancedChunker()
        
        logger.info("LlamaIndex data ingestion pipeline initialized with enhanced chunker")
    
    def load_documents_llama_index(self, documents_dir: Optional[str] = None) -> List[Document]:
        """
        Load documents using LlamaIndex SimpleDirectoryReader.
        
        Args:
            documents_dir: Directory containing documents
            
        Returns:
            List of LlamaIndex Document objects
        """
        documents_dir = documents_dir or settings.reference_documents_dir
        
        try:
            logger.info(f"Loading documents from: {documents_dir}")
            
            # Use LlamaIndex SimpleDirectoryReader
            reader = SimpleDirectoryReader(
                input_dir=documents_dir,
                recursive=False,
                exclude_hidden=True
            )
            
            # Load documents
            documents = reader.load_data()
            
            logger.info(f"Loaded {len(documents)} documents using LlamaIndex")
            
            # Log document details
            for i, doc in enumerate(documents):
                logger.debug(f"Document {i+1}: {doc.metadata.get('file_name', 'Unknown')} "
                           f"({len(doc.text)} characters)")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents with LlamaIndex: {e}")
            return []
    
    def preprocess_documents_llama_index(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess documents using LlamaIndex and custom preprocessing.
        
        Args:
            documents: List of LlamaIndex Document objects
            
        Returns:
            List of preprocessed Document objects
        """
        try:
            logger.info(f"Preprocessing {len(documents)} documents")
            
            preprocessed_documents = []
            
            for i, doc in enumerate(documents):
                # Preprocess text using our custom preprocessor
                preprocessed_text = self.preprocessor.clean_text(doc.text)
                
                # Create new document with preprocessed text
                preprocessed_doc = Document(
                    text=preprocessed_text,
                    metadata=doc.metadata
                )
                
                preprocessed_documents.append(preprocessed_doc)
                
                logger.debug(f"Preprocessed document {i+1}: "
                           f"{doc.metadata.get('file_name', 'Unknown')} "
                           f"({len(doc.text)} -> {len(preprocessed_text)} characters)")
            
            logger.info(f"Preprocessed {len(preprocessed_documents)} documents")
            return preprocessed_documents
            
        except Exception as e:
            logger.error(f"Error preprocessing documents: {e}")
            return documents
    
    def chunk_documents_llama_index(self, documents: List[Document]) -> List[TextNode]:
        """
        Chunk documents using LlamaIndex text splitters.
        
        Args:
            documents: List of LlamaIndex Document objects
            
        Returns:
            List of LlamaIndex TextNode objects
        """
        try:
            logger.info(f"Chunking {len(documents)} documents using LlamaIndex")
            
            all_nodes = []
            
            for i, doc in enumerate(documents):
                # Use enhanced chunker for each document
                nodes = self.enhanced_chunker.chunk_document(doc)
                
                # Add additional metadata
                for j, node in enumerate(nodes):
                    node.metadata.update({
                        "doc_id": doc.metadata.get("file_name", f"doc_{i}"),
                        "chunk_index": j,
                        "total_chunks": len(nodes),
                        "source_file": doc.metadata.get("file_name", "Unknown"),
                        "chunking_method": "enhanced_format_detection"
                    })
                
                all_nodes.extend(nodes)
                
                logger.debug(f"Chunked document {i+1}: {doc.metadata.get('file_name', 'Unknown')} "
                           f"into {len(nodes)} chunks using enhanced chunker")
            
            logger.info(f"Created {len(all_nodes)} chunks from {len(documents)} documents")
            return all_nodes
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            return []
    
    def process_documents_pipeline(self, documents_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete document processing pipeline using LlamaIndex.
        
        Args:
            documents_dir: Directory containing documents
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        try:
            logger.info("Starting LlamaIndex document processing pipeline")
            
            # Step 1: Load documents
            load_start = time.time()
            documents = self.load_documents_llama_index(documents_dir)
            load_time = time.time() - load_start
            
            if not documents:
                logger.error("No documents loaded, stopping pipeline")
                return {
                    "success": False,
                    "error": "No documents loaded",
                    "statistics": {}
                }
            
            # Step 2: Preprocess documents
            preprocess_start = time.time()
            preprocessed_docs = self.preprocess_documents_llama_index(documents)
            preprocess_time = time.time() - preprocess_start
            
            # Step 3: Chunk documents
            chunk_start = time.time()
            nodes = self.chunk_documents_llama_index(preprocessed_docs)
            chunk_time = time.time() - chunk_start
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            total_chars = sum(len(doc.text) for doc in documents)
            total_chars_processed = sum(len(doc.text) for doc in preprocessed_docs)
            
            statistics = {
                "documents_loaded": len(documents),
                "documents_preprocessed": len(preprocessed_docs),
                "chunks_created": len(nodes),
                "total_characters": total_chars,
                "total_characters_processed": total_chars_processed,
                "load_time": load_time,
                "preprocess_time": preprocess_time,
                "chunk_time": chunk_time,
                "total_time": total_time,
                "avg_chunks_per_doc": len(nodes) / len(documents) if documents else 0,
                "avg_chars_per_chunk": sum(len(node.text) for node in nodes) / len(nodes) if nodes else 0
            }
            
            logger.info(f"Document processing pipeline completed in {total_time:.3f}s")
            logger.info(f"Statistics: {statistics}")
            
            return {
                "success": True,
                "documents": documents,
                "preprocessed_docs": preprocessed_docs,
                "nodes": nodes,
                "statistics": statistics
            }
            
        except Exception as e:
            logger.error(f"Document processing pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def get_processing_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed processing statistics.
        
        Args:
            results: Results from process_documents_pipeline
            
        Returns:
            Dictionary with detailed statistics
        """
        if not results.get("success", False):
            return {"error": "Processing failed"}
        
        stats = results["statistics"]
        
        # Additional statistics
        nodes = results.get("nodes", [])
        documents = results.get("documents", [])
        
        # File type distribution
        file_types = {}
        for doc in documents:
            file_name = doc.metadata.get("file_name", "")
            if file_name:
                ext = Path(file_name).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        # Chunk size distribution
        chunk_sizes = [len(node.text) for node in nodes]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
        max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
        
        detailed_stats = {
            **stats,
            "file_types": file_types,
            "chunk_size_stats": {
                "average": avg_chunk_size,
                "minimum": min_chunk_size,
                "maximum": max_chunk_size,
                "total_chunks": len(chunk_sizes)
            },
            "processing_efficiency": {
                "chars_per_second": stats["total_characters_processed"] / stats["total_time"] if stats["total_time"] > 0 else 0,
                "chunks_per_second": stats["chunks_created"] / stats["total_time"] if stats["total_time"] > 0 else 0
            }
        }
        
        return detailed_stats


def main():
    """Run the LlamaIndex data ingestion pipeline."""
    print("🚀 LLAMAINDEX DATA INGESTION PIPELINE")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = LlamaIndexDataIngestion()
        
        # Process documents
        results = pipeline.process_documents_pipeline()
        
        if results["success"]:
            print("\n✅ DOCUMENT PROCESSING COMPLETED SUCCESSFULLY!")
            
            # Display statistics
            stats = pipeline.get_processing_stats(results)
            
            print(f"\n📊 PROCESSING STATISTICS:")
            print(f"   Documents loaded: {stats['documents_loaded']}")
            print(f"   Documents preprocessed: {stats['documents_preprocessed']}")
            print(f"   Chunks created: {stats['chunks_created']}")
            print(f"   Total characters: {stats['total_characters']:,}")
            print(f"   Processing time: {stats['total_time']:.3f}s")
            print(f"   Average chunks per document: {stats['avg_chunks_per_doc']:.1f}")
            print(f"   Average characters per chunk: {stats['avg_chars_per_chunk']:.0f}")
            
            # File type distribution
            if stats.get('file_types'):
                print(f"\n📁 FILE TYPE DISTRIBUTION:")
                for ext, count in stats['file_types'].items():
                    print(f"   {ext}: {count} files")
            
            # Performance metrics
            perf = stats.get('processing_efficiency', {})
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"   Characters per second: {perf.get('chars_per_second', 0):.0f}")
            print(f"   Chunks per second: {perf.get('chunks_per_second', 0):.1f}")
            
            print(f"\n🚀 Ready for embedding generation!")
            return True
            
        else:
            print(f"\n❌ DOCUMENT PROCESSING FAILED: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        print(f"\n❌ PIPELINE FAILED: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 