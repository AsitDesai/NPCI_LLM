#!/usr/bin/env python3
"""
Simple Data Ingestion Pipeline

This script provides a simple document ingestion pipeline that:
- Reads documents normally without complex parsing
- Chunks text into manageable pieces
- Preserves original content without category/scenario segregation
- Works with any text-based format
"""

import os
import sys
import time
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimpleChunk:
    """Represents a simple text chunk."""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    token_count: int


@dataclass
class ProcessingStats:
    """Statistics about document processing."""
    files_processed: int
    total_chunks: int
    valid_chunks: int
    processing_time: float
    avg_tokens_per_chunk: float


class SimpleDataIngestion:
    """
    Simple data ingestion pipeline that reads documents normally.
    
    This class provides a straightforward document processing pipeline
    that chunks text without complex parsing or categorization.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the simple data ingestion pipeline.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reference_documents_dir = settings.reference_documents_dir
        
        logger.info(f"Simple data ingestion pipeline initialized")
        logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def _generate_chunk_id(self, file_name: str, chunk_index: int) -> str:
        """
        Generate a unique identifier for a chunk.
        
        Args:
            file_name: Name of the source file
            chunk_index: Index of the chunk in the file
            
        Returns:
            Unique chunk identifier
        """
        # Create a unique ID using file name, index, and UUID
        base_id = f"{Path(file_name).stem}_{chunk_index}"
        unique_suffix = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        return f"{base_id}_{unique_suffix}"
    
    def _chunk_text(self, text: str, file_name: str) -> List[SimpleChunk]:
        """
        Chunk text into smaller pieces with overlap.
        
        Args:
            text: Text to chunk
            file_name: Name of the source file
            
        Returns:
            List of SimpleChunk objects
        """
        chunks = []
        
        if len(text) <= self.chunk_size:
            # Text is small enough to be a single chunk
            chunk_id = self._generate_chunk_id(file_name, 0)
            token_count = self._estimate_token_count(text)
            
            chunk = SimpleChunk(
                chunk_id=chunk_id,
                text=text,
                source_file=file_name,
                chunk_index=0,
                token_count=token_count
            )
            chunks.append(chunk)
        else:
            # Split text into overlapping chunks
            start = 0
            chunk_index = 0
            
            while start < len(text):
                end = start + self.chunk_size
                
                # If this is not the last chunk, try to break at a sentence boundary
                if end < len(text):
                    # Look for sentence endings within the last 100 characters
                    search_start = max(start + self.chunk_size - 100, start)
                    search_text = text[search_start:end]
                    
                    # Find the last sentence ending
                    sentence_endings = ['.', '!', '?', '\n\n']
                    last_ending = -1
                    
                    for ending in sentence_endings:
                        pos = search_text.rfind(ending)
                        if pos > last_ending:
                            last_ending = pos
                    
                    if last_ending > 0:
                        end = search_start + last_ending + 1
                
                # Extract chunk text
                chunk_text = text[start:end].strip()
                
                if chunk_text:  # Only create chunk if there's content
                    chunk_id = self._generate_chunk_id(file_name, chunk_index)
                    token_count = self._estimate_token_count(chunk_text)
                    
                    chunk = SimpleChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source_file=file_name,
                        chunk_index=chunk_index,
                        token_count=token_count
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Move to next chunk with overlap
                start = end - self.chunk_overlap
                if start >= len(text):
                    break
        
        return chunks
    
    def process_text_file(self, file_path: Path) -> List[SimpleChunk]:
        """
        Process a text file and create chunks.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of SimpleChunk objects
        """
        try:
            logger.info(f"Processing text file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean up the content
            content = content.strip()
            
            if not content:
                logger.warning(f"File {file_path} is empty")
                return []
            
            # Create chunks
            chunks = self._chunk_text(content, file_path.name)
            
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all supported files in the reference documents directory.
        
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        documents_dir = Path(self.reference_documents_dir)
        if not documents_dir.exists():
            logger.error(f"Documents directory does not exist: {documents_dir}")
            return {
                "success": False,
                "error": f"Documents directory does not exist: {documents_dir}",
                "chunks": [],
                "statistics": ProcessingStats(0, 0, 0, 0, 0)
            }
        
        # Find all text files
        text_files = list(documents_dir.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No text files found in {documents_dir}")
            return {
                "success": False,
                "error": "No text files found",
                "chunks": [],
                "statistics": ProcessingStats(0, 0, 0, 0, 0)
            }
        
        logger.info(f"Found {len(text_files)} text files to process")
        
        all_chunks = []
        files_processed = 0
        
        for file_path in text_files:
            try:
                chunks = self.process_text_file(file_path)
                all_chunks.extend(chunks)
                files_processed += 1
                
                logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        processing_time = time.time() - start_time
        avg_tokens_per_chunk = sum(chunk.token_count for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        
        statistics = ProcessingStats(
            files_processed=files_processed,
            total_chunks=len(all_chunks),
            valid_chunks=len(all_chunks),
            processing_time=processing_time,
            avg_tokens_per_chunk=avg_tokens_per_chunk
        )
        
        logger.info(f"Processing completed: {len(all_chunks)} chunks from {files_processed} files")
        logger.info(f"Processing time: {processing_time:.3f}s")
        logger.info(f"Average tokens per chunk: {avg_tokens_per_chunk:.1f}")
        
        return {
            "success": True,
            "chunks": all_chunks,
            "statistics": statistics
        }


def main():
    """Test the simple data ingestion pipeline."""
    print("🧪 SIMPLE DATA INGESTION TEST")
    print("="*50)
    
    try:
        # Initialize pipeline with increased overlap
        pipeline = SimpleDataIngestion(chunk_size=500, chunk_overlap=200)
        
        # Process all files
        print("\n📄 PROCESSING TEXT FILES:")
        results = pipeline.process_all_files()
        
        if results["success"]:
            stats = results["statistics"]
            chunks = results["chunks"]
            
            print(f"   ✅ Processing successful")
            print(f"   Files processed: {stats.files_processed}")
            print(f"   Total chunks: {stats.total_chunks}")
            print(f"   Valid chunks: {stats.valid_chunks}")
            print(f"   Processing time: {stats.processing_time:.3f}s")
            print(f"   Average tokens per chunk: {stats.avg_tokens_per_chunk:.1f}")
            
            # Show sample chunks
            if chunks:
                print(f"\n📋 SAMPLE CHUNKS:")
                for i, chunk in enumerate(chunks[:3]):
                    print(f"   Chunk {i+1}:")
                    print(f"     ID: {chunk.chunk_id}")
                    print(f"     Source: {chunk.source_file}")
                    print(f"     Tokens: {chunk.token_count}")
                    print(f"     Text preview: {chunk.text[:100]}...")
                    print()
        else:
            print(f"   ❌ Processing failed: {results.get('error')}")
            return False
        
        print(f"\n✅ Simple data ingestion test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Simple data ingestion test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    if not success:
        sys.exit(1)
