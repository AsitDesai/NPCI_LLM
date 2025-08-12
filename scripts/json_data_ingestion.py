#!/usr/bin/env python3
"""
JSON-specific data ingestion pipeline for processing JSON files with chunk objects.

This script handles the complete JSON document ingestion pipeline:
- JSON file loading and parsing
- Chunk object processing with unique identifiers
- Text concatenation for embedding generation
- Metadata preservation in payload
- Token counting and chunk size validation
"""

import os
import sys
import json
import time
import uuid
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
class ChunkObject:
    """Represents a chunk object from the JSON file."""
    chunk_id: str
    text: str
    payload: Dict[str, Any]
    token_count: int


@dataclass
class ProcessingStats:
    """Statistics about JSON processing."""
    files_processed: int
    total_chunks: int
    valid_chunks: int
    oversized_chunks: int
    processing_time: float
    avg_tokens_per_chunk: float


class JSONDataIngestion:
    """
    JSON-specific data ingestion pipeline.
    
    This class provides a complete JSON processing pipeline
    for handling JSON files with arrays of chunk objects.
    """
    
    def __init__(self, max_tokens_per_chunk: int = 200):
        """
        Initialize the JSON data ingestion pipeline.
        
        Args:
            max_tokens_per_chunk: Maximum tokens allowed per chunk
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.reference_documents_dir = settings.reference_documents_dir
        
        logger.info(f"JSON data ingestion pipeline initialized with max_tokens_per_chunk={max_tokens_per_chunk}")
    
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
    
    def _concatenate_fields(self, chunk_obj: Dict[str, Any]) -> str:
        """
        Concatenate relevant fields into a single string for embedding.
        
        Args:
            chunk_obj: Chunk object from JSON
            
        Returns:
            Concatenated text string
        """
        # Define the order and fields to concatenate
        field_order = [
            'category',
            'scenario', 
            'user_statement',
            'agent_response',
            'system_behavior',
            'agent_guideline'
        ]
        
        concatenated_parts = []
        
        for field in field_order:
            if field in chunk_obj and chunk_obj[field]:
                value = chunk_obj[field]
                if isinstance(value, str):
                    concatenated_parts.append(f"{field}: {value}")
                else:
                    concatenated_parts.append(f"{field}: {str(value)}")
        
        # Join all parts with double newlines for clear separation
        return "\n\n".join(concatenated_parts)
    
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
    
    def load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load and parse a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of chunk objects from the JSON file
        """
        try:
            logger.info(f"Loading JSON file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError(f"JSON file {file_path} does not contain an array")
            
            logger.info(f"Loaded {len(data)} chunk objects from {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return []
    
    def process_chunk_object(self, chunk_obj: Dict[str, Any], file_name: str, chunk_index: int) -> Optional[ChunkObject]:
        """
        Process a single chunk object from the JSON file.
        
        Args:
            chunk_obj: Chunk object from JSON
            file_name: Name of the source file
            chunk_index: Index of the chunk in the file
            
        Returns:
            Processed ChunkObject or None if invalid
        """
        try:
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(file_name, chunk_index)
            
            # Concatenate relevant fields into text
            text = self._concatenate_fields(chunk_obj)
            
            # Estimate token count
            token_count = self._estimate_token_count(text)
            
            # Check if chunk is within token limit
            if token_count > self.max_tokens_per_chunk:
                logger.warning(f"Chunk {chunk_id} exceeds token limit: {token_count} > {self.max_tokens_per_chunk}")
                return None
            
            # Create payload with all original metadata
            payload = {
                **chunk_obj,  # All original fields
                "chunk_id": chunk_id,
                "source_file": file_name,
                "chunk_index": chunk_index,
                "token_count": token_count,
                "processing_timestamp": time.time()
            }
            
            chunk_object = ChunkObject(
                chunk_id=chunk_id,
                text=text,
                payload=payload,
                token_count=token_count
            )
            
            logger.debug(f"Processed chunk {chunk_id}: {token_count} tokens")
            return chunk_object
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index} from {file_name}: {e}")
            return None
    
    def process_json_files(self, documents_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all JSON files in the specified directory.
        
        Args:
            documents_dir: Directory containing JSON files
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        try:
            documents_dir = documents_dir or self.reference_documents_dir
            documents_path = Path(documents_dir)
            
            if not documents_path.exists():
                logger.error(f"Documents directory does not exist: {documents_dir}")
                return {
                    "success": False,
                    "error": f"Documents directory does not exist: {documents_dir}",
                    "statistics": {}
                }
            
            logger.info(f"Processing JSON files from: {documents_dir}")
            
            # Find all JSON files
            json_files = list(documents_path.glob("*.json"))
            
            if not json_files:
                logger.warning(f"No JSON files found in {documents_dir}")
                return {
                    "success": False,
                    "error": f"No JSON files found in {documents_dir}",
                    "statistics": {}
                }
            
            logger.info(f"Found {len(json_files)} JSON files to process")
            
            all_chunks = []
            files_processed = 0
            total_chunks = 0
            valid_chunks = 0
            oversized_chunks = 0
            
            # Process each JSON file
            for json_file in json_files:
                try:
                    # Load chunk objects from JSON file
                    chunk_objects = self.load_json_file(json_file)
                    
                    if not chunk_objects:
                        logger.warning(f"No valid chunk objects found in {json_file}")
                        continue
                    
                    total_chunks += len(chunk_objects)
                    file_chunks = []
                    
                    # Process each chunk object
                    for i, chunk_obj in enumerate(chunk_objects):
                        processed_chunk = self.process_chunk_object(chunk_obj, json_file.name, i)
                        
                        if processed_chunk:
                            file_chunks.append(processed_chunk)
                            valid_chunks += 1
                        else:
                            oversized_chunks += 1
                    
                    all_chunks.extend(file_chunks)
                    files_processed += 1
                    
                    logger.info(f"Processed {json_file.name}: {len(file_chunks)} valid chunks out of {len(chunk_objects)} total")
                    
                except Exception as e:
                    logger.error(f"Error processing file {json_file}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            
            # Calculate statistics
            avg_tokens_per_chunk = sum(chunk.token_count for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
            
            statistics = ProcessingStats(
                files_processed=files_processed,
                total_chunks=total_chunks,
                valid_chunks=valid_chunks,
                oversized_chunks=oversized_chunks,
                processing_time=processing_time,
                avg_tokens_per_chunk=avg_tokens_per_chunk
            )
            
            logger.info(f"JSON processing completed in {processing_time:.3f}s")
            logger.info(f"Statistics: {statistics}")
            
            return {
                "success": True,
                "chunks": all_chunks,
                "statistics": statistics
            }
            
        except Exception as e:
            logger.error(f"JSON processing pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def get_processing_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed processing statistics.
        
        Args:
            results: Results from process_json_files
            
        Returns:
            Dictionary with detailed statistics
        """
        if not results.get("success", False):
            return {"error": "Processing failed"}
        
        stats = results["statistics"]
        chunks = results.get("chunks", [])
        
        # Token distribution
        token_counts = [chunk.token_count for chunk in chunks]
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        
        # Category distribution
        category_counts = {}
        type_counts = {}
        
        for chunk in chunks:
            category = chunk.payload.get("category", "unknown")
            chunk_type = chunk.payload.get("type", "unknown")
            
            category_counts[category] = category_counts.get(category, 0) + 1
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        detailed_stats = {
            "files_processed": stats.files_processed,
            "total_chunks": stats.total_chunks,
            "valid_chunks": stats.valid_chunks,
            "oversized_chunks": stats.oversized_chunks,
            "processing_time": stats.processing_time,
            "avg_tokens_per_chunk": stats.avg_tokens_per_chunk,
            "token_distribution": {
                "minimum": min_tokens,
                "maximum": max_tokens,
                "total_chunks": len(token_counts)
            },
            "category_distribution": category_counts,
            "type_distribution": type_counts,
            "processing_efficiency": {
                "chunks_per_second": stats.valid_chunks / stats.processing_time if stats.processing_time > 0 else 0,
                "files_per_second": stats.files_processed / stats.processing_time if stats.processing_time > 0 else 0
            }
        }
        
        return detailed_stats


def main():
    """Run the JSON data ingestion pipeline."""
    print("🚀 JSON DATA INGESTION PIPELINE")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = JSONDataIngestion(max_tokens_per_chunk=200)
        
        # Process JSON files
        results = pipeline.process_json_files()
        
        if results["success"]:
            print("\n✅ JSON PROCESSING COMPLETED SUCCESSFULLY!")
            
            # Display statistics
            stats = pipeline.get_processing_stats(results)
            
            print(f"\n📊 PROCESSING STATISTICS:")
            print(f"   Files processed: {stats['files_processed']}")
            print(f"   Total chunks: {stats['total_chunks']}")
            print(f"   Valid chunks: {stats['valid_chunks']}")
            print(f"   Oversized chunks: {stats['oversized_chunks']}")
            print(f"   Processing time: {stats['processing_time']:.3f}s")
            print(f"   Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
            
            # Token distribution
            token_dist = stats.get('token_distribution', {})
            print(f"\n📏 TOKEN DISTRIBUTION:")
            print(f"   Minimum tokens: {token_dist.get('minimum', 0)}")
            print(f"   Maximum tokens: {token_dist.get('maximum', 0)}")
            print(f"   Total chunks: {token_dist.get('total_chunks', 0)}")
            
            # Category distribution
            if stats.get('category_distribution'):
                print(f"\n📂 CATEGORY DISTRIBUTION:")
                for category, count in stats['category_distribution'].items():
                    print(f"   {category}: {count} chunks")
            
            # Type distribution
            if stats.get('type_distribution'):
                print(f"\n🏷️  TYPE DISTRIBUTION:")
                for chunk_type, count in stats['type_distribution'].items():
                    print(f"   {chunk_type}: {count} chunks")
            
            # Performance metrics
            perf = stats.get('processing_efficiency', {})
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"   Chunks per second: {perf.get('chunks_per_second', 0):.1f}")
            print(f"   Files per second: {perf.get('files_per_second', 0):.1f}")
            
            print(f"\n🚀 Ready for embedding generation!")
            return True
            
        else:
            print(f"\n❌ JSON PROCESSING FAILED: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"JSON data ingestion pipeline failed: {e}")
        print(f"\n❌ PIPELINE FAILED: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

