#!/usr/bin/env python3
"""
Enhanced data ingestion pipeline supporting both JSON and TXT file formats.

This script handles the complete document ingestion pipeline:
- JSON file loading and parsing (backward compatibility)
- TXT file parsing with structured Q&A extraction
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
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkObject:
    """Represents a chunk object from any supported file format."""
    chunk_id: str
    text: str
    payload: Dict[str, Any]
    token_count: int


@dataclass
class ProcessingStats:
    """Statistics about file processing."""
    files_processed: int
    total_chunks: int
    valid_chunks: int
    oversized_chunks: int
    processing_time: float
    avg_tokens_per_chunk: float
    json_files: int
    txt_files: int


class TXTParser:
    """Parser for TXT files with structured Q&A format."""
    
    def __init__(self):
        self.category_separator = "-" * 60
        self.qa_pattern = re.compile(r'Q:\s*"([^"]+)"\s*\nA:\s*(.+?)(?=\n\n|\nQ:|$)', re.DOTALL)
        self.scenario_pattern = re.compile(r'Scenario:\s*(.+?)\s*\nQ:\s*"([^"]+)"\s*\nA:\s*(.+?)(?=\n\n|\nScenario:|$)', re.DOTALL)
        self.faq_pattern = re.compile(r'Q:\s*"([^"]+)"\s*\nA:\s*(.+?)(?=\n\n|\nQ:|$)', re.DOTALL)
        self.guideline_pattern = re.compile(r'AGENT DESIGN GUIDELINES:\s*\n(.+?)(?=\n\n|\nCATEGORY:|$)', re.DOTALL)
        self.system_behavior_pattern = re.compile(r'System Behavior:\s*\n(.+?)(?=\n\n|\nScenario:|$)', re.DOTALL)
    
    def parse_txt_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a TXT file and extract structured data.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            List of chunk objects extracted from the TXT file
        """
        try:
            logger.info(f"Parsing TXT file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content by category separators
            categories = content.split(self.category_separator)
            chunks = []
            
            for category_content in categories:
                if not category_content.strip():
                    continue
                
                # Extract category name
                category_match = re.search(r'CATEGORY:\s*(.+?)\s*\n', category_content)
                if not category_match:
                    continue
                
                category_name = category_match.group(1).strip()
                
                # Extract system behavior
                system_behavior = ""
                system_match = self.system_behavior_pattern.search(category_content)
                if system_match:
                    system_behavior = system_match.group(1).strip()
                
                # Extract scenarios
                scenarios = self._extract_scenarios(category_content, category_name)
                chunks.extend(scenarios)
                
                # Extract FAQs
                faqs = self._extract_faqs(category_content, category_name)
                chunks.extend(faqs)
                
                # Extract guidelines
                guidelines = self._extract_guidelines(category_content, category_name)
                chunks.extend(guidelines)
            
            logger.info(f"Extracted {len(chunks)} chunks from TXT file: {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error parsing TXT file {file_path}: {e}")
            return []
    
    def _extract_scenarios(self, content: str, category: str) -> List[Dict[str, Any]]:
        """Extract scenario Q&A pairs from content."""
        scenarios = []
        matches = self.scenario_pattern.finditer(content)
        
        for match in matches:
            scenario_name = match.group(1).strip()
            question = match.group(2).strip()
            answer = match.group(3).strip()
            
            chunk_obj = {
                "category": category,
                "scenario": scenario_name,
                "user_statement": question,
                "agent_response": answer,
                "type": "scenario"
            }
            scenarios.append(chunk_obj)
        
        return scenarios
    
    def _extract_faqs(self, content: str, category: str) -> List[Dict[str, Any]]:
        """Extract FAQ Q&A pairs from content."""
        faqs = []
        
        # Find the FAQ section
        faq_section_match = re.search(r'FREQUENTLY ASKED QUESTIONS\s*\n(.+?)(?=\n\n|\nAGENT DESIGN GUIDELINES:|$)', content, re.DOTALL)
        if not faq_section_match:
            return faqs
        
        faq_section = faq_section_match.group(1)
        matches = self.faq_pattern.finditer(faq_section)
        
        for match in matches:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            
            chunk_obj = {
                "category": category,
                "scenario": "FAQ",
                "user_statement": question,
                "agent_response": answer,
                "type": "faq"
            }
            faqs.append(chunk_obj)
        
        return faqs
    
    def _extract_guidelines(self, content: str, category: str) -> List[Dict[str, Any]]:
        """Extract agent guidelines from content."""
        guidelines = []
        matches = self.guideline_pattern.finditer(content)
        
        for match in matches:
            guideline_text = match.group(1).strip()
            
            chunk_obj = {
                "category": category,
                "agent_guideline": guideline_text,
                "type": "guideline"
            }
            guidelines.append(chunk_obj)
        
        return guidelines


class EnhancedDataIngestion:
    """
    Enhanced data ingestion pipeline supporting both JSON and TXT formats.
    
    This class provides a complete processing pipeline for handling
    both JSON files with arrays of chunk objects and TXT files with
    structured Q&A format.
    """
    
    def __init__(self, max_tokens_per_chunk: int = 200):
        """
        Initialize the enhanced data ingestion pipeline.
        
        Args:
            max_tokens_per_chunk: Maximum tokens allowed per chunk
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.reference_documents_dir = settings.reference_documents_dir
        self.txt_parser = TXTParser()
        
        logger.info(f"Enhanced data ingestion pipeline initialized with max_tokens_per_chunk={max_tokens_per_chunk}")
    
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
            chunk_obj: Chunk object from any format
            
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
        Load and parse a JSON file (backward compatibility).
        
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
    
    def load_txt_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load and parse a TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            List of chunk objects extracted from the TXT file
        """
        return self.txt_parser.parse_txt_file(file_path)
    
    def process_chunk_object(self, chunk_obj: Dict[str, Any], file_name: str, chunk_index: int) -> Optional[ChunkObject]:
        """
        Process a single chunk object from any supported file format.
        
        Args:
            chunk_obj: Chunk object from any format
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
            
            return chunk_object
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index} from {file_name}: {e}")
            return None
    
    def process_json_files(self) -> Dict[str, Any]:
        """
        Process all JSON files in the reference documents directory.
        
        Returns:
            Dictionary with processing results and statistics
        """
        return self.process_files(file_extensions=['.json'])
    
    def process_txt_files(self) -> Dict[str, Any]:
        """
        Process all TXT files in the reference documents directory.
        
        Returns:
            Dictionary with processing results and statistics
        """
        return self.process_files(file_extensions=['.txt'])
    
    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all supported files (JSON and TXT) in the reference documents directory.
        
        Returns:
            Dictionary with processing results and statistics
        """
        return self.process_files(file_extensions=['.json', '.txt'])
    
    def process_files(self, file_extensions: List[str]) -> Dict[str, Any]:
        """
        Process files with specified extensions.
        
        Args:
            file_extensions: List of file extensions to process (e.g., ['.json', '.txt'])
            
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
                "statistics": ProcessingStats(0, 0, 0, 0, 0, 0, 0, 0)
            }
        
        # Find files with specified extensions
        files_to_process = []
        for ext in file_extensions:
            files_to_process.extend(documents_dir.glob(f"*{ext}"))
        
        if not files_to_process:
            logger.warning(f"No files found with extensions {file_extensions} in {documents_dir}")
            return {
                "success": False,
                "error": f"No files found with extensions {file_extensions}",
                "chunks": [],
                "statistics": ProcessingStats(0, 0, 0, 0, 0, 0, 0, 0)
            }
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        all_chunks = []
        json_files = 0
        txt_files = 0
        total_chunks = 0
        valid_chunks = 0
        oversized_chunks = 0
        
        for file_path in files_to_process:
            try:
                file_extension = file_path.suffix.lower()
                
                if file_extension == '.json':
                    chunk_objects = self.load_json_file(file_path)
                    json_files += 1
                elif file_extension == '.txt':
                    chunk_objects = self.load_txt_file(file_path)
                    txt_files += 1
                else:
                    logger.warning(f"Unsupported file extension: {file_extension}")
                    continue
                
                total_chunks += len(chunk_objects)
                
                # Process each chunk object
                for i, chunk_obj in enumerate(chunk_objects):
                    processed_chunk = self.process_chunk_object(chunk_obj, file_path.name, i)
                    
                    if processed_chunk:
                        all_chunks.append(processed_chunk)
                        valid_chunks += 1
                    else:
                        oversized_chunks += 1
                
                logger.info(f"Processed {file_path.name}: {len(chunk_objects)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        processing_time = time.time() - start_time
        avg_tokens_per_chunk = sum(chunk.token_count for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        
        statistics = ProcessingStats(
            files_processed=len(files_to_process),
            total_chunks=total_chunks,
            valid_chunks=valid_chunks,
            oversized_chunks=oversized_chunks,
            processing_time=processing_time,
            avg_tokens_per_chunk=avg_tokens_per_chunk,
            json_files=json_files,
            txt_files=txt_files
        )
        
        logger.info(f"Processing completed: {valid_chunks} valid chunks from {len(files_to_process)} files")
        logger.info(f"Processing time: {processing_time:.3f}s")
        logger.info(f"Average tokens per chunk: {avg_tokens_per_chunk:.1f}")
        
        return {
            "success": True,
            "chunks": all_chunks,
            "statistics": statistics
        }


def main():
    """Test the enhanced data ingestion pipeline."""
    print("🧪 ENHANCED DATA INGESTION TEST")
    print("="*50)
    
    try:
        # Initialize pipeline
        pipeline = EnhancedDataIngestion(max_tokens_per_chunk=200)
        
        # Test JSON files
        print("\n📄 TESTING JSON FILES:")
        json_results = pipeline.process_json_files()
        
        if json_results["success"]:
            stats = json_results["statistics"]
            print(f"   ✅ JSON processing successful")
            print(f"   Files processed: {stats.files_processed}")
            print(f"   Total chunks: {stats.total_chunks}")
            print(f"   Valid chunks: {stats.valid_chunks}")
            print(f"   Processing time: {stats.processing_time:.3f}s")
        else:
            print(f"   ❌ JSON processing failed: {json_results.get('error')}")
        
        # Test TXT files
        print("\n📄 TESTING TXT FILES:")
        txt_results = pipeline.process_txt_files()
        
        if txt_results["success"]:
            stats = txt_results["statistics"]
            print(f"   ✅ TXT processing successful")
            print(f"   Files processed: {stats.files_processed}")
            print(f"   Total chunks: {stats.total_chunks}")
            print(f"   Valid chunks: {stats.valid_chunks}")
            print(f"   Processing time: {stats.processing_time:.3f}s")
        else:
            print(f"   ❌ TXT processing failed: {txt_results.get('error')}")
        
        # Test all files
        print("\n📄 TESTING ALL FILES:")
        all_results = pipeline.process_all_files()
        
        if all_results["success"]:
            stats = all_results["statistics"]
            print(f"   ✅ Combined processing successful")
            print(f"   Total files: {stats.files_processed}")
            print(f"   JSON files: {stats.json_files}")
            print(f"   TXT files: {stats.txt_files}")
            print(f"   Total chunks: {stats.total_chunks}")
            print(f"   Valid chunks: {stats.valid_chunks}")
            print(f"   Processing time: {stats.processing_time:.3f}s")
            
            # Show sample chunks
            if all_results["chunks"]:
                print(f"\n📋 SAMPLE CHUNKS:")
                for i, chunk in enumerate(all_results["chunks"][:3]):
                    print(f"   Chunk {i+1}: {chunk.chunk_id}")
                    print(f"   Category: {chunk.payload.get('category', 'N/A')}")
                    print(f"   Type: {chunk.payload.get('type', 'N/A')}")
                    print(f"   Tokens: {chunk.token_count}")
                    print(f"   Text preview: {chunk.text[:100]}...")
                    print()
        else:
            print(f"   ❌ Combined processing failed: {all_results.get('error')}")
        
        print(f"\n✅ Enhanced data ingestion test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced data ingestion test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    if not success:
        sys.exit(1)
