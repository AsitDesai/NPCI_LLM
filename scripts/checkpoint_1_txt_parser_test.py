#!/usr/bin/env python3
"""
Checkpoint 1: TXT Parser Validation Test

This script validates that the enhanced data ingestion pipeline correctly:
1. Parses TXT files with structured Q&A format
2. Extracts categories, scenarios, FAQs, and guidelines
3. Maintains proper data structure for downstream processing
4. Generates valid chunks for embedding generation
"""

import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.enhanced_data_ingestion import EnhancedDataIngestion, TXTParser
from config.logging import get_logger

logger = get_logger(__name__)


def test_txt_parser_detailed():
    """Detailed test of TXT parser functionality."""
    print("🔍 CHECKPOINT 1: DETAILED TXT PARSER TEST")
    print("="*60)
    
    try:
        # Initialize parser
        parser = TXTParser()
        pipeline = EnhancedDataIngestion(max_tokens_per_chunk=200)
        
        # Test file path
        txt_file = Path("reference_documents/upi_decline.txt")
        
        if not txt_file.exists():
            print(f"❌ Test file not found: {txt_file}")
            return False
        
        print(f"📄 Testing file: {txt_file}")
        
        # Parse the file
        chunks = parser.parse_txt_file(txt_file)
        
        if not chunks:
            print("❌ No chunks extracted from TXT file")
            return False
        
        print(f"✅ Successfully extracted {len(chunks)} chunks")
        
        # Analyze chunk types
        chunk_types = {}
        categories = set()
        scenarios = set()
        faqs = set()
        guidelines = set()
        
        for chunk in chunks:
            chunk_type = chunk.get('type', 'unknown')
            category = chunk.get('category', 'unknown')
            
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            categories.add(category)
            
            if chunk_type == 'scenario':
                scenarios.add(chunk.get('scenario', 'unknown'))
            elif chunk_type == 'faq':
                faqs.add(chunk.get('user_statement', 'unknown')[:50])
            elif chunk_type == 'guideline':
                guidelines.add(chunk.get('agent_guideline', 'unknown')[:50])
        
        print(f"\n📊 CHUNK ANALYSIS:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Categories found: {len(categories)}")
        print(f"   Chunk types: {chunk_types}")
        
        print(f"\n📋 CATEGORIES:")
        for category in sorted(categories):
            print(f"   - {category}")
        
        print(f"\n🎭 SCENARIOS ({len(scenarios)}):")
        for scenario in sorted(list(scenarios)[:5]):  # Show first 5
            print(f"   - {scenario}")
        if len(scenarios) > 5:
            print(f"   ... and {len(scenarios) - 5} more")
        
        print(f"\n❓ FAQS ({len(faqs)}):")
        for faq in sorted(list(faqs)[:3]):  # Show first 3
            print(f"   - {faq}...")
        if len(faqs) > 3:
            print(f"   ... and {len(faqs) - 3} more")
        
        print(f"\n📝 GUIDELINES ({len(guidelines)}):")
        for guideline in sorted(list(guidelines)[:2]):  # Show first 2
            print(f"   - {guideline}...")
        if len(guidelines) > 2:
            print(f"   ... and {len(guidelines) - 2} more")
        
        # Validate chunk structure
        print(f"\n🔍 VALIDATING CHUNK STRUCTURE:")
        valid_chunks = 0
        invalid_chunks = 0
        
        for i, chunk in enumerate(chunks):
            required_fields = ['category', 'type']
            missing_fields = [field for field in required_fields if field not in chunk]
            
            if missing_fields:
                print(f"   ❌ Chunk {i}: Missing fields {missing_fields}")
                invalid_chunks += 1
            else:
                valid_chunks += 1
        
        print(f"   ✅ Valid chunks: {valid_chunks}")
        print(f"   ❌ Invalid chunks: {invalid_chunks}")
        
        # Test chunk processing
        print(f"\n⚙️ TESTING CHUNK PROCESSING:")
        processed_chunks = []
        
        for i, chunk_obj in enumerate(chunks):
            processed_chunk = pipeline.process_chunk_object(chunk_obj, txt_file.name, i)
            if processed_chunk:
                processed_chunks.append(processed_chunk)
        
        print(f"   ✅ Successfully processed {len(processed_chunks)} chunks")
        
        # Show sample processed chunks
        if processed_chunks:
            print(f"\n📋 SAMPLE PROCESSED CHUNKS:")
            for i, chunk in enumerate(processed_chunks[:3]):
                print(f"   Chunk {i+1}:")
                print(f"     ID: {chunk.chunk_id}")
                print(f"     Category: {chunk.payload.get('category')}")
                print(f"     Type: {chunk.payload.get('type')}")
                print(f"     Tokens: {chunk.token_count}")
                print(f"     Text preview: {chunk.text[:80]}...")
                print()
        
        # Validate token counts
        token_counts = [chunk.token_count for chunk in processed_chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        
        print(f"📊 TOKEN STATISTICS:")
        print(f"   Average tokens per chunk: {avg_tokens:.1f}")
        print(f"   Maximum tokens per chunk: {max_tokens}")
        print(f"   Chunks within limit: {sum(1 for tc in token_counts if tc <= 200)}")
        print(f"   Chunks exceeding limit: {sum(1 for tc in token_counts if tc > 200)}")
        
        # Final validation
        success_criteria = [
            len(chunks) > 0,
            valid_chunks > 0,
            len(processed_chunks) > 0,
            len(categories) > 0,
            'scenario' in chunk_types,
            'faq' in chunk_types,
            avg_tokens > 0
        ]
        
        all_passed = all(success_criteria)
        
        print(f"\n🎯 CHECKPOINT 1 VALIDATION:")
        print(f"   ✅ Chunks extracted: {len(chunks) > 0}")
        print(f"   ✅ Valid chunk structure: {valid_chunks > 0}")
        print(f"   ✅ Chunks processed: {len(processed_chunks) > 0}")
        print(f"   ✅ Categories found: {len(categories) > 0}")
        print(f"   ✅ Scenarios extracted: {'scenario' in chunk_types}")
        print(f"   ✅ FAQs extracted: {'faq' in chunk_types}")
        print(f"   ✅ Token calculation: {avg_tokens > 0}")
        
        if all_passed:
            print(f"\n🎉 CHECKPOINT 1 PASSED! TXT parser is working correctly.")
            print(f"   Ready to proceed to Step 2: Enhanced Embedding Builder")
            return True
        else:
            print(f"\n❌ CHECKPOINT 1 FAILED! Some validation criteria not met.")
            return False
            
    except Exception as e:
        logger.error(f"Checkpoint 1 test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


def main():
    """Run the checkpoint test."""
    success = test_txt_parser_detailed()
    
    if success:
        print(f"\n✅ CHECKPOINT 1 COMPLETED SUCCESSFULLY")
        print(f"   Next step: Update embedding builder for TXT support")
        return True
    else:
        print(f"\n❌ CHECKPOINT 1 FAILED")
        print(f"   Please fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
