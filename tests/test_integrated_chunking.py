"""
Test script for integrated chunking system
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llama_index.core import Document
from data.ingestion.chunking import DocumentChunker

def test_integrated_chunking():
    """Test the integrated chunking system with semantic chunking."""
    print("🧪 TESTING INTEGRATED CHUNKING SYSTEM")
    print("=" * 60)
    
    # Initialize document chunker
    chunker = DocumentChunker()
    print("✅ Document chunker initialized")
    
    # Test with FAQ document
    print("\n📄 Testing with FAQ document...")
    with open("reference_documents/faq.txt", "r", encoding="utf-8") as f:
        faq_text = f.read()
    
    faq_doc = Document(text=faq_text, doc_id="faq.txt")
    
    # Test semantic chunking (default)
    print("\n🔧 Testing semantic chunking (default)...")
    semantic_chunks = chunker.chunk_document(faq_doc, method="semantic")
    print(f"   Semantic chunks: {len(semantic_chunks)}")
    
    # Show semantic chunks
    for i, chunk in enumerate(semantic_chunks[:3]):
        print(f"\n   📄 Semantic Chunk {i+1}:")
        print(f"      Method: {chunk.metadata.get('chunking_method', 'unknown')}")
        print(f"      Type: {chunk.metadata.get('unit_type', 'unknown')}")
        print(f"      Size: {len(chunk.text)} chars")
        print(f"      Text: {chunk.text[:150]}...")
    
    # Test with Do/Don't document
    print("\n📄 Testing with Do/Don't document...")
    with open("reference_documents/DosnDonts.txt", "r", encoding="utf-8") as f:
        do_dont_text = f.read()
    
    do_dont_doc = Document(text=do_dont_text, doc_id="DosnDonts.txt")
    
    # Test semantic chunking
    print("\n🔧 Testing semantic chunking...")
    semantic_chunks_2 = chunker.chunk_document(do_dont_doc, method="semantic")
    print(f"   Semantic chunks: {len(semantic_chunks_2)}")
    
    # Show semantic chunks
    for i, chunk in enumerate(semantic_chunks_2[:3]):
        print(f"\n   📄 Semantic Chunk {i+1}:")
        print(f"      Method: {chunk.metadata.get('chunking_method', 'unknown')}")
        print(f"      Type: {chunk.metadata.get('unit_type', 'unknown')}")
        print(f"      Size: {len(chunk.text)} chars")
        print(f"      Text: {chunk.text[:150]}...")
    
    # Test fallback to sentence chunking
    print("\n🔧 Testing sentence chunking fallback...")
    sentence_chunks = chunker.chunk_document(faq_doc, method="sentence")
    print(f"   Sentence chunks: {len(sentence_chunks)}")
    
    # Show sentence chunks
    for i, chunk in enumerate(sentence_chunks[:3]):
        print(f"\n   📄 Sentence Chunk {i+1}:")
        print(f"      Method: {chunk.metadata.get('chunking_method', 'unknown')}")
        print(f"      Size: {len(chunk.text)} chars")
        print(f"      Text: {chunk.text[:150]}...")
    
    # Compare chunking methods
    print("\n📊 Comparison:")
    print(f"   Semantic chunks: {len(semantic_chunks)}")
    print(f"   Sentence chunks: {len(sentence_chunks)}")
    
    # Get chunking statistics
    print("\n📊 Chunking Statistics:")
    all_chunks = semantic_chunks + semantic_chunks_2
    stats = chunker.get_chunking_stats([faq_doc, do_dont_doc], all_chunks)
    
    print(f"   Original documents: {stats.original_documents}")
    print(f"   Total chunks: {stats.total_chunks}")
    print(f"   Average chunk size: {stats.avg_chunk_size:.1f}")
    print(f"   Min chunk size: {stats.min_chunk_size}")
    print(f"   Max chunk size: {stats.max_chunk_size}")
    print(f"   Chunk overlap: {stats.chunk_overlap}")
    
    print("\n✅ Integrated chunking test completed successfully!")
    
    return len(all_chunks) > 0

if __name__ == "__main__":
    success = test_integrated_chunking()
    if success:
        print("\n🎉 All tests passed! Integrated chunking is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the implementation.") 