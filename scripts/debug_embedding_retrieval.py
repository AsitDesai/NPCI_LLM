#!/usr/bin/env python3
"""
Debug Embedding Retrieval

This script helps debug what's being retrieved from our embeddings.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
from simple_retriever import SimpleSemanticRetriever


def debug_retrieval():
    """Debug what's being retrieved."""
    print("🔍 DEBUGGING EMBEDDING RETRIEVAL")
    print("="*60)
    
    try:
        # Initialize retriever
        retriever = SimpleSemanticRetriever()
        
        # Get collection info
        collection_info = retriever.get_collection_info()
        print(f"📊 Collection Info: {collection_info}")
        
        # Test queries that should match the content
        test_queries = [
            "insufficient balance",
            "UPI PIN incorrect",
            "collect request expired",
            "payee PSP timeout",
            "mandate hold",
            "bank unavailable",
            "first time user cooling off",
            "suspected fraud"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 40)
            
            results = retriever.retrieve(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"Result {i}:")
                    print(f"  Score: {result.score:.3f}")
                    print(f"  Source: {result.source_file}")
                    print(f"  Text: {result.text[:200]}...")
                    print()
            else:
                print("  No results found")
                
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    debug_retrieval()
