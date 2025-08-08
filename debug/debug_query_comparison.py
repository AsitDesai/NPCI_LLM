"""
Debug script to compare different queries and understand retrieval behavior
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from embeddings.embedder import LlamaIndexEmbedder
from embeddings.models import get_default_embedding_config
from embeddings.vector_store import QdrantVectorStore
from retrieval.retriever import SemanticRetriever

def debug_query_comparison():
    """Compare different queries to understand retrieval behavior."""
    print("🔍 DEBUGGING QUERY COMPARISON")
    print("=" * 60)
    
    # Initialize components
    print("1. Initializing components...")
    embedder = LlamaIndexEmbedder(get_default_embedding_config())
    vector_store = QdrantVectorStore()
    retriever = SemanticRetriever(embedder, vector_store)
    
    print("✅ Components initialized")
    
    # Test queries
    test_queries = [
        "how can i check my payment status",
        "which browsers can i use for this service", 
        "what is the refund policy",
        "how do i reset my password",
        "what payment methods do you accept"
    ]
    
    print(f"\n2. Testing {len(test_queries)} different queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*40}")
        print(f"🔍 QUERY {i}: '{query}'")
        print(f"{'='*40}")
        
        try:
            # Retrieve documents
            results = retriever.retrieve(query, top_k=3)
            
            if results:
                print(f"✅ Found {len(results)} results")
                for j, result in enumerate(results):
                    print(f"   📄 Result {j+1}: Score={result.score:.3f}")
                    print(f"      Text: {result.text[:100]}...")
                    print(f"      Source: {result.source_document}")
            else:
                print("❌ No results found")
                
                # Let's check what's in the vector store
                print("   🔍 Checking vector store contents...")
                try:
                    collection_info = vector_store.client.get_collection("rag_embeddings")
                    print(f"      Collection points: {collection_info.points_count}")
                    
                    # Try a broader search
                    print("   🔍 Trying broader search...")
                    broader_results = retriever.retrieve(query, top_k=10, score_threshold=0.1)
                    print(f"      Broader search found: {len(broader_results)} results")
                    
                except Exception as e:
                    print(f"      Error checking vector store: {e}")
                    
        except Exception as e:
            print(f"❌ Error with query: {e}")
    
    print(f"\n{'='*60}")
    print("📊 ANALYSIS")
    print(f"{'='*60}")
    
    # Check what documents are actually in the vector store
    print("\n3. Checking vector store contents...")
    try:
        collection_info = vector_store.client.get_collection("rag_embeddings")
        print(f"✅ Total points in collection: {collection_info.points_count}")
        
        # Get all points
        all_points = vector_store.client.scroll(
            collection_name="rag_embeddings",
            limit=20
        )
        
        print(f"✅ Retrieved {len(all_points[0])} points")
        
        # Analyze the content
        print("\n📄 CONTENT ANALYSIS:")
        for i, point in enumerate(all_points[0]):
            payload = point.payload
            text = payload.get('text', '')[:100] if 'text' in payload else 'No text'
            print(f"   Point {i+1}: {text}...")
            
    except Exception as e:
        print(f"❌ Error analyzing vector store: {e}")
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    debug_query_comparison() 