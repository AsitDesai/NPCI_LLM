"""
Simple debug script to test text extraction without vector store
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_simple_text():
    """Debug text extraction with mock data."""
    print("🔍 DEBUGGING TEXT EXTRACTION (Simple)")
    print("=" * 60)
    
    # Mock LlamaIndex Document objects based on what we saw in the logs
    class MockDocument:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata
    
    # Create mock documents based on what we saw in the debug output
    mock_docs = [
        MockDocument(
            text="** A: Yes, we offer a 30-day money-back guarantee. If you're not completely satisfied with our service, you can request a full refund within 30 days of your purchase. **Q5: How do I update my billing information?** A: You can update your billing information by going to Account Settings > Billing Information. Make sure to update both your payment method and billing address. ## Account Management **Q6: How do I reset my password?** A: Click on \"Forgot Password\" on the login page, enter your email address, and follow the instructions sent to your email. The reset link will expire in 24 hours. **Q7: Can I change my account email address?** A: Yes, you can change your email address in Account Settings > Profile. You'll need to verify the new email address before the change takes effect.",
            metadata={'file_name': 'faq.txt', 'chunk_index': 2}
        ),
        MockDocument(
            text="If your payment remains pending for more than 5 business days, please contact our support team with your transaction ID. **Q3: What payment methods do you accept?** A: We accept all major credit cards (Visa, MasterCard, American Express), debit cards, bank transfers, and digital wallets including PayPal and Apple Pay. **Q4: Can I get a refund if I'm not satisfied?** A: Yes, we offer a 30-day money-back guarantee. If you're not completely satisfied with our service, you can request a full refund within 30 days of your purchase. **Q5: How do I update my billing information?** A: You can update your billing information by going to Account Settings > Billing Information. Make sure to update both your payment method and billing address. ## Account Management **Q6: How do I reset my password?",
            metadata={'file_name': 'faq.txt', 'chunk_index': 1}
        )
    ]
    
    print("1. Testing text extraction from mock documents...")
    
    for i, doc in enumerate(mock_docs):
        print(f"\n📄 Mock Document {i+1}:")
        print(f"   Text length: {len(doc.text)}")
        print(f"   Text preview: {doc.text[:100]}...")
        print(f"   Text is empty: {not doc.text.strip()}")
        print(f"   Metadata: {doc.metadata}")
    
    # Test the conversion logic from RAG CLI
    print("\n2. Testing conversion logic from RAG CLI...")
    
    from retrieval.retriever import RetrievalResult
    
    for i, doc in enumerate(mock_docs):
        print(f"\n🔍 Converting Mock Document {i+1} to RetrievalResult:")
        
        # Use the same logic as in RAG CLI
        text = doc.text if hasattr(doc, 'text') else str(doc)
        
        retrieval_result = RetrievalResult(
            text=text,
            score=0.8,
            metadata=doc.metadata if hasattr(doc, 'metadata') else {},
            source_document=doc.metadata.get('file_name', 'unknown') if hasattr(doc, 'metadata') else 'unknown',
            chunk_index=doc.metadata.get('chunk_index', i) if hasattr(doc, 'metadata') else i
        )
        
        print(f"   Text length: {len(retrieval_result.text)}")
        print(f"   Text preview: {retrieval_result.text[:100]}...")
        print(f"   Text is empty: {not retrieval_result.text.strip()}")
        print(f"   Source: {retrieval_result.source_document}")
    
    # Test context building
    print("\n3. Testing context building...")
    
    try:
        from retrieval.context_builder import ContextBuilder
        
        context_builder = ContextBuilder()
        
        # Convert mock docs to RetrievalResult objects
        retrieval_results = []
        for i, doc in enumerate(mock_docs):
            text = doc.text if hasattr(doc, 'text') else str(doc)
            
            retrieval_result = RetrievalResult(
                text=text,
                score=0.8,
                metadata=doc.metadata if hasattr(doc, 'metadata') else {},
                source_document=doc.metadata.get('file_name', 'unknown') if hasattr(doc, 'metadata') else 'unknown',
                chunk_index=doc.metadata.get('chunk_index', i) if hasattr(doc, 'metadata') else i
            )
            retrieval_results.append(retrieval_result)
        
        # Build context
        context_info = context_builder.build_context(retrieval_results, "what is the refund policy")
        
        print(f"✅ Context built successfully")
        print(f"   Context length: {len(context_info.context)} characters")
        print(f"   Total tokens: {context_info.total_tokens}")
        print(f"   Num chunks: {context_info.num_chunks}")
        
        print(f"\n📝 CONTEXT SENT TO MODEL:")
        print("=" * 60)
        print(context_info.context)
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error building context: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    debug_simple_text() 