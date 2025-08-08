"""
Simple test for Mistral API only
"""

import os
from generation.prompt_templates import PromptTemplates, PromptStyle
from generation.generator import ResponseGenerator
from generation.post_processor import PostProcessor

def test_mistral_only():
    """Test Mistral API without embeddings."""
    print("🧪 Testing Mistral API Only...")
    
    try:
        # Set API key
        os.environ["MISTRAL_API_KEY"] = "htsiRa57UO5unjCb3vBAHk3HS0oP1s0l"
        
        # Initialize components
        templates = PromptTemplates()
        generator = ResponseGenerator()
        processor = PostProcessor()
        
        print("✅ Components initialized")
        
        # Test query
        query = "How do I check my payment status?"
        context = "You can check your payment status by logging into your account portal or calling customer service."
        
        # Format prompt
        prompt = templates.format_prompt(query, context, PromptStyle.CONCISE)
        print(f"✅ Prompt formatted ({len(prompt)} chars)")
        
        # Generate response
        response = generator.generate(prompt, style="concise")
        print(f"✅ Response generated ({len(response)} chars)")
        print(f"📝 Response: {response[:200]}...")
        
        # Process response
        processed = processor.process(response)
        print(f"✅ Response processed")
        print(f"📊 Confidence: {processed['confidence']:.2f}")
        
        # Display result
        display = processor.format_for_display(processed)
        print(f"\n🎯 FINAL RESULT:")
        print(display)
        
        print("🎉 Mistral API test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_mistral_only() 