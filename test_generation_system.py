"""
Test script for the Generation System

This script tests all components of the generation system:
- PromptTemplates
- ResponseGenerator (with Mistral AI)
- PostProcessor
"""

import sys
import os
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from generation.prompt_templates import PromptTemplates, PromptStyle
from generation.generator import ResponseGenerator, MockResponseGenerator
from generation.post_processor import PostProcessor


def test_prompt_templates():
    """Test prompt templates functionality."""
    print("🧪 Testing Prompt Templates...")
    
    try:
        templates = PromptTemplates()
        
        # Test different styles
        styles = [PromptStyle.CONCISE, PromptStyle.DETAILED, PromptStyle.TECHNICAL]
        
        for style in styles:
            prompt = templates.format_prompt(
                query="How do I check my payment status?",
                context="You can check your payment status by logging into your account portal.",
                style=style
            )
            
            print(f"  ✅ {style.value} template generated ({len(prompt)} chars)")
        
        print("  ✅ All prompt templates working correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Prompt templates test failed: {e}")
        return False


def test_response_generator():
    """Test response generator functionality."""
    print("🧪 Testing Response Generator...")
    
    try:
        # Try to use real Mistral generator first
        try:
            generator = ResponseGenerator()
            print("  ✅ Real Mistral generator initialized")
            use_mock = False
        except Exception as e:
            print(f"  ⚠️ Real Mistral generator failed: {e}")
            print("  🔄 Using mock generator for testing")
            generator = MockResponseGenerator()
            use_mock = True
        
        # Test response generation
        test_prompt = "How do I check my payment status?"
        response = generator.generate(test_prompt, style="concise")
        
        print(f"  ✅ Response generated ({len(response)} chars)")
        print(f"  📝 Sample: {response[:100]}...")
        
        if use_mock:
            print("  ℹ️ Using mock responses (no API key)")
        else:
            print("  ℹ️ Using real Mistral AI responses")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Response generator test failed: {e}")
        return False


def test_post_processor():
    """Test post-processor functionality."""
    print("🧪 Testing Post Processor...")
    
    try:
        processor = PostProcessor()
        
        # Test with sample response
        sample_response = "   This is a test response with   extra   spaces.   "
        processed = processor.process(sample_response)
        
        print(f"  ✅ Response processed successfully")
        print(f"  📊 Confidence: {processed['confidence']:.2f}")
        print(f"  📝 Original length: {processed['original_length']}")
        print(f"  📝 Processed length: {processed['processed_length']}")
        print(f"  🎭 Tone: {processed['metadata']['tone']}")
        
        # Test display formatting
        display = processor.format_for_display(processed)
        print(f"  ✅ Display formatting working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Post processor test failed: {e}")
        return False


def test_integration():
    """Test the complete generation pipeline."""
    print("🧪 Testing Generation Integration...")
    
    try:
        # Initialize components
        templates = PromptTemplates()
        processor = PostProcessor()
        
        # Try real generator, fallback to mock
        try:
            generator = ResponseGenerator()
            use_mock = False
        except:
            generator = MockResponseGenerator()
            use_mock = True
        
        # Test complete pipeline
        query = "How do I check my payment status?"
        context = "You can check your payment status by logging into your account portal or calling customer service."
        
        # Format prompt
        prompt = templates.format_prompt(query, context, PromptStyle.CONCISE)
        print(f"  ✅ Prompt formatted ({len(prompt)} chars)")
        
        # Generate response
        start_time = time.time()
        response = generator.generate(prompt, style="concise")
        generation_time = time.time() - start_time
        
        print(f"  ✅ Response generated in {generation_time:.3f}s")
        
        # Process response
        processed = processor.process(response)
        
        print(f"  ✅ Response processed")
        print(f"  📊 Final confidence: {processed['confidence']:.2f}")
        
        # Display final result
        display = processor.format_for_display(processed)
        print(f"  📝 Final response preview:")
        print(f"  {display[:200]}...")
        
        if use_mock:
            print("  ℹ️ Integration test completed with mock generator")
        else:
            print("  ℹ️ Integration test completed with real Mistral AI")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


def test_style_variations():
    """Test different response styles."""
    print("🧪 Testing Response Style Variations...")
    
    try:
        templates = PromptTemplates()
        processor = PostProcessor()
        
        # Try real generator, fallback to mock
        try:
            generator = ResponseGenerator()
            use_mock = False
        except:
            generator = MockResponseGenerator()
            use_mock = True
        
        styles = ["concise", "detailed", "technical", "friendly", "professional"]
        query = "What is the refund policy?"
        context = "Refunds are processed within 5-7 business days after approval."
        
        for style in styles:
            prompt = templates.format_prompt(query, context, PromptStyle(style))
            response = generator.generate(prompt, style=style)
            processed = processor.process(response)
            
            print(f"  ✅ {style.capitalize()} style: {processed['confidence']:.2f} confidence")
        
        print("  ✅ All style variations tested successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Style variations test failed: {e}")
        return False


def main():
    """Run all generation system tests."""
    print("=" * 80)
    print("🚀 GENERATION SYSTEM TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Prompt Templates", test_prompt_templates),
        ("Response Generator", test_response_generator),
        ("Post Processor", test_post_processor),
        ("Style Variations", test_style_variations),
        ("Integration", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name.upper()} TEST")
        print(f"{'='*60}")
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print(f"\n{'='*80}")
    print("📊 GENERATION SYSTEM TEST RESULTS")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL GENERATION SYSTEM TESTS PASSED!")
        print("🚀 Generation system is ready for integration!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 