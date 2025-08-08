#!/usr/bin/env python3
"""
Test script to check model endpoint functionality.
Tests connection, response quality, and basic inference capabilities.
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_environment() -> Dict[str, Any]:
    """Load environment variables from env.server file."""
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env.server')
    load_dotenv(env_file)
    
    return {
        'server_model_endpoint': os.getenv('SERVER_MODEL_ENDPOINT'),
        'server_model_api_key': os.getenv('SERVER_MODEL_API_KEY'),
        'mistral_api_key': os.getenv('MISTRAL_API_KEY'),
        'mistral_model': os.getenv('MISTRAL_MODEL', 'mistral-small-latest'),
    }

def test_endpoint_connectivity(endpoint: str, api_key: str = None) -> Dict[str, Any]:
    """Test basic connectivity to model endpoint."""
    try:
        headers = {}
        if api_key and api_key != 'your_server_model_api_key_here':
            headers['Authorization'] = f'Bearer {api_key}'
        
        # Test with health endpoint first
        health_url = endpoint.rstrip('/') + '/health'
        response = requests.get(health_url, headers=headers, timeout=10)
        
        return {
            'success': True,
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'headers': dict(response.headers),
            'content_type': response.headers.get('content-type', 'unknown')
        }
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Connection refused'}
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def test_model_inference(endpoint: str, api_key: str = None) -> Dict[str, Any]:
    """Test model inference with a simple prompt."""
    test_prompts = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Explain quantum computing in one sentence."
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        try:
            headers = {'Content-Type': 'application/json'}
            if api_key and api_key != 'your_server_model_api_key_here':
                headers['Authorization'] = f'Bearer {api_key}'
            
            # Try different payload formats
            payload_formats = [
                # Format 1: Chat completion format (OpenAI compatible)
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.7
                },
                # Format 2: Standard completion format
                {
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.7
                },
                # Format 3: Simple text format
                {
                    "text": prompt,
                    "max_length": 100
                }
            ]
            
            success = False
            for j, payload in enumerate(payload_formats):
                try:
                    # Use the chat completions endpoint
                    chat_url = endpoint.rstrip('/') + '/v1/chat/completions'
                    response = requests.post(
                        chat_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            'prompt': prompt,
                            'success': True,
                            'response': result,
                            'response_time': response.elapsed.total_seconds(),
                            'payload_format': j + 1
                        })
                        success = True
                        break
                    else:
                        print(f"   Format {j+1} failed with status {response.status_code}")
                        
                except Exception as e:
                    print(f"   Format {j+1} failed: {str(e)}")
                    continue
            
            if not success:
                results.append({
                    'prompt': prompt,
                    'success': False,
                    'error': 'All payload formats failed'
                })
                
        except Exception as e:
            results.append({
                'prompt': prompt,
                'success': False,
                'error': str(e)
            })
    
    return {
        'success': any(r['success'] for r in results),
        'results': results,
        'successful_tests': sum(1 for r in results if r['success'])
    }

async def test_async_inference(endpoint: str, api_key: str = None) -> Dict[str, Any]:
    """Test async model inference."""
    try:
        headers = {'Content-Type': 'application/json'}
        if api_key and api_key != 'your_server_model_api_key_here':
            headers['Authorization'] = f'Bearer {api_key}'
        
        payload = {
            "prompt": "Hello, this is an async test.",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            chat_url = endpoint.rstrip('/') + '/v1/chat/completions'
            async with session.post(
                chat_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'status_code': response.status,
                        'response': result,
                        'response_time': response.headers.get('X-Response-Time', 'unknown')
                    }
                else:
                    return {
                        'success': False,
                        'error': f'HTTP {response.status}',
                        'status_code': response.status
                    }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def test_response_quality(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze response quality from successful tests."""
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        return {
            'success': False,
            'error': 'No successful responses to analyze'
        }
    
    analysis = {
        'total_responses': len(successful_results),
        'avg_response_time': 0,
        'response_lengths': [],
        'content_analysis': {}
    }
    
    total_time = 0
    for result in successful_results:
        response_time = result.get('response_time', 0)
        total_time += response_time
        
        # Analyze response content
        response_content = str(result.get('response', ''))
        analysis['response_lengths'].append(len(response_content))
        
        # Basic content analysis
        if 'error' in response_content.lower():
            analysis['content_analysis']['contains_error'] = True
        
        if len(response_content) < 10:
            analysis['content_analysis']['very_short'] = True
    
    analysis['avg_response_time'] = total_time / len(successful_results) if successful_results else 0
    analysis['min_response_length'] = min(analysis['response_lengths']) if analysis['response_lengths'] else 0
    analysis['max_response_length'] = max(analysis['response_lengths']) if analysis['response_lengths'] else 0
    
    return {
        'success': True,
        'analysis': analysis
    }

def test_mistral_fallback(config: Dict[str, Any]) -> Dict[str, Any]:
    """Test Mistral API as fallback if server model fails."""
    if not config['mistral_api_key'] or config['mistral_api_key'] == 'your_mistral_api_key_here':
        return {
            'success': False,
            'error': 'Mistral API key not configured'
        }
    
    try:
        import mistralai
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        
        client = MistralClient(api_key=config['mistral_api_key'])
        
        response = client.chat(
            model=config['mistral_model'],
            messages=[ChatMessage(role="user", content="Hello, this is a test.")]
        )
        
        return {
            'success': True,
            'response': response.choices[0].message.content,
            'model': config['mistral_model']
        }
    except ImportError:
        return {
            'success': False,
            'error': 'mistralai package not installed'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main test function."""
    print("🔍 Testing Model Endpoint")
    print("=" * 50)
    
    # Load environment configuration
    config = load_environment()
    
    print(f"📋 Configuration loaded:")
    print(f"   Server Model Endpoint: {config['server_model_endpoint']}")
    print(f"   Mistral Model: {config['mistral_model']}")
    print()
    
    if not config['server_model_endpoint']:
        print("❌ No model endpoint configured")
        print("   Please set SERVER_MODEL_ENDPOINT in env.server")
        return 1
    
    # Test 1: Endpoint connectivity
    print("1️⃣ Testing Endpoint Connectivity")
    print("-" * 40)
    
    connectivity_result = test_endpoint_connectivity(
        config['server_model_endpoint'],
        config['server_model_api_key']
    )
    
    if connectivity_result['success']:
        print(f"✅ Endpoint connectivity successful")
        print(f"   Status Code: {connectivity_result['status_code']}")
        print(f"   Response Time: {connectivity_result['response_time']:.3f}s")
        print(f"   Content Type: {connectivity_result['content_type']}")
    else:
        print(f"❌ Endpoint connectivity failed: {connectivity_result['error']}")
    
    # Test 2: Model inference
    print("\n2️⃣ Testing Model Inference")
    print("-" * 40)
    
    inference_result = test_model_inference(
        config['server_model_endpoint'],
        config['server_model_api_key']
    )
    
    if inference_result['success']:
        print(f"✅ Model inference successful")
        print(f"   Successful tests: {inference_result['successful_tests']}/3")
        
        # Show sample response
        for result in inference_result['results']:
            if result['success']:
                response_text = str(result.get('response', ''))[:100]
                print(f"   Sample response: {response_text}...")
                break
    else:
        print(f"❌ Model inference failed")
        for result in inference_result['results']:
            if not result['success']:
                print(f"   Error: {result['error']}")
    
    # Test 3: Async inference
    print("\n3️⃣ Testing Async Inference")
    print("-" * 40)
    
    async_result = asyncio.run(test_async_inference(
        config['server_model_endpoint'],
        config['server_model_api_key']
    ))
    
    if async_result['success']:
        print(f"✅ Async inference successful")
        print(f"   Status Code: {async_result['status_code']}")
    else:
        print(f"❌ Async inference failed: {async_result['error']}")
    
    # Test 4: Response quality analysis
    print("\n4️⃣ Testing Response Quality")
    print("-" * 40)
    
    if inference_result['success']:
        quality_result = test_response_quality(inference_result['results'])
        if quality_result['success']:
            analysis = quality_result['analysis']
            print(f"✅ Response quality analysis successful")
            print(f"   Total responses: {analysis['total_responses']}")
            print(f"   Avg response time: {analysis['avg_response_time']:.3f}s")
            print(f"   Response length range: {analysis['min_response_length']}-{analysis['max_response_length']}")
        else:
            print(f"❌ Response quality analysis failed: {quality_result['error']}")
    else:
        print("⚠️  Skipping response quality analysis (no successful responses)")
    
    # Test 5: Mistral fallback
    print("\n5️⃣ Testing Mistral Fallback")
    print("-" * 40)
    
    mistral_result = test_mistral_fallback(config)
    if mistral_result['success']:
        print(f"✅ Mistral fallback successful")
        print(f"   Model: {mistral_result['model']}")
        print(f"   Response: {mistral_result['response'][:50]}...")
    else:
        print(f"⚠️  Mistral fallback not available: {mistral_result['error']}")
    
    # Summary
    print("\n📊 Summary")
    print("=" * 50)
    tests_passed = 0
    total_tests = 5
    
    if connectivity_result['success']:
        tests_passed += 1
    if inference_result['success']:
        tests_passed += 1
    if async_result['success']:
        tests_passed += 1
    if inference_result['success'] and test_response_quality(inference_result['results'])['success']:
        tests_passed += 1
    if mistral_result['success']:
        tests_passed += 1
    
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed >= 3:
        print("🎉 Model endpoint is functional!")
        return 0
    elif tests_passed >= 2:
        print("⚠️  Model endpoint partially functional. Some issues detected.")
        return 0
    else:
        print("❌ Model endpoint has significant issues. Check configuration.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
