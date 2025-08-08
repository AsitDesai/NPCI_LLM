#!/usr/bin/env python3
"""
Test script to check server connectivity.
Tests basic network connectivity to the server endpoints.
"""

import os
import sys
import asyncio
import aiohttp
import requests
from typing import Dict, Any
from dotenv import load_dotenv

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_environment() -> Dict[str, Any]:
    """Load environment variables from env.server file."""
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env.server')
    load_dotenv(env_file)
    
    return {
        'api_host': os.getenv('API_HOST', '0.0.0.0'),
        'api_port': int(os.getenv('API_PORT', 8000)),
        'server_model_endpoint': os.getenv('SERVER_MODEL_ENDPOINT'),
        'qdrant_host': os.getenv('QDRANT_HOST', '0.0.0.0:6334'),
        'qdrant_port': int(os.getenv('QDRANT_PORT', 6334)),
    }

def test_basic_server_connectivity(host: str, port: int) -> bool:
    """Test basic TCP connectivity to the server."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"❌ Basic connectivity test failed: {e}")
        return False

def test_http_endpoint(url: str) -> Dict[str, Any]:
    """Test HTTP endpoint connectivity."""
    try:
        response = requests.get(url, timeout=10)
        return {
            'success': True,
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'headers': dict(response.headers)
        }
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Connection refused'}
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

async def test_async_connectivity(url: str) -> Dict[str, Any]:
    """Test async connectivity to an endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                return {
                    'success': True,
                    'status_code': response.status,
                    'headers': dict(response.headers)
                }
    except aiohttp.ClientConnectorError:
        return {'success': False, 'error': 'Connection refused'}
    except asyncio.TimeoutError:
        return {'success': False, 'error': 'Request timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    """Main test function."""
    print("🔍 Testing Server Connectivity")
    print("=" * 50)
    
    # Load environment configuration
    config = load_environment()
    
    print(f"📋 Configuration loaded:")
    print(f"   API Host: {config['api_host']}")
    print(f"   API Port: {config['api_port']}")
    print(f"   Model Endpoint: {config['server_model_endpoint']}")
    print(f"   Qdrant Host: {config['qdrant_host']}")
    print(f"   Qdrant Port: {config['qdrant_port']}")
    print()
    
    # Test 1: Basic server connectivity
    print("1️⃣ Testing Basic Server Connectivity")
    print("-" * 40)
    
    api_url = f"http://{config['api_host']}:{config['api_port']}"
    basic_connectivity = test_basic_server_connectivity(config['api_host'], config['api_port'])
    
    if basic_connectivity:
        print("✅ Basic TCP connectivity successful")
    else:
        print("❌ Basic TCP connectivity failed")
    
    # Test 2: HTTP endpoint test
    print("\n2️⃣ Testing HTTP Endpoint")
    print("-" * 40)
    
    http_result = test_http_endpoint(api_url)
    if http_result['success']:
        print(f"✅ HTTP endpoint accessible")
        print(f"   Status Code: {http_result['status_code']}")
        print(f"   Response Time: {http_result['response_time']:.3f}s")
    else:
        print(f"❌ HTTP endpoint test failed: {http_result['error']}")
    
    # Test 3: Async connectivity test
    print("\n3️⃣ Testing Async Connectivity")
    print("-" * 40)
    
    async_result = asyncio.run(test_async_connectivity(api_url))
    if async_result['success']:
        print(f"✅ Async connectivity successful")
        print(f"   Status Code: {async_result['status_code']}")
    else:
        print(f"❌ Async connectivity failed: {async_result['error']}")
    
    # Test 4: Model endpoint test (if configured)
    if config['server_model_endpoint']:
        print("\n4️⃣ Testing Model Endpoint")
        print("-" * 40)
        
        model_result = test_http_endpoint(config['server_model_endpoint'])
        if model_result['success']:
            print(f"✅ Model endpoint accessible")
            print(f"   Status Code: {model_result['status_code']}")
            print(f"   Response Time: {model_result['response_time']:.3f}s")
        else:
            print(f"❌ Model endpoint test failed: {model_result['error']}")
    
    # Summary
    print("\n📊 Summary")
    print("=" * 50)
    tests_passed = 0
    total_tests = 3 + (1 if config['server_model_endpoint'] else 0)
    
    if basic_connectivity:
        tests_passed += 1
    if http_result['success']:
        tests_passed += 1
    if async_result['success']:
        tests_passed += 1
    if config['server_model_endpoint'] and model_result['success']:
        tests_passed += 1
    
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All connectivity tests passed!")
        return 0
    else:
        print("⚠️  Some connectivity tests failed. Check your server configuration.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


