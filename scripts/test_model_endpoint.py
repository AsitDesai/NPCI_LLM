#!/usr/bin/env python3
"""
Simple script to test if the model endpoint is working.
No fallbacks, no complex logic - just direct testing.
"""

import requests
import json

# Hardcoded endpoint - no environment variables
ENDPOINT_BASE = "http://183.82.7.228:9519"
CHAT_ENDPOINT = f"{ENDPOINT_BASE}/v1/chat/completions"

def test_model_endpoint():
    """Test if the model endpoint is working"""
    print("🔍 Testing Model Endpoint")
    print("=" * 50)
    print(f"Endpoint: {ENDPOINT_BASE}")
    print(f"Chat endpoint: {CHAT_ENDPOINT}")
    print("=" * 50)
    
    # Simple test payload
    payload = {
        "model": "NPCI_Greviance",
        "messages": [
            {"role": "user", "content": "Hello, this is a test"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        print("📤 Sending test request...")
        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: Model endpoint is working!")
            try:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"📝 Response: {content}")
            except:
                print(f"📝 Raw response: {response.text}")
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            print(f"📝 Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: {e}")
    
    print("=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_model_endpoint()
