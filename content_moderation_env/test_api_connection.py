#!/usr/bin/env python3
"""Test API connection to HuggingFace Router with Llama-3.3-70B-Instruct."""

import os
import sys
from openai import OpenAI

# Get credentials from environment
api_key = os.getenv("HF_TOKEN")
api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

print("=" * 60)
print("🧪 HuggingFace Router API Connection Test")
print("=" * 60)

# Check credentials
if not api_key:
    print("❌ ERROR: No API key found!")
    print("   Set HF_TOKEN environment variable")
    sys.exit(1)

print(f"✓ API Key: {api_key[:20]}...")
print(f"✓ Base URL: {api_base_url}")
print(f"✓ Model: {model_name}")
print()

# Initialize client
print("Initializing OpenAI client...")
try:
    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url
    )
    print("✓ Client initialized")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
    sys.exit(1)

# Test API call
print()
print("Testing API call...")
print("-" * 60)

try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=5,
        temperature=0
    )
    
    print("✓ API Response received!")
    print()
    print("Response:")
    message_content = response.choices[0].message.content
    print(f"  {message_content}")
    print()
    print("=" * 60)
    print("✅ API Connection Test PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ API call failed: {e}")
    print()
    print("Possible causes:")
    print("  1. Invalid or missing HF_TOKEN")
    print("  2. Invalid API key")
    print("  3. Network connectivity issue")
    print("  4. Invalid model name")
    sys.exit(1)
