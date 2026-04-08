#!/usr/bin/env python3
"""Test API connection to OpenRouter with GPT-4.1."""

import os
import sys
from openai import OpenAI

# Get credentials from environment
api_key = os.getenv("HF_TOKEN") or os.getenv("OPENROUTER_API_KEY")
api_base_url = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
model_name = os.getenv("MODEL_NAME", "openai/gpt-4.1")

print("=" * 60)
print("🧪 OpenRouter API Connection Test")
print("=" * 60)

# Check credentials
if not api_key:
    print("❌ ERROR: No API key found!")
    print("   Set HF_TOKEN or OPENROUTER_API_KEY environment variable")
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
    print("  1. Insufficient OpenRouter credits")
    print("  2. Invalid API key")
    print("  3. Network connectivity issue")
    print("  4. Invalid model name")
    sys.exit(1)
