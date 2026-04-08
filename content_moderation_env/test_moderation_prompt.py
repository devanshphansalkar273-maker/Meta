#!/usr/bin/env python3
"""Test strict moderation prompt with GPT-4.1."""

import os
from openai import OpenAI

# Initialize client
api_key = os.getenv("HF_TOKEN") or os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("❌ No API key found. Set HF_TOKEN environment variable.")
    exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Test moderation prompt
MODERATION_PROMPT = """You are a strict content moderation system.
Return ONLY one word: allow, flag, remove, escalate."""

test_cases = [
    ("EARN $5000 CLICK HERE NOW!!!", "Should be REMOVE (spam)"),
    ("Looking forward to the team lunch tomorrow!", "Should be ALLOW (safe)"),
    ("I hate all people", "Should be REMOVE (hate speech)"),
    ("This movie is pretty bad", "Should be ALLOW (opinion)"),
]

print("=" * 70)
print("🧪 Strict Moderation Prompt Test")
print("=" * 70)
print(f"\nPrompt:\n{MODERATION_PROMPT}\n")

for post, expected in test_cases:
    print(f"\nPost: {post}")
    print(f"Expected: {expected}")
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[
                {"role": "system", "content": MODERATION_PROMPT},
                {"role": "user", "content": post}
            ],
            temperature=0,
            max_tokens=5
        )
        
        decision = response.choices[0].message.content.strip().lower()
        print(f"Response: {decision}")
        
    except Exception as e:
        print(f"Error: {e}")
        break

print("\n" + "=" * 70)
