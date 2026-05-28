#!/usr/bin/env python3
"""Quick test: ping each NIM model with a minimal tool-calling request."""
import os, json, requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("NIM_API_KEY")
BASE    = "https://integrate.api.nvidia.com/v1"

NIM_MODELS = [
    "meta/llama-3.3-70b-instruct",
    "google/gemma-4-31b-it",
]
OR_MODELS = [
    "anthropic/claude-sonnet-4-5",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat-v3-0324",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2411",
]

TOOL = {
    "type": "function",
    "function": {
        "name": "adjust_palette",
        "description": "Apply color axis shifts.",
        "parameters": {
            "type": "object",
            "properties": {
                "axes": {"type": "object", "properties": {"warmth": {"type": "number"}}},
            },
            "required": ["axes"],
        },
    },
}

OR_KEY = os.getenv("OPENROUTER_API_KEY")

def test_model(model, base_url, api_key, extra_headers={}):
    print(f"\n{'─'*50}\nTesting: {model}")
    try:
        res = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", **extra_headers},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Make the palette warmer. Call adjust_palette now."}],
                "tools": [TOOL],
                "max_tokens": 256,
                "stream": False,
            },
            timeout=30,
        )
        if res.ok:
            msg = res.json()["choices"][0]["message"]
            tc = msg.get("tool_calls")
            if tc:
                print(f"  ✓ Tool call: {tc[0]['function']['name']}({tc[0]['function']['arguments'][:80]})")
            else:
                print(f"  ✗ Text only: {msg.get('content','')[:120]}")
        else:
            print(f"  ✗ HTTP {res.status_code}: {res.text[:200]}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n═══ NIM ═══")
for m in NIM_MODELS:
    test_model(m, BASE, API_KEY)

print("\n═══ OpenRouter ═══")
for m in OR_MODELS:
    test_model(m, "https://openrouter.ai/api/v1", OR_KEY,
               {"HTTP-Referer": "http://localhost:8010", "X-Title": "autoRecolor"})
