"""
Central config. Loads settings from a .env file (if present) or real
environment variables.

Supports two provider modes, both using the OpenAI-compatible SDK:
  - "groq"   (default) -> free tier, no credit card, get a key at
              https://console.groq.com/keys
  - "openai" -> paid, get a key at https://platform.openai.com/api-keys

Set LLM_PROVIDER=openai in .env to switch.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; real env vars still work without it

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

if LLM_PROVIDER == "openai":
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    BASE_URL = "https://api.openai.com/v1"
    MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
else:  # groq
    API_KEY = os.environ.get("GROQ_API_KEY", "")
    BASE_URL = "https://api.groq.com/openai/v1"
    MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "20"))
MAX_TOOL_ITERATIONS = int(os.environ.get("MAX_TOOL_ITERATIONS", "5"))


def is_configured() -> bool:
    return bool(API_KEY)


def setup_instructions() -> str:
    if LLM_PROVIDER == "openai":
        return (
            "No OPENAI_API_KEY found.\n"
            "1. Get a key: https://platform.openai.com/api-keys\n"
            "2. Copy .env.example to .env\n"
            "3. Set OPENAI_API_KEY=your_key_here in .env"
        )
    return (
        "No GROQ_API_KEY found.\n"
        "1. Get a FREE key (no credit card needed): https://console.groq.com/keys\n"
        "2. Copy .env.example to .env\n"
        "3. Set GROQ_API_KEY=your_key_here in .env"
    )
