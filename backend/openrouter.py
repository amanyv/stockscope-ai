# backend/openrouter.py
from pathlib import Path
import httpx, json

# ---- Load config.json ----
CONFIG_PATH = Path(__file__).parent / "config.json"

if not CONFIG_PATH.exists():
    raise RuntimeError("config.json not found in backend/")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

OPENROUTER_API_KEY = config.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing in config.json")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


async def llm_complete(
    messages,
    model="openai/gpt-4.1-mini",
    max_tokens=600,
    temperature=0.2,
):
    """
    Chat completion helper
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"]


async def embed_text(
    text: str,
    model: str = "text-embedding-3-small",
):
    """
    Embedding helper for RAG
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": text,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    return data["data"][0]["embedding"]
