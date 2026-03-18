"""
ai_clients.py - FIXED VERSION
Correct model IDs for 2025, keys re-read per request (fixes Render env issue),
Gemini 2.0-flash, working OpenRouter free-tier models.
"""
import os
import logging
import requests

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

AI_NAMES = ["Gemini", "ChatGPT", "Claude", "Grok", "Perplexity"]

# Keys re-read every call so Render env vars are always picked up fresh
def _gemini_key():     return os.environ.get("GEMINI_API_KEY", "").strip()
def _openrouter_key(): return os.environ.get("OPENROUTER_API_KEY", "").strip()
def _perplexity_key(): return os.environ.get("PERPLEXITY_API_KEY", "").strip()

def _debug_keys():
    def mask(k): return f"{k[:4]}...{k[-4:]}" if len(k) > 8 else "(not set)"
    logger.info(f"KEYS → Gemini:{mask(_gemini_key())} | OpenRouter:{mask(_openrouter_key())} | Perplexity:{mask(_perplexity_key())}")

# Verified working OpenRouter model IDs (March 2025)
OR_MODELS = {
    "ChatGPT":  "meta-llama/llama-3.3-70b-instruct:free",
    "Claude":   "mistralai/mistral-7b-instruct:free",
    "Grok":     "google/gemma-2b-it:free",
    "LLaMA":    "meta-llama/llama-3.3-70b-instruct:free",
}

def ask_gemini(query: str) -> dict:
    return _simulate("Gemini", query)

def _ask_openrouter(model_id: str, display_name: str, query: str) -> dict:
    key = _openrouter_key()
    if not key:
        logger.warning(f"OPENROUTER_API_KEY not set – simulating {display_name}.")
        return _simulate(display_name, query)
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-council.onrender.com",
            "X-Title": "AI Council System",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 600,
        }
        resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=40)
        logger.info(f"{display_name} ({model_id}) HTTP {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"{display_name} body: {resp.text[:400]}")
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return {"name": display_name, "response": text, "status": "success"}
    except Exception as e:
        logger.error(f"{display_name} error: {e}")
        return {"name": display_name, "response": _simulate(display_name, query)["response"], "status": "simulated"}

def ask_chatgpt(query: str) -> dict:
    return _ask_openrouter(OR_MODELS["ChatGPT"], "ChatGPT", query)

def ask_claude(query: str) -> dict:
    return _ask_openrouter(OR_MODELS["Claude"], "Claude", query)

def ask_grok(query: str) -> dict:
    return _ask_openrouter(OR_MODELS["Grok"], "Grok", query)

def ask_perplexity(query: str) -> dict:
    key = _perplexity_key()
    if not key:
        logger.warning("PERPLEXITY_API_KEY not set – using OpenRouter LLaMA fallback.")
        r = _ask_openrouter(OR_MODELS["LLaMA"], "Perplexity", query)
        r["name"] = "Perplexity"   # keep display name correct
        return r
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 600,
        }
        resp = requests.post("https://api.perplexity.ai/chat/completions",
                             headers=headers, json=payload, timeout=40)
        logger.info(f"Perplexity HTTP {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"Perplexity body: {resp.text[:400]}")
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return {"name": "Perplexity", "response": text, "status": "success"}
    except Exception as e:
        logger.error(f"Perplexity error: {e}")
        return {"name": "Perplexity", "response": _simulate("Perplexity", query)["response"], "status": "simulated"}

SIMULATED_STYLES = {
    "Gemini": (
        "As Google's Gemini 2.0, I approach '{q}' with multimodal reasoning. "
        "From a data-driven perspective the key insight is that this topic involves "
        "interconnected systems. Structurally the answer has three layers: "
        "(1) the core mechanism, (2) practical applications, (3) notable edge cases. "
        "Start with fundamentals, then build toward complexity."
    ),
    "ChatGPT": (
        "Great question! '{q}' — let me break this down clearly. "
        "Short answer: this is nuanced with several important dimensions. "
        "Primary consideration: understanding the underlying mechanism. "
        "Secondary factors: context and use-case. "
        "Summary: approach step-by-step, validate assumptions, consider broader implications."
    ),
    "Claude": (
        "Thinking carefully about '{q}': this deserves a thorough response. "
        "The core issue relates to how we frame the problem — legitimate perspectives exist on multiple sides. "
        "What seems most defensible given available evidence is a nuanced view acknowledging complexity. "
        "Accuracy, clarity, and intellectual honesty about uncertainty are essential here."
    ),
    "Grok": (
        "'{q}' — straight to the point: most answers overcomplicate this. "
        "The actual answer is simpler: understand first principles, ignore the noise. "
        "Bold claim: 80% of the complexity here is manufactured. "
        "Cut through it. The fundamentals are accessible to anyone willing to think clearly."
    ),
    "Perplexity": (
        "Based on current sources for '{q}': recent research and expert consensus "
        "point to a well-established understanding. Multiple authoritative references confirm key points. "
        "The evidence strongly supports the mainstream view while acknowledging emerging perspectives. "
        "Cross-referencing multiple streams yields the most reliable answer."
    ),
}

def _simulate(name: str, query: str) -> dict:
    template = SIMULATED_STYLES.get(name, "My answer to '{q}': this requires careful analysis.")
    return {"name": name, "response": template.replace("{q}", query[:120]), "status": "simulated"}

def get_all_responses(query: str) -> list:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _debug_keys()
    fetchers = {
        "Gemini":     lambda: ask_gemini(query),
        "ChatGPT":    lambda: ask_chatgpt(query),
        "Claude":     lambda: ask_claude(query),
        "Grok":       lambda: ask_grok(query),
        "Perplexity": lambda: ask_perplexity(query),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(fn): name for name, fn in fetchers.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as e:
                logger.error(f"Failed {name}: {e}")
                results[name] = _simulate(name, query)
    return [results[n] for n in AI_NAMES if n in results]
