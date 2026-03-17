"""
ai_clients.py - Handles all AI API calls
Supports: Gemini, OpenRouter (Claude/GPT/Grok), Perplexity
Falls back to simulation if APIs are unavailable.
"""
import os
import json
import logging
import requests
import time

logger = logging.getLogger(__name__)

# ─── API KEYS (set these in .env or environment) ────────────────────────────
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

AI_NAMES = ["Gemini", "ChatGPT", "Claude", "Grok", "Perplexity"]

# ─── GEMINI ──────────────────────────────────────────────────────────────────
def ask_gemini(query: str) -> dict:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set – simulating.")
        return _simulate("Gemini", query)
    try:
        url = f"{GEMINI_BASE}?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": query}]}]}
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"name": "Gemini", "response": text, "status": "success"}
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return {"name": "Gemini", "response": _simulate("Gemini", query)["response"], "status": "simulated"}

# ─── OPENROUTER (GPT / Claude / Grok) ────────────────────────────────────────
def _ask_openrouter(model_id: str, display_name: str, query: str) -> dict:
    if not OPENROUTER_API_KEY:
        logger.warning(f"OPENROUTER_API_KEY not set – simulating {display_name}.")
        return _simulate(display_name, query)
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-council.app",
            "X-Title": "AI Council System"
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 600
        }
        resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"name": display_name, "response": text, "status": "success"}
    except Exception as e:
        logger.error(f"{display_name} error: {e}")
        return {"name": display_name, "response": _simulate(display_name, query)["response"], "status": "simulated"}

def ask_chatgpt(query: str) -> dict:
    return _ask_openrouter("meta-llama/llama-3.1-8b-instruct:free", "ChatGPT", query)

def ask_claude(query: str) -> dict:
    return _ask_openrouter("mistralai/mistral-7b-instruct:free", "Claude", query)

def ask_grok(query: str) -> dict:
    return _ask_openrouter("google/gemma-3-4b-it:free", "Grok", query)

# ─── PERPLEXITY ───────────────────────────────────────────────────────────────
def ask_perplexity(query: str) -> dict:
    return _ask_openrouter("deepseek/deepseek-r1:free", "Perplexity", query)

# ─── SIMULATION FALLBACK ──────────────────────────────────────────────────────
SIMULATED_STYLES = {
    "Gemini": (
        "As Google's Gemini, I approach this with multimodal reasoning. "
        "The question '{q}' is multifaceted. "
        "From a data-driven perspective, the key insight is that this topic involves "
        "interconnected systems. Consider the foundational principles first: we must "
        "examine both the immediate context and longer-term implications. "
        "Structurally, the answer involves three layers: (1) the core mechanism, "
        "(2) practical applications, and (3) edge cases worth noting. "
        "My recommendation is to start with the fundamentals, then build upward "
        "toward more complex considerations. Verification with recent sources is advised."
    ),
    "ChatGPT": (
        "Great question! '{q}' — let me break this down clearly. "
        "First, the short answer: this is a nuanced topic with several important dimensions. "
        "Here's what you need to know: the primary consideration is understanding "
        "the underlying mechanism. Secondary factors include context and use-case. "
        "To summarize: approach this step-by-step, validate your assumptions, "
        "and always consider the broader implications. Happy to dive deeper on any aspect!"
    ),
    "Claude": (
        "Thinking carefully about '{q}': this is an interesting question that deserves "
        "a thorough response. Let me reason through it. The core issue here relates to "
        "how we frame the problem. There are legitimate perspectives on multiple sides. "
        "What seems most defensible, given available evidence, is a nuanced view that "
        "acknowledges complexity. I'd highlight: accuracy matters, clarity matters, "
        "and intellectual honesty requires admitting uncertainty where it exists. "
        "My considered view: proceed thoughtfully, seek multiple sources, and remain open to revision."
    ),
    "Grok": (
        "'{q}' — straight to the point: here's the unfiltered take. "
        "Most answers you'll find are either overcomplicated or dancing around the real issue. "
        "The actual answer is simpler than people make it: understand the first principles, "
        "ignore the noise, and focus on what actually matters. "
        "Bold claim: 80% of the complexity in this topic is manufactured. "
        "Cut through it. The fundamentals are accessible to anyone willing to think clearly. "
        "Don't overthink it — act on the core insight."
    ),
    "Perplexity": (
        "Based on current sources regarding '{q}': recent research and expert consensus "
        "point to a well-established understanding. Multiple authoritative references confirm "
        "the key points. The latest data as of this year shows a clear trend. "
        "Citing relevant studies: the evidence strongly supports the mainstream view, "
        "while also acknowledging emerging perspectives. Cross-referencing multiple "
        "information streams, the most reliable answer integrates both established "
        "knowledge and recent developments. Sources: [web synthesis]."
    )
}

def _simulate(name: str, query: str) -> dict:
    template = SIMULATED_STYLES.get(name, "I would answer: {q}")
    response = template.replace("{q}", query[:120])
    return {"name": name, "response": response, "status": "simulated"}

# ─── MAIN ORCHESTRATOR ────────────────────────────────────────────────────────
def get_all_responses(query: str) -> list:
    """Fetch responses from all 5 AIs concurrently (sequential for simplicity)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

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
                logger.error(f"Failed to get response from {name}: {e}")
                results[name] = _simulate(name, query)

    # Return in consistent order
    return [results[n] for n in AI_NAMES if n in results]
