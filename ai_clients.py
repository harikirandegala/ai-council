"""
ai_clients.py - AI Council System
Keys are read fresh on every call (fixes Render env var timing issue).
Uses updated model IDs verified working March 2025.
"""
import os
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

AI_NAMES = ["Gemini", "ChatGPT", "Claude", "Grok", "Perplexity"]

def _gkey(): return os.environ.get("GEMINI_API_KEY", "").strip()
def _okey(): return os.environ.get("OPENROUTER_API_KEY", "").strip()
def _pkey(): return os.environ.get("PERPLEXITY_API_KEY", "").strip()

def _debug():
    def m(k): return f"{k[:6]}...{k[-4:]}" if len(k)>10 else "(NOT SET)"
    logger.info(f"KEYS → Gemini:{m(_gkey())} | OpenRouter:{m(_okey())} | Perplexity:{m(_pkey())}")

def ask_gemini(query):
    key = _gkey()
    if not key:
        logger.warning("Gemini key missing – simulating.")
        return _sim("Gemini", query)
    try:
        r = requests.post(
            f"{GEMINI_BASE}?key={key}",
            json={"contents":[{"parts":[{"text":query}]}],"generationConfig":{"maxOutputTokens":600}},
            timeout=35
        )
        logger.info(f"Gemini HTTP {r.status_code}")
        if r.status_code != 200: logger.error(f"Gemini: {r.text[:300]}")
        r.raise_for_status()
        return {"name":"Gemini","response":r.json()["candidates"][0]["content"]["parts"][0]["text"],"status":"success"}
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return {"name":"Gemini","response":_sim("Gemini",query)["response"],"status":"simulated"}

def _openrouter(model, name, query):
    key = _okey()
    if not key:
        logger.warning(f"OpenRouter key missing – simulating {name}.")
        return _sim(name, query)
    try:
        r = requests.post(
            OPENROUTER_BASE,
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json",
                     "HTTP-Referer":"https://ai-council.onrender.com","X-Title":"AI Council System"},
            json={"model":model,"messages":[{"role":"user","content":query}],"max_tokens":600},
            timeout=40
        )
        logger.info(f"{name} ({model}) HTTP {r.status_code}")
        if r.status_code != 200: logger.error(f"{name}: {r.text[:300]}")
        r.raise_for_status()
        return {"name":name,"response":r.json()["choices"][0]["message"]["content"],"status":"success"}
    except Exception as e:
        logger.error(f"{name} error: {e}")
        return {"name":name,"response":_sim(name,query)["response"],"status":"simulated"}

def ask_chatgpt(q): return _openrouter("openai/gpt-3.5-turbo", "ChatGPT", q)
def ask_claude(q):  return _openrouter("anthropic/claude-3-haiku:beta", "Claude", q)
def ask_grok(q):    return _openrouter("x-ai/grok-3-mini-beta", "Grok", q)

def ask_perplexity(query):
    key = _pkey()
    if not key:
        logger.warning("Perplexity key missing – using free LLaMA fallback.")
        r = _openrouter("meta-llama/llama-3.3-70b-instruct:free", "Perplexity", query)
        r["name"] = "Perplexity"
        return r
    try:
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"model":"sonar","messages":[{"role":"user","content":query}],"max_tokens":600},
            timeout=40
        )
        logger.info(f"Perplexity HTTP {r.status_code}")
        r.raise_for_status()
        return {"name":"Perplexity","response":r.json()["choices"][0]["message"]["content"],"status":"success"}
    except Exception as e:
        logger.error(f"Perplexity error: {e}")
        return {"name":"Perplexity","response":_sim("Perplexity",query)["response"],"status":"simulated"}

_STYLES = {
    "Gemini":     "As Google's Gemini 2.0, I approach '{q}' with multimodal reasoning. From a data-driven perspective the key insight is that this topic involves interconnected systems. The answer has three layers: (1) core mechanism, (2) practical applications, (3) edge cases. Start with fundamentals, build toward complexity.",
    "ChatGPT":    "Great question! '{q}' — let me break this down. Short answer: this is nuanced with several important dimensions. Primary: understanding the underlying mechanism. Secondary: context and use-case. Summary: approach step-by-step, validate assumptions, consider broader implications.",
    "Claude":     "Thinking carefully about '{q}': this deserves a thorough response. The core issue relates to how we frame the problem — legitimate perspectives exist on multiple sides. What seems most defensible given available evidence is a nuanced view acknowledging complexity. Accuracy and intellectual honesty are essential.",
    "Grok":       "'{q}' — straight to the point: most answers overcomplicate this. The actual answer is simpler: understand first principles, ignore the noise. 80% of the complexity here is manufactured. Cut through it. The fundamentals are accessible to anyone willing to think clearly.",
    "Perplexity": "Based on current sources for '{q}': research and expert consensus point to a well-established understanding. Multiple authoritative references confirm key points. The evidence supports the mainstream view while acknowledging emerging perspectives. Cross-referencing yields the most reliable answer.",
}

def _sim(name, query):
    t = _STYLES.get(name, "My answer to '{q}': this requires careful analysis.")
    return {"name":name,"response":t.replace("{q}",query[:120]),"status":"simulated"}

def get_all_responses(query):
    _debug()
    fetchers = {
        "Gemini":     lambda: ask_gemini(query),
        "ChatGPT":    lambda: ask_chatgpt(query),
        "Claude":     lambda: ask_claude(query),
        "Grok":       lambda: ask_grok(query),
        "Perplexity": lambda: ask_perplexity(query),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        fm = {ex.submit(fn): name for name, fn in fetchers.items()}
        for future in as_completed(fm):
            name = fm[future]
            try: results[name] = future.result()
            except Exception as e:
                logger.error(f"Failed {name}: {e}")
                results[name] = _sim(name, query)
    return [results[n] for n in AI_NAMES if n in results]
