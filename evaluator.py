"""
evaluator.py - FIXED VERSION
Each AI evaluates all others. Uses updated model IDs.
Falls back to rule-based scoring if APIs fail.
"""
import os
import json
import logging
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

def _gemini_key():     return os.environ.get("GEMINI_API_KEY", "").strip()
def _openrouter_key(): return os.environ.get("OPENROUTER_API_KEY", "").strip()

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# One model per evaluator role (free tier verified March 2025)
EVAL_MODELS = {
    "Gemini":     ("gemini",     None),
    "ChatGPT":    ("openrouter", "openai/gpt-3.5-turbo"),
    "Claude":     ("openrouter", "anthropic/claude-3-haiku:beta"),
    "Grok":       ("openrouter", "x-ai/grok-3-mini-beta"),
    "Perplexity": ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
}

EVAL_PROMPT_TEMPLATE = """You are {evaluator_name}, acting as an impartial judge.

The user asked: "{query}"

Below are responses from {count} different AI systems. Evaluate EACH one on:
- Accuracy (factual correctness)
- Clarity (easy to understand)
- Completeness (covers the topic well)

Give a score 1-10 and a brief 1-sentence reason for each.

Responses:
{responses_block}

Reply ONLY with valid JSON — no markdown fences, no preamble:
{{
  "evaluations": [
    {{"ai_name": "...", "score": <1-10>, "feedback": "..."}},
    ...
  ]
}}
"""

def _build_responses_block(responses: list, exclude_name: str) -> str:
    lines = []
    for r in responses:
        if r["name"] != exclude_name:
            short = r["response"][:350].replace('"', "'")
            lines.append(f'[{r["name"]}]: "{short}"')
    return "\n\n".join(lines)

def _call_gemini_eval(prompt: str) -> str:
    key = _gemini_key()
    url = f"{GEMINI_BASE}?key={key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 500}
    }
    resp = requests.post(url, json=payload, timeout=35)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

def _call_openrouter_eval(model_id: str, prompt: str) -> str:
    key = _openrouter_key()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-council.onrender.com",
        "X-Title": "AI Council System",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
    }
    resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=40)
    logger.info(f"Eval ({model_id}) HTTP {resp.status_code}")
    if resp.status_code != 200:
        logger.error(f"Eval body: {resp.text[:300]}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _parse_eval_json(raw: str, responses: list, exclude_name: str) -> list:
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)
        evals = data.get("evaluations", [])
        return [e for e in evals if e.get("ai_name") != exclude_name]
    except Exception as e:
        logger.warning(f"JSON parse failed ({e}) – rule-based fallback.")
        return _rule_based_eval(responses, exclude_name)

def _rule_based_eval(responses: list, exclude_name: str) -> list:
    import random
    result = []
    for r in responses:
        if r["name"] == exclude_name:
            continue
        text = r["response"]
        length_score = min(10, max(4, len(text) // 60))
        bonus = 1 if any(k in text.lower() for k in ["because", "therefore", "first", "second", "example", "key"]) else 0
        score = min(10, length_score + bonus + random.randint(0, 1))
        result.append({
            "ai_name": r["name"],
            "score": score,
            "feedback": f"Response is {'comprehensive' if score >= 7 else 'adequate'} with {'clear structure' if bonus else 'basic coverage'}."
        })
    return result

def evaluate_by_one_ai(evaluator_name: str, responses: list, query: str) -> dict:
    backend, model_id = EVAL_MODELS[evaluator_name]
    others = [r for r in responses if r["name"] != evaluator_name]
    prompt = EVAL_PROMPT_TEMPLATE.format(
        evaluator_name=evaluator_name,
        query=query[:200],
        count=len(others),
        responses_block=_build_responses_block(responses, evaluator_name)
    )
    raw = None
    try:
        if backend == "gemini" and _gemini_key():
            raw = _call_gemini_eval(prompt)
        elif backend == "openrouter" and _openrouter_key():
            raw = _call_openrouter_eval(model_id, prompt)
        else:
            raise ValueError("No API key available for evaluation")
        evals = _parse_eval_json(raw, responses, evaluator_name)
    except Exception as e:
        logger.warning(f"{evaluator_name} eval failed ({e}), rule-based fallback.")
        evals = _rule_based_eval(responses, evaluator_name)
    return {"evaluator": evaluator_name, "scores": evals}

def evaluate_all_responses(responses: list, query: str = "") -> list:
    ai_names = [r["name"] for r in responses]
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(evaluate_by_one_ai, name, responses, query): name
            for name in ai_names
        }
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Eval by {name} crashed: {e}")
                results.append({"evaluator": name, "scores": _rule_based_eval(responses, name)})
    return results
