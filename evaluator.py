"""
evaluator.py - Each AI evaluates all other AI responses.
Uses the Gemini API (or OpenRouter) to perform evaluations.
Falls back to rule-based scoring if APIs unavailable.
"""
import os
import json
import logging
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Models used for evaluation (one per AI name)
EVAL_MODELS = {
    "Gemini":     ("gemini", None),
    "ChatGPT":    ("openrouter", "meta-llama/llama-3.1-8b-instruct:free"),
    "Claude":     ("openrouter", "mistralai/mistral-7b-instruct:free"),
    "Grok":       ("openrouter", "google/gemma-3-4b-it:free"),
    "Perplexity": ("openrouter", "deepseek/deepseek-r1:free"),
}

EVAL_PROMPT_TEMPLATE = """You are {evaluator_name}, acting as a judge.

The user asked: "{query}"

Below are responses from {count} different AI systems. Evaluate EACH response on:
- Accuracy (factual correctness)
- Clarity (easy to understand)
- Completeness (covers the topic well)

For each AI, give a score from 1-10 and a brief 1-sentence reason.

Responses to evaluate:
{responses_block}

Reply ONLY with valid JSON in this exact format (no markdown, no extra text):
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
            short = r["response"][:400].replace('"', "'")
            lines.append(f'[{r["name"]}]: "{short}"')
    return "\n\n".join(lines)

def _call_gemini_eval(prompt: str) -> str:
    url = f"{GEMINI_BASE}?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

def _call_openrouter_eval(model_id: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-council.app",
        "X-Title": "AI Council System"
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def _parse_eval_json(raw: str, responses: list, exclude_name: str) -> list:
    """Parse JSON from model response, fallback to rule-based if needed."""
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)
        evals = data.get("evaluations", [])
        # Filter out self-evaluation
        return [e for e in evals if e.get("ai_name") != exclude_name]
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}. Using rule-based fallback.")
        return _rule_based_eval(responses, exclude_name)

def _rule_based_eval(responses: list, exclude_name: str) -> list:
    """Simple length + keyword heuristic scoring when API is unavailable."""
    import random
    result = []
    for r in responses:
        if r["name"] == exclude_name:
            continue
        text = r["response"]
        length_score = min(10, max(4, len(text) // 60))
        # Boost for structured answers
        bonus = 1 if any(k in text.lower() for k in ["because", "therefore", "first", "second", "example"]) else 0
        score = min(10, length_score + bonus + random.randint(0, 1))
        result.append({
            "ai_name": r["name"],
            "score": score,
            "feedback": f"Response is {'comprehensive' if score >= 7 else 'adequate'} with {'clear structure' if bonus else 'basic coverage'}."
        })
    return result

def evaluate_by_one_ai(evaluator_name: str, responses: list, query: str) -> dict:
    """One AI evaluates all others."""
    backend, model_id = EVAL_MODELS[evaluator_name]
    responses_block = _build_responses_block(responses, evaluator_name)
    others = [r for r in responses if r["name"] != evaluator_name]

    prompt = EVAL_PROMPT_TEMPLATE.format(
        evaluator_name=evaluator_name,
        query=query[:200],
        count=len(others),
        responses_block=responses_block
    )

    raw = None
    try:
        if backend == "gemini" and GEMINI_API_KEY:
            raw = _call_gemini_eval(prompt)
        elif backend == "openrouter" and OPENROUTER_API_KEY:
            raw = _call_openrouter_eval(model_id, prompt)
        else:
            raise ValueError("No API key available")

        evals = _parse_eval_json(raw, responses, evaluator_name)
    except Exception as e:
        logger.warning(f"{evaluator_name} evaluation failed ({e}), using rule-based.")
        evals = _rule_based_eval(responses, evaluator_name)

    return {"evaluator": evaluator_name, "scores": evals}

def evaluate_all_responses(responses: list, query: str = "") -> list:
    """All AIs evaluate all others in parallel."""
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
                logger.error(f"Evaluation by {name} failed: {e}")
                results.append({
                    "evaluator": name,
                    "scores": _rule_based_eval(responses, name)
                })

    return results
