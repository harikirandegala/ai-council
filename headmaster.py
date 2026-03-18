"""
headmaster.py - FIXED VERSION
Uses Gemini 2.0-flash or OpenRouter GPT-3.5 to synthesize the final verdict.
"""
import os
import json
import logging
import requests
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

def _gemini_key():     return os.environ.get("GEMINI_API_KEY", "").strip()
def _openrouter_key(): return os.environ.get("OPENROUTER_API_KEY", "").strip()

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

HEADMASTER_PROMPT = """You are the HEAD MASTER AI — a supreme evaluator overseeing a council of 5 AI systems.

The user asked: "{query}"

You have received responses and peer evaluations from: Gemini, ChatGPT, Claude, Grok, and Perplexity.

=== AI RESPONSES ===
{responses_block}

=== PEER EVALUATION SCORES (average out of 10) ===
{scores_block}

Your task:
1. Identify the BEST answer based on accuracy, clarity, completeness, and peer scores.
2. Write a synthesized final answer that is better than any single response.
3. Declare a WINNER (the AI whose answer was closest to optimal).

Reply ONLY with valid JSON — no markdown fences, no extra text:
{{
  "winner": "<AI Name>",
  "winner_reason": "<1-2 sentence reason>",
  "final_answer": "<Synthesized comprehensive answer — minimum 3 paragraphs>",
  "confidence": <1-10>,
  "key_insights": ["<insight 1>", "<insight 2>", "<insight 3>"]
}}
"""

def _compute_average_scores(evaluations: list) -> dict:
    totals = defaultdict(list)
    for block in evaluations:
        for entry in block.get("scores", []):
            ai = entry.get("ai_name")
            score = entry.get("score", 0)
            if ai:
                totals[ai].append(score)
    return {ai: round(sum(scores) / len(scores), 2) for ai, scores in totals.items() if scores}

def _build_responses_block(responses: list) -> str:
    parts = []
    for r in responses:
        short = r["response"][:450].replace('"', "'")
        parts.append(f'[{r["name"]}]:\n"{short}"')
    return "\n\n".join(parts)

def _build_scores_block(avg_scores: dict) -> str:
    lines = [f"  {ai}: {score}/10" for ai, score in sorted(avg_scores.items(), key=lambda x: -x[1])]
    return "\n".join(lines)

def _call_gemini(prompt: str) -> str:
    key = _gemini_key()
    url = f"{GEMINI_BASE}?key={key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 900}
    }
    resp = requests.post(url, json=payload, timeout=45)
    logger.info(f"HeadMaster Gemini HTTP {resp.status_code}")
    if resp.status_code != 200:
        logger.error(f"HeadMaster Gemini body: {resp.text[:400]}")
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

def _call_openrouter(prompt: str) -> str:
    key = _openrouter_key()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-council.onrender.com",
        "X-Title": "AI Council System",
    }
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 900,
    }
    resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=45)
    logger.info(f"HeadMaster OpenRouter HTTP {resp.status_code}")
    if resp.status_code != 200:
        logger.error(f"HeadMaster OpenRouter body: {resp.text[:400]}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _fallback_decision(responses: list, avg_scores: dict) -> dict:
    winner = max(avg_scores, key=avg_scores.get) if avg_scores else (responses[0]["name"] if responses else "Unknown")
    winner_resp = next((r["response"] for r in responses if r["name"] == winner), "")
    return {
        "winner": winner,
        "winner_reason": f"{winner} received the highest average peer score and demonstrated the most comprehensive coverage of the topic.",
        "final_answer": (
            f"Based on the council's evaluation, {winner}'s response was selected as the best answer.\n\n"
            + winner_resp
            + "\n\n[Note: This synthesis used rule-based scoring. Add API keys for AI-powered synthesis.]"
        ),
        "confidence": int(round(avg_scores.get(winner, 7))),
        "key_insights": [
            "Peer evaluation identified the most accurate and complete response",
            "Clarity and structure were key differentiators between responses",
            "The winning response best addressed the core question"
        ]
    }

def determine_best_answer(responses: list, evaluations: list, query: str = "") -> dict:
    avg_scores = _compute_average_scores(evaluations)
    prompt = HEADMASTER_PROMPT.format(
        query=query[:200] if query else "[see responses below]",
        responses_block=_build_responses_block(responses),
        scores_block=_build_scores_block(avg_scores)
    )
    try:
        if _gemini_key():
            raw = _call_gemini(prompt)
        elif _openrouter_key():
            raw = _call_openrouter(prompt)
        else:
            raise ValueError("No API keys configured")

        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)
        result["avg_scores"] = avg_scores
        return result
    except Exception as e:
        logger.error(f"Head Master failed ({e}) – using fallback.")
        result = _fallback_decision(responses, avg_scores)
        result["avg_scores"] = avg_scores
        return result
