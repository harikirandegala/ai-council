"""
headmaster.py - The Head Master AI analyzes all responses and evaluations,
then selects and synthesizes the best final answer.
"""
import os
import json
import logging
import requests
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

HEADMASTER_PROMPT = """You are the HEAD MASTER AI — a supreme evaluator overseeing a council of 5 AI systems.

The user asked: "{query}"

You have received responses and peer evaluations from: Gemini, ChatGPT, Claude, Grok, and Perplexity.

=== AI RESPONSES ===
{responses_block}

=== PEER EVALUATION SCORES (average out of 10) ===
{scores_block}

Your task:
1. Identify the BEST answer based on: accuracy, clarity, completeness, and peer scores.
2. Write a synthesized final answer that is better than any single response.
3. Declare a WINNER (the AI whose answer was closest to optimal).

Reply ONLY with valid JSON (no markdown, no extra text):
{{
  "winner": "<AI Name>",
  "winner_reason": "<1-2 sentence reason why this AI won>",
  "final_answer": "<Your synthesized, comprehensive final answer — at least 3 paragraphs>",
  "confidence": <1-10>,
  "key_insights": ["<insight 1>", "<insight 2>", "<insight 3>"]
}}
"""

def _compute_average_scores(evaluations: list) -> dict:
    """Compute average score received by each AI across all evaluators."""
    totals = defaultdict(list)
    for eval_block in evaluations:
        for score_entry in eval_block.get("scores", []):
            ai = score_entry.get("ai_name")
            score = score_entry.get("score", 0)
            if ai:
                totals[ai].append(score)

    averages = {}
    for ai, scores in totals.items():
        averages[ai] = round(sum(scores) / len(scores), 2) if scores else 0
    return averages

def _build_responses_block(responses: list) -> str:
    parts = []
    for r in responses:
        short = r["response"][:500].replace('"', "'")
        parts.append(f'[{r["name"]}]:\n"{short}"')
    return "\n\n".join(parts)

def _build_scores_block(avg_scores: dict) -> str:
    lines = []
    for ai, score in sorted(avg_scores.items(), key=lambda x: -x[1]):
        lines.append(f"  {ai}: {score}/10")
    return "\n".join(lines)

def _call_gemini(prompt: str) -> str:
    url = f"{GEMINI_BASE}?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=payload, timeout=40)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

def _call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-council.app",
        "X-Title": "AI Council System"
    }
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 900
    }
    resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=40)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def _fallback_decision(responses: list, avg_scores: dict) -> dict:
    """Rule-based fallback when no API is available."""
    if avg_scores:
        winner = max(avg_scores, key=avg_scores.get)
    else:
        winner = responses[0]["name"] if responses else "Unknown"

    winner_resp = next((r["response"] for r in responses if r["name"] == winner), "")

    return {
        "winner": winner,
        "winner_reason": f"{winner} received the highest average score from peer evaluations and demonstrated the most comprehensive coverage.",
        "final_answer": (
            f"Based on the council evaluation, {winner}'s response was selected as the best answer.\n\n"
            + winner_resp
            + "\n\n[Note: This selection was made using rule-based scoring. Connect API keys for AI-powered synthesis.]"
        ),
        "confidence": round(avg_scores.get(winner, 7)),
        "key_insights": [
            "Peer evaluation identified the most accurate response",
            "Clarity and completeness were key differentiators",
            "The winning response best addressed the core question"
        ]
    }

def determine_best_answer(responses: list, evaluations: list, query: str = "") -> dict:
    """Head Master analyzes everything and returns final verdict."""
    avg_scores = _compute_average_scores(evaluations)

    prompt = HEADMASTER_PROMPT.format(
        query=query[:200] if query else "[see responses below]",
        responses_block=_build_responses_block(responses),
        scores_block=_build_scores_block(avg_scores)
    )

    raw = None
    try:
        if GEMINI_API_KEY:
            raw = _call_gemini(prompt)
        elif OPENROUTER_API_KEY:
            raw = _call_openrouter(prompt)
        else:
            raise ValueError("No API keys configured")

        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)
        result["avg_scores"] = avg_scores
        return result

    except Exception as e:
        logger.error(f"Head Master API failed: {e}. Using fallback.")
        result = _fallback_decision(responses, avg_scores)
        result["avg_scores"] = avg_scores
        return result
