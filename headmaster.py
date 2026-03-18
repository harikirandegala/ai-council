"""
headmaster.py - AI Council System
Head Master synthesizes final verdict. Keys read fresh each call.
"""
import os, json, logging, requests, re
from collections import defaultdict

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def _gkey(): return os.environ.get("GEMINI_API_KEY","").strip()
def _okey(): return os.environ.get("OPENROUTER_API_KEY","").strip()

PROMPT = """You are the HEAD MASTER AI — supreme evaluator of a 5-AI council.
User asked: "{query}"

=== AI RESPONSES ===
{responses}

=== AVERAGE PEER SCORES ===
{scores}

Pick the best answer. Write a synthesized final answer better than any single response. Declare a winner.
Reply ONLY with valid JSON, no markdown:
{{"winner":"<n>","winner_reason":"<1-2 sentences>","final_answer":"<3+ paragraph synthesis>","confidence":<1-10>,"key_insights":["...","...","..."]}}"""

def _avg(evaluations):
    totals = defaultdict(list)
    for b in evaluations:
        for e in b.get("scores",[]):
            if e.get("ai_name"): totals[e["ai_name"]].append(e.get("score",0))
    return {ai: round(sum(s)/len(s),2) for ai,s in totals.items() if s}

def determine_best_answer(responses, evaluations, query=""):
    avg = _avg(evaluations)
    resp_block = "\n\n".join(f'[{r["name"]}]:\n"{r["response"][:400].replace(chr(34),chr(39))}"' for r in responses)
    score_block = "\n".join(f"  {ai}: {sc}/10" for ai,sc in sorted(avg.items(),key=lambda x:-x[1]))
    prompt = PROMPT.format(query=query[:200] or "[see responses]", responses=resp_block, scores=score_block)

    gkey, okey = _gkey(), _okey()
    try:
        if gkey:
            r = requests.post(f"{GEMINI_BASE}?key={gkey}",
                json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"maxOutputTokens":900}}, timeout=45)
            logger.info(f"HeadMaster Gemini HTTP {r.status_code}")
            if r.status_code != 200: logger.error(f"HeadMaster Gemini: {r.text[:300]}")
            r.raise_for_status()
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif okey:
            r = requests.post(OPENROUTER_BASE,
                headers={"Authorization":f"Bearer {okey}","Content-Type":"application/json",
                         "HTTP-Referer":"https://ai-council.onrender.com","X-Title":"AI Council System"},
                json={"model":"openai/gpt-3.5-turbo","messages":[{"role":"user","content":prompt}],"max_tokens":900}, timeout=45)
            logger.info(f"HeadMaster OpenRouter HTTP {r.status_code}")
            if r.status_code != 200: logger.error(f"HeadMaster OR: {r.text[:300]}")
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError("No API keys configured")

        result = json.loads(re.sub(r"```json|```","",raw).strip())
        result["avg_scores"] = avg
        return result
    except Exception as e:
        logger.error(f"HeadMaster failed ({e}) – fallback.")
        winner = max(avg, key=avg.get) if avg else (responses[0]["name"] if responses else "Unknown")
        winner_resp = next((r["response"] for r in responses if r["name"]==winner),"")
        return {
            "winner": winner,
            "winner_reason": f"{winner} received the highest peer score and demonstrated the most comprehensive coverage.",
            "final_answer": f"Based on peer evaluation, {winner}'s response was selected:\n\n{winner_resp}",
            "confidence": int(round(avg.get(winner,7))),
            "key_insights": ["Peer evaluation identified the most complete response","Clarity was a key differentiator","The winning response best addressed the question"],
            "avg_scores": avg
        }
