"""
evaluator.py - AI Council System
Peer evaluation. Keys read fresh each call.
"""
import os, json, logging, requests, re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def _gkey(): return os.environ.get("GEMINI_API_KEY","").strip()
def _okey(): return os.environ.get("OPENROUTER_API_KEY","").strip()

EVAL_MODELS = {
    "Gemini":     ("gemini",     None),
    "ChatGPT":    ("openrouter", "openai/gpt-3.5-turbo"),
    "Claude":     ("openrouter", "anthropic/claude-3-haiku:beta"),
    "Grok":       ("openrouter", "x-ai/grok-3-mini-beta"),
    "Perplexity": ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
}

PROMPT = """You are {name}, acting as an impartial judge.
User asked: "{query}"
Rate each of the {count} AI responses below. Score each 1-10 on accuracy, clarity, completeness.
{block}
Reply ONLY with valid JSON, no markdown:
{{"evaluations":[{{"ai_name":"...","score":<1-10>,"feedback":"..."}}]}}"""

def _block(responses, exclude):
    return "\n\n".join(
        f'[{r["name"]}]: "{r["response"][:300].replace(chr(34),chr(39))}"' 
        for r in responses if r["name"] != exclude
    )

def _rule_based(responses, exclude):
    import random
    out = []
    for r in responses:
        if r["name"] == exclude: continue
        s = min(10, max(4, len(r["response"])//60) + (1 if any(k in r["response"].lower() for k in ["because","first","example","key"]) else 0) + random.randint(0,1))
        out.append({"ai_name":r["name"],"score":s,"feedback":f'Response is {"comprehensive" if s>=7 else "adequate"}.' })
    return out

def evaluate_by_one_ai(evaluator_name, responses, query):
    backend, model_id = EVAL_MODELS[evaluator_name]
    others = [r for r in responses if r["name"] != evaluator_name]
    prompt = PROMPT.format(name=evaluator_name, query=query[:200], count=len(others), block=_block(responses, evaluator_name))
    try:
        gkey, okey = _gkey(), _okey()
        if backend == "gemini" and gkey:
            r = requests.post(f"{GEMINI_BASE}?key={gkey}",
                json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"maxOutputTokens":500}}, timeout=35)
            logger.info(f"Eval Gemini HTTP {r.status_code}")
            r.raise_for_status()
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif okey:
            r = requests.post(OPENROUTER_BASE,
                headers={"Authorization":f"Bearer {okey}","Content-Type":"application/json",
                         "HTTP-Referer":"https://ai-council.onrender.com","X-Title":"AI Council System"},
                json={"model": model_id or "openai/gpt-3.5-turbo",
                      "messages":[{"role":"user","content":prompt}],"max_tokens":500}, timeout=40)
            logger.info(f"Eval {evaluator_name} ({model_id}) HTTP {r.status_code}")
            if r.status_code != 200: logger.error(f"Eval {evaluator_name}: {r.text[:200]}")
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError("No API key")
        clean = re.sub(r"```json|```","",raw).strip()
        evals = json.loads(clean).get("evaluations",[])
        return {"evaluator":evaluator_name,"scores":[e for e in evals if e.get("ai_name")!=evaluator_name]}
    except Exception as e:
        logger.warning(f"{evaluator_name} eval failed ({e}), rule-based fallback.")
        return {"evaluator":evaluator_name,"scores":_rule_based(responses,evaluator_name)}

def evaluate_all_responses(responses, query=""):
    results = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        fm = {ex.submit(evaluate_by_one_ai, r["name"], responses, query): r["name"] for r in responses}
        for future in as_completed(fm):
            name = fm[future]
            try: results.append(future.result())
            except Exception as e:
                logger.error(f"Eval crashed {name}: {e}")
                results.append({"evaluator":name,"scores":_rule_based(responses,name)})
    return results
