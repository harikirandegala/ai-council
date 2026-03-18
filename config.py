import os

GEMINI_API_KEY     = "AIzaSyD5v9ya8PPPiWoG-pEDBzXSNcRqLGUdIWQ"
OPENROUTER_API_KEY = "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"
PERPLEXITY_API_KEY = ""

def get_gemini_key():
    return os.environ.get("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY

def get_openrouter_key():
    return os.environ.get("OPENROUTER_API_KEY", "").strip() or OPENROUTER_API_KEY

def get_perplexity_key():
    return os.environ.get("PERPLEXITY_API_KEY", "").strip() or PERPLEXITY_API_KEY
