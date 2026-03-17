# 🏛️ AI Council System

> Five AI minds deliberate. One Head Master decides. Wisdom emerges.

A full-stack system where **Gemini, ChatGPT, Claude, Grok, and Perplexity** all answer the same question, evaluate each other's responses, and a **Head Master AI** synthesizes the best final answer.

---

## ✨ Features

- 🤖 **5 AI Responders** — Gemini, ChatGPT, Claude, Grok, Perplexity
- ⚖️ **Peer Evaluation** — Every AI scores every other AI (accuracy, clarity, completeness)
- 👑 **Head Master AI** — Synthesizes all answers + scores into one superior final answer
- 📊 **Score Visualization** — Average scores, bar charts, evaluation table
- 🔄 **Graceful Fallback** — Works even with NO API keys (smart simulation mode)
- ⚡ **Parallel Execution** — All API calls run concurrently for speed

---

## 📁 Project Structure

```
ai_council/
├── app.py              # Flask server + routes
├── ai_clients.py       # All AI API calls (Gemini, OpenRouter, Perplexity)
├── evaluator.py        # Peer evaluation logic
├── headmaster.py       # Final decision engine
├── run.py              # Dev startup (loads .env)
├── requirements.txt    # Python dependencies
├── Procfile            # For Render/Railway deployment
├── .env.example        # Environment variable template
└── templates/
    └── index.html      # Frontend (single file, no build step)
```

---

## 🚀 Local Setup (5 minutes)

### 1. Prerequisites
- Python 3.9+
- pip

### 2. Clone & Install
```bash
git clone <your-repo-url> ai_council
cd ai_council
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
cp .env.example .env
# Edit .env with your keys (see API Keys section below)
```

### 4. Run
```bash
python run.py
# Open http://localhost:5000
```

> **No API keys?** No problem! The system runs in simulation mode automatically.

---

## 🔑 API Keys Setup

### A) Gemini (Google) — FREE
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy into `.env` as `GEMINI_API_KEY`

### B) OpenRouter — FREE (gives GPT + Claude + Grok access)
1. Go to https://openrouter.ai/keys
2. Create account
3. Generate API key
4. Copy into `.env` as `OPENROUTER_API_KEY`
5. Note: Add a small credit ($1-5) to unlock more models, or use free models only

**Free models available on OpenRouter:**
- `openai/gpt-3.5-turbo` (limited free requests)
- `anthropic/claude-3-haiku` (limited free)
- `meta-llama/llama-3.1-8b-instruct:free` (truly free)
- `x-ai/grok-beta` (limited free)

### C) Perplexity — Optional
1. Go to https://www.perplexity.ai/settings/api
2. Get API key (has free trial credits)
3. Copy into `.env` as `PERPLEXITY_API_KEY`

---

## 🌐 Deployment (Free)

### Option 1: Render.com (Recommended)
1. Push your code to GitHub
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Add Environment Variables (your API keys) in Render dashboard
6. Deploy!

### Option 2: Railway.app
1. Push to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Add environment variables in Railway dashboard
4. Railway auto-detects Python and uses `Procfile`

### Option 3: Vercel (Serverless)
Create `vercel.json`:
```json
{
  "builds": [{"src": "app.py", "use": "@vercel/python"}],
  "routes": [{"src": "/(.*)", "dest": "app.py"}]
}
```
Then: `vercel deploy`

---

## 🔧 How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Step 1: Parallel AI Responses      │
│  Gemini ─ ChatGPT ─ Claude ─ ...    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Step 2: Peer Evaluation            │
│  Each AI rates all others (1-10)    │
│  + short feedback                   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Step 3: Head Master Analysis       │
│  - Reads all responses + scores     │
│  - Picks winner                     │
│  - Synthesizes final answer         │
└─────────────────────────────────────┘
```

---

## 🐛 Debugging

Logs are printed to console with timestamps. To increase verbosity:
```python
# In app.py, change:
logging.basicConfig(level=logging.DEBUG, ...)
```

Common issues:
- **429 Too Many Requests** → You've hit the free tier limit. Wait or add credits.
- **API key invalid** → Check `.env` file and restart the server
- **Slow responses** → Normal for 5 concurrent API calls. Timeout is 30s per call.

---

## 📝 License

MIT — Free to use, modify, and deploy.
