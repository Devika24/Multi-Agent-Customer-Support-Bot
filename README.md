# Multi-Agent Customer Support Bot (CrewAI + Ollama)

This repo contains a deployable **multi-agent customer support assistant** built with:

- **CrewAI** for multi-agent orchestration
- **Ollama** with `llama3` as the local LLM
- **FastAPI** for the HTTP API

## 🚀 Setup

```bash
# In this folder
pip install -r requirements.txt

# Install & run Ollama separately (if not already)
# See https://ollama.ai for install instructions
ollama pull llama3
ollama serve
```

## ▶️ Run the API

```bash
uvicorn app:app --reload --port 8000
```

Then open:

- Interactive docs: `http://localhost:8000/docs`

Example request:

```bash
curl -X POST "http://localhost:8000/ask-support" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

You will get a JSON response like:

```json
{
  "reply": "...",
  "escalate": false,
  "reason": null
}
```

Extend `data/faq_data.json` with your own FAQs to adapt this bot to any product or company.

