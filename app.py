from fastapi import FastAPI
from pydantic import BaseModel
import json

from crew_setup import build_support_crew, load_faq_context


app = FastAPI(
    title="Multi-Agent Customer Support Bot",
    description="Customer support QA assistant using CrewAI + Ollama (llama3).",
    version="0.1.0",
)


class SupportRequest(BaseModel):
    query: str


class SupportResponse(BaseModel):
    reply: str
    escalate: bool
    reason: str | None = None


@app.post("/ask-support", response_model=SupportResponse)
def ask_support(req: SupportRequest):
    faq_context = load_faq_context()
    crew = build_support_crew()

    raw_result = crew.kickoff(inputs={
        "user_query": req.query,
        "faq_context": faq_context,
    })

    text = str(raw_result).strip()

    try:
        data = json.loads(text)
    except Exception:
        return SupportResponse(
            reply=text,
            escalate=False,
            reason="Model did not return valid JSON; full text used as reply.",
        )

    reply = data.get("reply", text)
    escalate = bool(data.get("escalate", False))
    reason = data.get("reason")

    return SupportResponse(reply=reply, escalate=escalate, reason=reason)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
