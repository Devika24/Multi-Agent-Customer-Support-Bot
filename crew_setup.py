import json
import pathlib
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOllama


def get_llm():
    """Return an Ollama-backed chat model (llama3)."""
    return ChatOllama(model="llama3", temperature=0.2)


def load_faq_context() -> str:
    """Load FAQ data from JSON and return as a text block."""
    data_path = pathlib.Path(__file__).parent / "data" / "faq_data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        faq_items = json.load(f)

    blocks = []
    for item in faq_items:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            blocks.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(blocks)


def build_support_crew() -> Crew:
    llm = get_llm()

    intent_agent = Agent(
        role="Intent Classifier",
        goal="Classify the intent of a customer support query.",
        backstory=(
            "You read user queries and classify them into 'faq', 'billing', "
            "'technical', or 'other'. You always output strict JSON."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    retrieval_agent = Agent(
        role="Knowledge Retrieval Specialist",
        goal="Select the most relevant FAQ entries for a given user query.",
        backstory=(
            "You are given a FAQ knowledge base as text and a user query, "
            "and you return the most relevant 1-3 Q&A pairs as JSON."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    response_agent = Agent(
        role="Support Response Agent",
        goal="Craft a clear, empathetic, and accurate response using the retrieved FAQs and intent.",
        backstory=(
            "You are a senior customer support representative. You read the user query, "
            "intent classification, and selected FAQ snippets, and generate a helpful reply."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    escalation_agent = Agent(
        role="Escalation Decision Agent",
        goal="Decide whether to escalate the ticket to a human, based on confidence and complexity.",
        backstory=(
            "You evaluate all previous steps and decide if a human should take over, "
            "based on risk, ambiguity, or low confidence."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    intent_task = Task(
        description=(
            "You are given a customer support query:\n"
            "USER_QUERY: {user_query}\n\n"
            "Classify it into one of: 'faq', 'billing', 'technical', 'other'.\n"
            "Return STRICT JSON with fields:\n"
            "{\"intent\": \"faq|billing|technical|other\", \"confidence\": 0.xx}"
        ),
        expected_output='A JSON string like {"intent": "faq", "confidence": 0.93}.',
        agent=intent_agent,
    )

    retrieval_task = Task(
        description=(
            "You are given:\n"
            "- USER_QUERY: {user_query}\n"
            "- FAQ_KNOWLEDGE (Q&A pairs):\n"
            "{faq_context}\n\n"
            "Select the 1-3 most relevant Q&A pairs.\n"
            "Return STRICT JSON as:\n"
            "{\"selected_faqs\": [\"Q: ... A: ...\", \"Q: ... A: ...\"]}"
        ),
        expected_output='A JSON string with a "selected_faqs" list.',
        agent=retrieval_agent,
        context=[intent_task],
    )

    response_task = Task(
        description=(
            "You are a support agent. You are provided:\n"
            "- USER_QUERY: {user_query}\n"
            "- CONTEXT from previous steps (intent + selected FAQs).\n\n"
            "Write a concise, friendly, and accurate reply to the user.\n"
            "Return STRICT JSON as:\n"
            "{\"reply\": \"<final answer>\", \"confidence\": 0.xx}"
        ),
        expected_output='A JSON string with "reply" and "confidence".',
        agent=response_agent,
        context=[intent_task, retrieval_task],
    )

    escalation_task = Task(
        description=(
            "You are given the full context of this interaction (intent, retrieved FAQs, reply).\n"
            "{context}\n\n"
            "Decide if this should be escalated to a human.\n"
            "Escalate if: confidence < 0.7 OR high-risk topics (payments, legal, security) OR strong user frustration.\n"
            "Return STRICT JSON as:\n"
            "{\"escalate\": true|false, \"reason\": \"short explanation\"}"
        ),
        expected_output='A JSON string like {"escalate": true, "reason": "billing dispute with low confidence"}',
        agent=escalation_agent,
        context=[intent_task, retrieval_task, response_task],
    )

    crew = Crew(
        agents=[intent_agent, retrieval_agent, response_agent, escalation_agent],
        tasks=[intent_task, retrieval_task, response_task, escalation_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew
