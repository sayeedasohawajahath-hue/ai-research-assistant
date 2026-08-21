"""
retrieve -> format context -> generate answer, using Groq.
"""
import os
from langchain_groq import ChatGroq

AGENT_MODEL = "openai/gpt-oss-120b"

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=AGENT_MODEL, api_key=os.environ.get("GROQ_API_KEY"))
    return _llm


def format_context(docs) -> str:
    """Joins retrieved document chunks into a single context string."""
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        parts.append(f"[Source {i+1}: {source} p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_from_context(question: str, docs) -> dict:
    """
    Given a question and retrieved docs, generate an answer grounded in that context.
    Returns {"answer": str, "context": str} so the judge can later score faithfulness.
    """
    context = format_context(docs)

    prompt = f"""You are a helpful research assistant. Answer the question using ONLY the context below.
If the context doesn't contain enough information to answer, say so honestly.

Context:
{context}

Question: {question}

Answer:"""

    llm = get_llm()
    response = llm.invoke(prompt)
    return {"answer": response.content, "context": context}