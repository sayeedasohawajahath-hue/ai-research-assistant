"""
Trims/summarizes conversation history only once it gets "too long".
Short conversations pass through untouched.
"""
import os
import config  # noqa: F401  (loads .env)
import tiktoken
from langchain_groq import ChatGroq

SUMMARY_MODEL = "openai/gpt-oss-120b"
TOKEN_THRESHOLD = 3000
TURN_THRESHOLD = 12
KEEP_RECENT_TURNS = 4

_encoding = None
_summary_llm = None


def get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def get_summary_llm():
    global _summary_llm
    if _summary_llm is None:
        _summary_llm = ChatGroq(model=SUMMARY_MODEL, api_key=os.environ.get("GROQ_API_KEY"))
    return _summary_llm


def count_tokens(messages: list) -> int:
    """Rough token count of a list of {"role": ..., "content": ...} messages."""
    enc = get_encoding()
    total = 0
    for m in messages:
        total += len(enc.encode(m.get("content", "")))
    return total


def needs_summarization(messages: list) -> bool:
    if len(messages) > TURN_THRESHOLD:
        return True
    if count_tokens(messages) > TOKEN_THRESHOLD:
        return True
    return False


def summarize_older_half(messages: list) -> str:
    """Summarizes the older half of the conversation into a short memory note."""
    split_point = max(len(messages) - KEEP_RECENT_TURNS, 1)
    older = messages[:split_point]

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in older)
    prompt = f"""Summarize the following conversation history into a short memory note
that preserves key facts, decisions, and context needed to continue the conversation.
Be concise (a few sentences).

Conversation:
{transcript}

Summary:"""

    llm = get_summary_llm()
    response = llm.invoke(prompt)
    return response.content.strip()


def get_context_for_agent(messages: list) -> tuple[list, bool]:
    """
    Returns (messages_to_send, was_summarized).
    If the conversation isn't too long, returns messages unchanged.
    Otherwise, returns a compressed version: [summary message] + recent turns.
    """
    if not needs_summarization(messages):
        return messages, False

    split_point = max(len(messages) - KEEP_RECENT_TURNS, 1)
    recent = messages[split_point:]
    summary_text = summarize_older_half(messages)

    summary_message = {
        "role": "system",
        "content": f"[Earlier conversation summary]: {summary_text}",
    }
    return [summary_message] + recent, True