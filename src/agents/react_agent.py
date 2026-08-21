"""
The ReAct agent: given a user turn, decides whether to call rag_lookup,
web_search, both, or neither. Uses LangGraph's prebuilt create_react_agent.
"""
import os
import config  # noqa: F401  (loads .env on import)
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from src.agents.tools import rag_lookup, web_search

AGENT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a helpful AI research assistant. You have access to two tools:

1. rag_lookup - searches the document collection (an uploaded PDF if present, otherwise
   a seed collection about the ReAct paper / agentic reasoning).
2. web_search - searches the live web for current information.

Decide which tool(s) to use based on the question. For broad questions with multiple
parts, you may call web_search more than once in the same turn to cover each part
before composing your final answer. If a question is fully answerable from the
document collection, prefer rag_lookup. If it needs current/live information,
use web_search. You can use both if genuinely helpful.
When calling web_search, always pass exactly one argument named "query" containing your search text.

Always give a clear, direct final answer after using your tools.
"""

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        llm = ChatGroq(model=AGENT_MODEL, api_key=os.environ.get("GROQ_API_KEY"))
        _agent = create_react_agent(llm, tools=[rag_lookup, web_search], prompt=SYSTEM_PROMPT)
    return _agent


def run_agent(user_message: str, chat_history: list | None = None) -> dict:
    """
    Runs the agent on a single user message (plus optional prior history).
    Returns {"answer": str, "messages": list} - messages includes all
    intermediate tool calls, useful for logging which tools fired.
    """
    agent = get_agent()
    messages = (chat_history or []) + [{"role": "user", "content": user_message}]
    result = agent.invoke({"messages": messages})
    final_message = result["messages"][-1]
    return {"answer": final_message.content, "messages": result["messages"]}