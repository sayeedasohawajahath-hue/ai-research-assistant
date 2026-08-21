"""
Tools available to the ReAct agent: rag_lookup and web_search.
"""
from langchain_core.tools import tool
from ddgs import DDGS

from src.rag.retrievers import get_seed_retriever
from src.rag.chain import answer_from_context

# Set by app.py when a user uploads a PDF this session
_upload_retriever = None


def set_upload_retriever(retriever):
    global _upload_retriever
    _upload_retriever = retriever


def clear_upload_retriever():
    global _upload_retriever
    _upload_retriever = None


@tool
def rag_lookup(question: str) -> str:
    """
    Looks up information from the document collection to answer a question.
    Checks the user's uploaded PDF first if one exists, otherwise falls back
    to the seed collection (currently: the ReAct paper). Use this for questions
    about the uploaded document or about ReAct-style agents/tool-calling.
    """
    retriever = _upload_retriever if _upload_retriever is not None else get_seed_retriever()
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant information found in the document collection."
    result = answer_from_context(question, docs)
    return result["answer"]


@tool
def web_search(query: str) -> str:
    """
    Searches the live web for current information. Use this for questions
    about recent events, current facts, or anything not covered by the
    document collection. Can be called multiple times in one turn for
    different sub-aspects of a broad question.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No web results found."
    formatted = []
    for r in results:
        formatted.append(f"{r.get('title', '')}\n{r.get('body', '')}\nSource: {r.get('href', '')}")
    return "\n\n".join(formatted)