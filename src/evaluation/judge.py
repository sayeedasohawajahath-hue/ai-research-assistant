"""
LLM-as-judge: scores a RAG answer's faithfulness to its retrieved context.
Uses temperature=0 so repeated runs on the same answer get the same score.
"""
import os
import json
import config  # noqa: F401  (loads .env)
from langchain_groq import ChatGroq

JUDGE_MODEL = "openai/gpt-oss-120b"

_judge_llm = None


def get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatGroq(
            model=JUDGE_MODEL,
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0,
        )
    return _judge_llm


def judge_answer(question: str, context: str, answer: str) -> dict:
    """
    Scores faithfulness of `answer` to `context` on a 1-5 scale.
    Returns {"score": int, "justification": str}.
    """
    prompt = f"""You are an evaluator judging whether an AI-generated answer is faithful
to (i.e., actually supported by) the context it was given. Score the answer's
faithfulness on a scale of 1 to 5:

1 = completely unsupported / contradicts the context
3 = partially supported, some unsupported claims
5 = fully supported by the context, no fabrication

Respond ONLY with valid JSON in this exact format, no other text:
{{"score": <int 1-5>, "justification": "<one sentence>"}}

Context:
{context}

Question: {question}

Answer: {answer}
"""
    llm = get_judge_llm()
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown code fences if the model added them anyway
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        result = json.loads(raw)
        return {"score": int(result["score"]), "justification": result["justification"]}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 0, "justification": f"Could not parse judge response: {raw}"}