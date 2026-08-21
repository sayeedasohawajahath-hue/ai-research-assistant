"""
Guardrails using Groq-hosted Prompt Guard 2 (meta-llama/llama-prompt-guard-2-86m).
Used as both an input rail (before the agent runs) and an output rail
(before the final answer is shown).
"""
import os
import config  # noqa: F401  (loads .env)
from groq import Groq

GUARDRAIL_MODEL = "meta-llama/llama-prompt-guard-2-86m"

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client

def attack_score(text: str) -> float:
    """
    Returns a 0-1 score of how likely the text is a prompt injection /
    jailbreak attempt. Prompt Guard 2 returns the probability directly
    as a numeric string.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=GUARDRAIL_MODEL,
        messages=[{"role": "user", "content": text}],
    )
    raw = response.choices[0].message.content.strip()
    try:
        return float(raw)
    except ValueError:
        return 0.5  # unexpected format -> treat as uncertain/medium risk


def is_flagged(text: str, threshold: float = 0.5) -> bool:
    """Returns True if the text's attack score exceeds the threshold."""
    return attack_score(text) >= threshold