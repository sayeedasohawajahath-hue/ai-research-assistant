"""
Structured JSONL logger: one line per pipeline event (guardrail_input,
tool_call, retrieval, memory_summarized, judge_score, guardrail_output,
final_answer). Separate from Python's standard logging module.
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone

LOGS_DIR = "logs"


def get_session_log_path(session_id: str) -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    return os.path.join(LOGS_DIR, f"session_{session_id}.jsonl")


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(session_id: str, event_type: str, payload: dict, latency_ms: float | None = None):
    """
    Appends one structured event line to the session's JSONL log.
    event_type: one of guardrail_input, tool_call, retrieval,
    memory_summarized, judge_score, guardrail_output, final_answer
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
        "latency_ms": latency_ms,
    }
    path = get_session_log_path(session_id)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_session_log(session_id: str) -> list[dict]:
    """Reads back all events for a session as a list of dicts (for the History tab)."""
    path = get_session_log_path(session_id)
    if not os.path.exists(path):
        return []
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


class Timer:
    """Small helper: `with Timer() as t: ...` then use t.elapsed_ms"""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000