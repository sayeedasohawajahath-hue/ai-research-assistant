import os
import streamlit as st
import config  # noqa: F401  (loads .env)

from src.agents.tools import set_upload_retriever, clear_upload_retriever
from src.agents.react_agent import run_agent
from src.rag.retrievers import build_upload_retriever, get_seed_retriever
from src.rag.chain import answer_from_context
from src.guardrails.prompt_guard import attack_score, is_flagged
from src.evaluation.judge import judge_answer
from src.memory.context_manager import get_context_for_agent, count_tokens
from src.utils.logger import new_session_id, log_event, read_session_log, Timer

st.set_page_config(page_title="AI Research & Knowledge Assistant", layout="wide")

# ---- Session state setup ----
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role","content","meta":{...}}
if "guardrail_threshold" not in st.session_state:
    st.session_state.guardrail_threshold = 0.5
if "uploaded_pdf_name" not in st.session_state:
    st.session_state.uploaded_pdf_name = None

sid = st.session_state.session_id

st.title("AI Research & Knowledge Assistant")

tab_chat, tab_upload, tab_history = st.tabs(["Chat", "Upload", "History / Logs"])

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    st.session_state.guardrail_threshold = st.slider(
        "Guardrail threshold", 0.0, 1.0, st.session_state.guardrail_threshold, 0.05
    )
    token_count = count_tokens(st.session_state.messages)
    st.metric("Conversation tokens", token_count)
    if st.session_state.uploaded_pdf_name:
        st.info(f"Active upload: {st.session_state.uploaded_pdf_name}")

# ---- Upload tab ----
with tab_upload:
    st.subheader("Upload a PDF to query alongside the seed collection")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_path = f"/tmp/{uploaded_file.name}" if os.name != "nt" else f"uploaded_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Indexing your PDF..."):
            retriever = build_upload_retriever(temp_path)
            set_upload_retriever(retriever)
            st.session_state.uploaded_pdf_name = uploaded_file.name
        st.success(f"'{uploaded_file.name}' is now queryable in the Chat tab.")
    if st.session_state.uploaded_pdf_name and st.button("Clear uploaded PDF"):
        clear_upload_retriever()
        st.session_state.uploaded_pdf_name = None
        st.rerun()

# ---- Chat tab ----
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            meta = msg.get("meta")
            if meta:
                badges = []
                if meta.get("guardrail_flagged"):
                    badges.append("🚫 Guardrail flagged")
                if meta.get("judge_score") is not None:
                    badges.append(f"Faithfulness: {meta['judge_score']}/5 ✓")
                if meta.get("summarized"):
                    badges.append("🧠 Memory summarized")
                if badges:
                    st.caption(" | ".join(badges))

    user_input = st.chat_input("Ask a question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # --- Input guardrail ---
        with Timer() as t:
            in_score = attack_score(user_input)
        in_flagged = in_score >= st.session_state.guardrail_threshold
        log_event(sid, "guardrail_input", {"text": user_input, "score": in_score, "flagged": in_flagged}, t.elapsed_ms)

        if in_flagged:
            answer = "I can't process that request — it was flagged by the input guardrail as a potential prompt injection or jailbreak attempt."
            meta = {"guardrail_flagged": True}
            with st.chat_message("assistant"):
                st.write(answer)
                st.caption("🚫 Guardrail flagged")
            st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
        else:
            # --- Memory / context management ---
            history_for_agent, was_summarized = get_context_for_agent(
                [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            )
            if was_summarized:
                log_event(sid, "memory_summarized", {"turn_count": len(st.session_state.messages)})

            # --- Run the agent ---
            with st.spinner("Thinking..."):
                with Timer() as t:
                    result = run_agent(user_input, history_for_agent)
                log_event(sid, "tool_call", {"messages": str(result["messages"])}, t.elapsed_ms)

            answer = result["answer"]

            # --- Output guardrail ---
            with Timer() as t:
                out_score = attack_score(answer)
            out_flagged = out_score >= st.session_state.guardrail_threshold
            log_event(sid, "guardrail_output", {"text": answer, "score": out_score, "flagged": out_flagged}, t.elapsed_ms)

           # --- Judge (best-effort: only meaningful for RAG-grounded answers) ---
            judge_score = None
            try:
                from src.agents.tools import _upload_retriever
                active_retriever = _upload_retriever if _upload_retriever is not None else get_seed_retriever()
                docs = active_retriever.invoke(user_input)
                context_result = answer_from_context(user_input, docs)
                j = judge_answer(user_input, context_result["context"], answer)
                judge_score = j["score"]
                log_event(sid, "judge_score", j)
            except Exception:
                pass

            meta = {
                "guardrail_flagged": out_flagged,
                "judge_score": judge_score,
                "summarized": was_summarized,
            }
            with st.chat_message("assistant"):
                st.write(answer)
                badges = []
                if out_flagged:
                    badges.append("🚫 Output guardrail flagged")
                if judge_score is not None:
                    badges.append(f"Faithfulness: {judge_score}/5 ✓")
                if was_summarized:
                    badges.append("🧠 Memory summarized")
                if badges:
                    st.caption(" | ".join(badges))

            st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
            log_event(sid, "final_answer", {"answer": answer})

# ---- History tab ----
with tab_history:
    st.subheader(f"Session log: {sid}")
    events = read_session_log(sid)
    if events:
        st.dataframe(events, use_container_width=True)
    else:
        st.info("No events logged yet this session.")