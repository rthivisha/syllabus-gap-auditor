import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
import streamlit as st
import importlib
from ui.components import architecture_board
importlib.reload(architecture_board)
from ui.components.architecture_board import render_live_ingestion_board

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Syllabus Gap Audit AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Clean Dark Navy & Indigo Typography Theme (Matching Reference UI) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #090e1a;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1424;
        border-right: 1px solid #1e293b;
    }
    .hero-banner {
        background: linear-gradient(125deg, #0f172a 0%, #17153b 55%, #0f172a 100%);
        border: 1px solid #283556;
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        color: #7dd3fc;
        letter-spacing: -0.4px;
        margin: 0 0 6px 0;
    }
    .hero-sub {
        font-size: 13.5px;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }
    .engine-pill {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
    }
    .upload-card {
        background: #0d1526;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 14px;
    }
    .upload-card h3 {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
        margin: 0 0 6px 0;
    }
    .upload-card p {
        font-size: 13.5px;
        color: #94a3b8;
        margin: 0;
    }
    .divider-label {
        text-align: center;
        color: #64748b;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        margin: 18px 0 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "visitor_id" not in st.session_state:
    st.session_state.visitor_id = f"guest_{uuid.uuid4().hex[:8]}"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None  # Starts clean on first load!
if "selected_doc_name" not in st.session_state:
    st.session_state.selected_doc_name = None
if "selected_doc_pages" not in st.session_state:
    st.session_state.selected_doc_pages = None
if "selected_doc_chunks" not in st.session_state:
    st.session_state.selected_doc_chunks = None
if "paired_sample_video" not in st.session_state:
    st.session_state.paired_sample_video = None
if "uploaded_doc_id" not in st.session_state:
    st.session_state.uploaded_doc_id = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "awaiting_scope" not in st.session_state:
    st.session_state.awaiting_scope = False
if "active_video_url" not in st.session_state:
    st.session_state.active_video_url = None

if "cached_docs" not in st.session_state:
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=5)
        st.session_state.cached_docs = r.json() if r.ok else []
    except Exception:
        st.session_state.cached_docs = []

# Curated Sample Books + Paired Lecture Videos for 1-Click Recruiter Demo
SAMPLE_PAIRINGS = [
    {
        "match_key": "harness",
        "label": "📘 Harness Engineering Guide (83p)",
        "video_url": "https://www.youtube.com/watch?v=mv9kfF_kJd0",
        "video_label": "🎥 Sample Lecture: AI Harness & Context Engineering",
    },
    {
        "match_key": "claude",
        "label": "📘 Claude Code Architecture (108p)",
        "video_url": "https://www.youtube.com/watch?v=mv9kfF_kJd0",
        "video_label": "🎥 Sample Lecture: Agentic Coding Systems",
    },
]

# --- Minimal Left Sidebar (Matching Reference UI) ---
with st.sidebar:
    st.markdown("### 📁 Document Knowledge Base")
    if st.session_state.selected_doc_name:
        st.success(
            f"**{st.session_state.selected_doc_name}**\n\n"
            f"{st.session_state.selected_doc_pages} pages · {st.session_state.selected_doc_chunks} vectors in Pinecone"
        )
        if st.button("🔄 Switch / Upload Another PDF", use_container_width=True):
            st.session_state.selected_doc_id = None
            st.session_state.selected_doc_name = None
            st.session_state.selected_doc_pages = None
            st.session_state.selected_doc_chunks = None
            st.session_state.paired_sample_video = None
            st.session_state.suggested_questions = []
            st.session_state.messages = []
            st.session_state.awaiting_scope = False
            st.rerun()
    else:
        st.info("No document active yet.\n\nUpload a PDF or pick a ready sample book to start.")

    st.divider()

    with st.expander("🔍 Inspect Chunks (Under the Hood)", expanded=False):
        active_id = st.session_state.get("selected_doc_id")
        if active_id and st.button("Load First 10 Pinecone Chunks", use_container_width=True):
            c_res = requests.get(f"{API_BASE}/documents/{active_id}/chunks", params={"limit": 10}, timeout=8)
            st.session_state["inspected_chunks"] = c_res.json() if c_res.ok else []
        for ch in st.session_state.get("inspected_chunks", []):
            st.caption(f"`Page {ch.get('page_number', '?')}` · `{ch.get('id', '')}` · **{ch.get('chars', len(ch.get('text', '')))} chars**")
            with st.container(border=True):
                st.markdown(ch.get("text", "")[:450])

    with st.expander("⚙️ Inference Engine & BYOK", expanded=False):
        st.caption("Primary: **Gemini 2.5 Flash** · Fast Q&A & Failover: **Groq Live**")
        st.text_input("Your Gemini API Key (Optional)", type="password", key="byok_gemini_key")
        st.text_input("Your Groq API Key (Optional)", type="password", key="byok_groq_key")

# --- Hero Header Banner (Always Visible at Top) ---
st.markdown(
    """
    <div class="hero-banner">
        <div>
            <div class="hero-title">⚡ Syllabus Gap Audit AI</div>
            <p class="hero-sub">
                Inverse-RAG Coverage Auditor: PDF Parsing → Adaptive Section Chunking → Pinecone Vector Index →
                Set-Difference Retrieval → Parallel Map-Reduce Gap Audit.
            </p>
        </div>
        <div class="engine-pill">● Engine: Gemini 2.5 + Groq + Pinecone</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# STATE 1: CLEAN ONBOARDING LANDING VIEW (Shown until a PDF is uploaded or picked)
# ==============================================================================
if not st.session_state.selected_doc_id:
    st.markdown(
        """
        <div class="upload-card">
            <h3>📥 Upload Syllabus or Textbook PDF to Run Inverse-RAG</h3>
            <p>Drop your PDF below and click <b>Ingest & Embed</b> to index it into Pinecone, then paste any YouTube lecture URL to uncover missed topics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_action = st.columns([3.2, 1.1], vertical_alignment="center")
    with col_up:
        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key="landing_pdf_uploader",
        )
    with col_action:
        ingest_btn = st.button(
            "🚀 Ingest & Embed",
            type="primary",
            disabled=(uploaded_pdf is None),
            use_container_width=True,
            key="landing_ingest_btn",
        )

    board_slot = st.empty()
    render_live_ingestion_board(
        active_step=6 if st.session_state.selected_doc_id else 0,
        doc_name=st.session_state.selected_doc_name,
        page_count=st.session_state.selected_doc_pages,
        vector_count=st.session_state.selected_doc_chunks,
        elapsed_sec=st.session_state.get("last_ingest_sec"),
        placeholder=board_slot,
    )

    if uploaded_pdf is not None and ingest_btn:
        info, up_err = None, None
        try:
            with requests.post(
                f"{API_BASE}/documents/upload/stream",
                files={"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")},
                data={"user_id": st.session_state.visitor_id},
                stream=True,
                timeout=300,
            ) as r:
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    evt = json.loads(line[6:])
                    if evt.get("type") == "step":
                        render_live_ingestion_board(
                            active_step=evt["step"],
                            doc_name=uploaded_pdf.name,
                            placeholder=board_slot,
                        )
                    elif evt.get("type") == "done":
                        info = evt["payload"]
                    elif evt.get("type") == "error":
                        up_err = evt["text"]
        except Exception as exc:
            up_err = str(exc)

        if up_err:
            st.error(up_err)
        elif info:
            st.session_state.selected_doc_id = info["doc_id"]
            st.session_state.selected_doc_name = info["filename"]
            st.session_state.selected_doc_pages = info.get("page_count", "?")
            st.session_state.selected_doc_chunks = info.get("chunk_count", "?")
            st.session_state.last_ingest_sec = info.get("elapsed_sec")
            st.session_state.paired_sample_video = None
            render_live_ingestion_board(
                active_step=6,
                doc_name=info["filename"],
                page_count=info.get("page_count"),
                vector_count=info.get("chunk_count"),
                elapsed_sec=info.get("elapsed_sec"),
                placeholder=board_slot,
            )
            d_res = requests.get(f"{API_BASE}/documents", timeout=5)
            if d_res.ok:
                st.session_state.cached_docs = d_res.json()
            st.rerun()

    # Instant 1-Click Sample Books + Paired Lecture Videos
    ready_docs = [d for d in st.session_state.cached_docs if d.get("status") == "ready"]
    matched_samples = []
    seen_sample_ids = set()
    for sp in SAMPLE_PAIRINGS:
        found = next(
            (
                d
                for d in ready_docs
                if sp["match_key"] in (d.get("filename") or d.get("doc_name") or d.get("doc_id") or "").lower()
                and d.get("doc_id") not in seen_sample_ids
            ),
            None,
        )
        if found:
            seen_sample_ids.add(found.get("doc_id"))
            matched_samples.append((sp, found))

    if matched_samples:
        st.markdown('<div class="divider-label">OR TEST INSTANTLY WITH A READY SAMPLE BOOK & PAIRED LECTURE</div>', unsafe_allow_html=True)
        s_cols = st.columns(len(matched_samples))
        for idx, (sp, doc_obj) in enumerate(matched_samples):
            with s_cols[idx]:
                if st.button(sp["label"], key=f"sample_pick_{doc_obj['doc_id']}", use_container_width=True):
                    doc_title = doc_obj.get("filename") or doc_obj.get("doc_name") or sp["label"]
                    st.session_state.selected_doc_id = doc_obj["doc_id"]
                    st.session_state.selected_doc_name = doc_title
                    st.session_state.selected_doc_pages = doc_obj.get("page_count", "?")
                    st.session_state.selected_doc_chunks = doc_obj.get("chunk_count", "?")
                    st.session_state.paired_sample_video = sp
                    st.session_state.suggested_questions = []  # Only shown for user-uploaded PDFs
                    st.rerun()

# ==============================================================================
# STATE 2: REACTIVE AUDIT & CHAT WORKSPACE (Appears after PDF is active)
# ==============================================================================
else:
    render_live_ingestion_board(
        active_step=6,
        doc_name=st.session_state.selected_doc_name,
        page_count=st.session_state.selected_doc_pages,
        vector_count=st.session_state.selected_doc_chunks,
        elapsed_sec=st.session_state.get("last_ingest_sec"),
    )

    # If a sample book was picked, offer its paired YouTube lecture URL as a 1-click audit trigger
    if st.session_state.paired_sample_video and not st.session_state.messages:
        sp = st.session_state.paired_sample_video
        c1, c2 = st.columns([3, 1.3], vertical_alignment="center")
        with c1:
            st.info(f"**Paired Lecture Ready:** `{sp['video_url']}` — Click to audit which topics from **{st.session_state.selected_doc_name}** this video missed.")
        with c2:
            if st.button("▶️ Audit Paired Lecture", type="primary", use_container_width=True):
                st.session_state.queued_pill_prompt = f"Audit this lecture against the whole book: {sp['video_url']}"
                st.rerun()

    # Render Chat & Audit History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            sources = msg.get("sources") or msg.get("citations") or []
            if sources:
                st.caption("📚 Verified Document Sources")
                s_cols = st.columns(min(3, len(sources)))
                for s_idx, src in enumerate(sources[:3]):
                    with s_cols[s_idx]:
                        with st.container(border=True):
                            st.markdown(f"**Page {src.get('page_number', '?')}** · *{src.get('section_title', '')[:42]}*")
                            quote = (src.get("text") or src.get("evidence_quote") or src.get("snippet") or src.get("parent_text") or "")[:180]
                            st.caption(f"{quote}...")

# ==============================================================================
# ROOT-LEVEL PINNED CHAT INPUT (Always accessible at bottom)
# ==============================================================================
typed_prompt = st.chat_input(
    "Ask about this project, ask about your PDF, or paste a YouTube lecture URL to audit..."
)
active_prompt = st.session_state.pop("queued_pill_prompt", None) or typed_prompt

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    for tok in active_prompt.split():
        if "youtube.com/" in tok or "youtu.be/" in tok:
            st.session_state.active_video_url = tok.strip("()[]<>.,")
            break

    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.status("Running LangGraph Inverse-RAG pipeline...", expanded=True) as run_status:
            t0 = time.perf_counter()
            final_payload, err_msg = None, None
            payload = {
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.visitor_id,
                "doc_id": st.session_state.selected_doc_id,
                "message": active_prompt,
                "video_url": st.session_state.active_video_url if st.session_state.awaiting_scope else None,
                "prior_clarification_asked": st.session_state.awaiting_scope,
                "user_gemini_key": st.session_state.get("byok_gemini_key") or None,
                "user_groq_key": st.session_state.get("byok_groq_key") or None,
            }
            try:
                with requests.post(f"{API_BASE}/chat/turn/stream", json=payload, stream=True, timeout=120) as r:
                    for raw_line in r.iter_lines(decode_unicode=True):
                        if not raw_line or not raw_line.startswith("data: "):
                            continue
                        evt = json.loads(raw_line[6:])
                        if evt.get("type") == "stage":
                            run_status.update(label=evt["text"], state="running")
                        elif evt.get("type") == "done":
                            final_payload = evt["payload"]
                        elif evt.get("type") == "error":
                            err_msg = evt["text"]
            except Exception as exc:
                err_msg = str(exc)

            elapsed = round(time.perf_counter() - t0, 2)
            if err_msg:
                run_status.update(label=f"Failed ({elapsed}s)", state="error")
                st.error(err_msg)
            elif final_payload:
                run_status.update(label=f"Completed in {elapsed}s", state="complete", expanded=False)
                trace_data = final_payload.get("trace") or {}
                st.session_state.last_audit_trace = {
                    "elapsed_sec": elapsed,
                    "latency_ms": trace_data.get("latency_ms", elapsed * 1000),
                    "windows_count": trace_data.get("windows_count", 0),
                    "sections_checked": trace_data.get("sections_checked", 0),
                    "evaluator_status": trace_data.get("evaluator_status", "passed"),
                    "retry_count": trace_data.get("retry_count", 0),
                }
                reply = final_payload.get("reply") or final_payload.get("chat_response") or ""
                sources = (
                    final_payload.get("citations")
                    or final_payload.get("retrieved_evidence")
                    or final_payload.get("cited_chunks")
                    or []
                )
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
                st.session_state.awaiting_scope = (
                    "would you like to analyze this video" in reply.lower()
                    or bool(final_payload.get("awaiting_scope"))
                )
                st.rerun()
