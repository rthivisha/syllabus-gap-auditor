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

if "show_sidebar_uploader" not in st.session_state:
    st.session_state.show_sidebar_uploader = False
if "guest_calls_used" not in st.session_state:
    st.session_state.guest_calls_used = 0

def _reset_to_initial_landing() -> None:
    """Restores the exact initial landing page state with zero active documents or chat history."""
    for key in [
        "selected_doc_id",
        "selected_doc_name",
        "selected_doc_pages",
        "selected_doc_chunks",
        "paired_sample_video",
        "uploaded_doc_id",
        "active_video_url",
        "last_ingest_sec",
        "last_audit_trace",
        "inspected_chunks",
    ]:
        st.session_state[key] = None
    st.session_state.messages = []
    st.session_state.suggested_questions = []
    st.session_state.awaiting_scope = False
    st.session_state.show_sidebar_uploader = False

with st.sidebar:
    st.markdown("### 📁 Document Knowledge Base")

    # Show active indexed document card (matching reference layout)
    if st.session_state.selected_doc_id:
        st.markdown("**Indexed Documents (1):**")
        col_doc, col_del = st.columns([4.2, 1], vertical_alignment="center")
        with col_doc:
            short_name = st.session_state.selected_doc_name or "Active PDF"
            if len(short_name) > 22:
                short_name = short_name[:20] + "..."
            st.markdown(
                f"""
                <div style="background:#162033;border:1px solid #283556;border-radius:10px;padding:9px 12px;font-size:13px;font-weight:600;color:#e2e8f0;">
                    🟢 {short_name}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_del:
            if st.button("🗑️", key="remove_active_doc_btn", help="Remove active document"):
                _reset_to_initial_landing()
                st.rerun()

        st.caption(
            f"{st.session_state.selected_doc_pages or '?'} pg · "
            f"{st.session_state.selected_doc_chunks or '?'} chunks"
        )
    else:
        st.info("No document active.\n\nUpload a PDF or select the sample book to begin.")

    st.markdown("")

    # 1. "+ Upload Another Document" button + inline sidebar uploader
    if st.button("➕ Upload Another Document", use_container_width=True, key="sidebar_toggle_upload"):
        st.session_state.show_sidebar_uploader = not st.session_state.show_sidebar_uploader

    if st.session_state.show_sidebar_uploader:
        sb_pdf = st.file_uploader(
            "Choose PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key="sidebar_pdf_uploader",
        )
        if sb_pdf is not None and st.button("🚀 Ingest PDF", type="primary", use_container_width=True, key="sidebar_ingest_btn"):
            with st.spinner(f"Indexing {sb_pdf.name} into Pinecone..."):
                info, up_err = None, None
                try:
                    with requests.post(
                        f"{API_BASE}/documents/upload/stream",
                        files={"file": (sb_pdf.name, sb_pdf.getvalue(), "application/pdf")},
                        data={"user_id": st.session_state.visitor_id},
                        stream=True,
                        timeout=300,
                    ) as r:
                        for line in r.iter_lines(decode_unicode=True):
                            if not line or not line.startswith("data: "):
                                continue
                            evt = json.loads(line[6:])
                            if evt.get("type") == "done":
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
                    st.session_state.uploaded_doc_id = info["doc_id"]
                    st.session_state.last_ingest_sec = info.get("elapsed_sec")
                    st.session_state.suggested_questions = info.get("suggested_questions", [])
                    st.session_state.show_sidebar_uploader = False
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": (
                                f"✅ **`{info['filename']}`** is indexed into Pinecone ({info.get('page_count', '?')} pages, "
                                f"{info.get('chunk_count', '?')} vectors).\n\n"
                                "- 🎥 Paste a **YouTube lecture URL** below to audit missed topics\n"
                                "- 📄 Ask any question about this PDF or about this project"
                            ),
                        }
                    ]
                    st.rerun()

    # 2. "🗑️ Clear All Knowledge" — resets UI cleanly back to initial first-load screen
    if st.button("🗑️ Clear All Knowledge", use_container_width=True, key="sidebar_clear_all_btn"):
        try:
            requests.post(f"{API_BASE}/documents/clear-user-uploads", timeout=6)
            d_res = requests.get(f"{API_BASE}/documents", timeout=5)
            if d_res.ok:
                st.session_state.cached_docs = d_res.json()
        except Exception:
            pass
        _reset_to_initial_landing()
        st.rerun()

    st.divider()

    # 3. Inspect Chunks (Under the Hood)
    with st.expander("🔍 Inspect Chunks (Under the Hood)", expanded=False):
        active_id = st.session_state.get("selected_doc_id")
        if active_id and st.button("Load First 10 Pinecone Chunks", use_container_width=True):
            c_res = requests.get(f"{API_BASE}/documents/{active_id}/chunks", params={"limit": 10}, timeout=8)
            st.session_state["inspected_chunks"] = c_res.json() if c_res.ok else []
        for ch in st.session_state.get("inspected_chunks", []):
            st.caption(f"`Page {ch['page_number']}` · `{ch['id']}` · **{ch['chars']} chars**")
            with st.container(border=True):
                st.markdown(ch["text"][:450])

    # 4. Secure Inference Engine & 3-Call Free Quota
    with st.expander("⚙️ Inference Engine & Quota", expanded=False):
        has_byok = bool(st.session_state.get("byok_gemini_key") or st.session_state.get("byok_groq_key"))
        if has_byok:
            st.success("🔓 **BYOK Mode Active** — Unlimited turns using your session key.")
        else:
            used = min(3, st.session_state.guest_calls_used)
            st.info(
                f"🔒 **Managed Server Keys Active** (Keys stored securely on backend)\n\n"
                f"**Free Trial Quota:** `{used} / 3` LLM calls used (includes Gap Audits & Q&A)."
            )
        st.caption("Primary: **Gemini 2.5 Flash** · Fast Q&A & Failover: **Groq Live**")
        st.text_input("Optional Gemini API Key (BYOK)", type="password", key="byok_gemini_key", placeholder="AIza...")
        st.text_input("Optional Groq API Key (BYOK)", type="password", key="byok_groq_key", placeholder="gsk_...")

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

    # Always-visible Ready Sample Books row (1-click instant demo without re-uploading)
    ready_docs = [d for d in (st.session_state.get("cached_docs") or []) if d.get("status") == "ready"]
    harness_doc = next(
        (d for d in ready_docs if "harness" in d.get("filename", "").lower()),
        None,
    )
    claude_doc = next(
        (d for d in ready_docs if "claude" in d.get("filename", "").lower()),
        None,
    )

    sample_buttons = []
    if harness_doc:
        sample_buttons.append(
            {
                "doc": harness_doc,
                "label": f"📘 {harness_doc['filename']} ({harness_doc.get('page_count', 83)}p · Ready in Pinecone)",
                "video_url": "https://www.youtube.com/watch?v=mv9kfF_kJd0",
            }
        )
    if claude_doc:
        sample_buttons.append(
            {
                "doc": claude_doc,
                "label": f"📘 {claude_doc['filename']} ({claude_doc.get('page_count', 108)}p · Ready in Pinecone)",
                "video_url": "https://www.youtube.com/watch?v=mv9kfF_kJd0",
            }
        )
    # Fallback: if cached_docs is empty on cold load, still show the pre-indexed Harness PDF button
    if not sample_buttons:
        sample_buttons.append(
            {
                "doc": {
                    "doc_id": "harness-engineering-e81b5a4855ad",
                    "filename": "harness engineering.pdf",
                    "page_count": 83,
                    "chunk_count": 82,
                },
                "label": "📘 harness engineering.pdf (83p · Ready in Pinecone)",
                "video_url": "https://www.youtube.com/watch?v=mv9kfF_kJd0",
            }
        )

    st.markdown(
        '<div class="divider-label">OR TEST INSTANTLY WITH AN ALREADY-INDEXED SAMPLE BOOK</div>',
        unsafe_allow_html=True,
    )
    sb_cols = st.columns(len(sample_buttons))
    for s_idx, item in enumerate(sample_buttons):
        doc_obj = item["doc"]
        with sb_cols[s_idx]:
            if st.button(item["label"], key=f"quick_sample_{doc_obj['doc_id']}", use_container_width=True):
                st.session_state.selected_doc_id = doc_obj["doc_id"]
                st.session_state.selected_doc_name = doc_obj["filename"]
                st.session_state.selected_doc_pages = doc_obj.get("page_count", 83)
                st.session_state.selected_doc_chunks = doc_obj.get("chunk_count", 82)
                st.session_state.paired_sample_video = {
                    "video_url": item["video_url"],
                    "label": item["label"],
                }
                st.session_state.suggested_questions = [
                    f"Summarize the key topics covered in {doc_obj['filename']}",
                    "What are the core architectural principles in this PDF?",
                    "How does this Syllabus Gap Audit AI project work?",
                    f"Audit this YouTube lecture against the whole book: {item['video_url']}",
                ]
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": (
                            f"✅ Activated **`{doc_obj['filename']}`** ({doc_obj.get('page_count', 83)} pages, "
                            f"{doc_obj.get('chunk_count', 82)} vectors in Pinecone).\n\n"
                            f"- 🎥 **Paired Demo Lecture:** `{item['video_url']}`\n"
                            "- 📄 Ask any question about this PDF, ask about this project, or click **▶️ Audit Paired Lecture** above!"
                        ),
                    }
                ]
                st.rerun()

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
            st.session_state.suggested_questions = info.get("suggested_questions", [])
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"✅ **`{info['filename']}`** is indexed into Pinecone ({info.get('page_count', '?')} pages, "
                        f"{info.get('chunk_count', '?')} vectors).\n\n"
                        "What would you like to do next?\n"
                        "- 🎥 **Paste a YouTube lecture URL** below to audit which topics from this PDF the video missed\n"
                        "- 📄 **Ask any question** about the contents of this PDF\n"
                        "- ⚡ **Ask about this project** to explore its LangGraph & Inverse-RAG architecture"
                    ),
                }
            ]
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

    if st.session_state.get("suggested_questions"):
        st.caption("💡 **SUGGESTED QUESTIONS (DERIVED FROM YOUR INGESTED PDF & PROJECT):**")
        q_cols = st.columns(4)
        for q_idx, q_text in enumerate(st.session_state["suggested_questions"][:4]):
            with q_cols[q_idx]:
                if st.button(f"👉 {q_text}", key=f"post_ingest_q_{q_idx}", use_container_width=True):
                    st.session_state.queued_pill_prompt = q_text
                    st.rerun()

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
                has_byok = bool(st.session_state.get("byok_gemini_key") or st.session_state.get("byok_groq_key"))
                if not has_byok:
                    st.session_state.guest_calls_used += 1
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
