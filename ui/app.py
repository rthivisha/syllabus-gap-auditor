import html
import json
import uuid

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Syllabus Gap Audit AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #172554 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 14px;
        padding: 1.35rem 1.7rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
        margin: 0 0 0.3rem 0;
    }
    .hero-sub {
        color: #cbd5e1;
        font-size: 0.92rem;
        margin: 0;
    }
    .engine-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #34d399;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    /* Clean Pill Chat Input with High-Contrast Light Text */
    [data-testid="stChatInput"] {
        background-color: #0f172a !important;
        border: 1.5px solid rgba(255, 255, 255, 0.28) !important;
        border-radius: 28px !important;
        box-shadow: 0 10px 28px -10px rgba(0, 0, 0, 0.5) !important;
        padding: 4px 8px !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: transparent !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #f8fafc !important;
        caret-color: #ffffff !important;
        -webkit-text-fill-color: #f8fafc !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(248, 250, 252, 0.65) !important;
        -webkit-text-fill-color: rgba(248, 250, 252, 0.65) !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 999px !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_session_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "visitor_id": f"guest_{uuid.uuid4().hex[:8]}",
        "messages": [],
        "active_video_url": None,
        "active_video_id": None,
        "show_inline_transcript": False,
        "awaiting_scope": False,
        "selected_doc_id": None,
        "selected_doc_name": None,
        "selected_doc_pages": "—",
        "selected_doc_chunks": "—",
        "suggested_questions": [],
        "queued_pill_prompt": None,
        "feedback_submitted_ids": [],
        "cached_docs": None,
        "last_uploaded_sig": None,
        "upload_success_message": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


def _fetch_docs(force_refresh: bool = False) -> list:
    cached_docs = st.session_state.get("cached_docs")
    if not force_refresh and isinstance(cached_docs, list):
        return cached_docs

    try:
        response = requests.get(f"{API_BASE}/documents", timeout=5)
        if not response.ok:
            raise requests.RequestException(
                f"Could not load documents ({response.status_code})."
            )
        documents = response.json()
        if not isinstance(documents, list):
            raise ValueError("Invalid documents response: expected a list.")
    except Exception:
        if not isinstance(cached_docs, list):
            cached_docs = []
        st.session_state.cached_docs = cached_docs
        return cached_docs

    st.session_state.cached_docs = documents
    return documents


def _fetch_sessions() -> list:
    try:
        r = requests.get(
            f"{API_BASE}/sessions",
            params={"user_id": st.session_state.visitor_id},
            timeout=8,
        )
        return r.json() if r.ok else []
    except Exception:
        return []


def _fetch_quota() -> dict:
    try:
        r = requests.get(f"{API_BASE}/quota/{st.session_state.visitor_id}", timeout=5)
        return r.json() if r.ok else {"used": 0, "limit": 3, "remaining": 3}
    except Exception:
        return {"used": 0, "limit": 3, "remaining": 3}


def _render_verified_sources(sources: list) -> None:
    if not sources:
        return

    st.caption("📚 Verified Document Sources")
    source_columns = st.columns(min(3, len(sources)))
    for idx, source in enumerate(sources[:3]):
        raw_score = source.get("bi_encoder_score", source.get("score"))
        match_label = "Match —"
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            rerank_score = source.get(
                "flashrank_score", source.get("rerank_score")
            )
            try:
                score = float(rerank_score)
            except (TypeError, ValueError):
                score = -1.0
        if 0.0 <= score <= 1.0:
            match_label = f"Match {round(score * 100)}%"

        quote = str(source.get("snippet") or source.get("text") or "").strip()
        if len(quote) > 160:
            quote = quote[:160].rstrip() + "..."
        page = html.escape(str(source.get("page_number", 1)))
        section = html.escape(str(source.get("section_title", ""))[:45])
        chunk_id = html.escape(str(source.get("chunk_id", "")))
        safe_quote = html.escape(quote)
        with source_columns[idx]:
            st.markdown(
                f"""
                <div style="background:#0f172a;border:1px solid rgba(148,163,184,0.25);
                            border-radius:10px;padding:10px 12px;height:100%;">
                    <div style="color:#f8fafc;font-weight:700;font-size:12px;margin-bottom:4px;">
                        📄 Page {page} ({match_label})
                    </div>
                    <div style="color:#38bdf8;font-size:11px;margin-bottom:6px;">
                        {section}
                    </div>
                    <div style="color:#94a3b8;font-size:10px;margin-bottom:6px;word-break:break-all;">
                        Chunk: {chunk_id}
                    </div>
                    <div style="color:#cbd5e1;font-size:11px;line-height:1.4;">
                        &ldquo;{safe_quote}&rdquo;
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_audit_trace(trace: dict) -> None:
    verdict = trace.get("evaluator_status", "passed").upper()
    badge = "✅ PASSED" if verdict == "PASSED" else "❌ FAILED"
    total_sec = round(trace.get("latency_ms", 0) / 1000, 1)
    with st.expander(
        f"Agent Execution Trace — {badge} "
        f"({total_sec}s | Retries: {trace.get('retry_count', 0)})",
        expanded=False,
    ):
        timings = trace.get("step_timings", {})
        st.markdown(
            f"- **1. Video & Transcription:** `{trace.get('windows_count', 0)}` "
            f"windows (`{timings.get('total_ingestion_sec', 0.0)}s`)\n"
            f"- **2. BGE Embedding + Pinecone & FlashRank:** "
            f"Embed `{timings.get('bge_embedding_sec', 0.0)}s` | Retrieval & "
            f"Rerank `{timings.get('bi_encoder_plus_flashrank_sec', 0.0)}s`\n"
            f"- **3. LLM Audit & `skills.md` Evaluator:** "
            f"`{trace.get('sections_checked', 0)}` sections | "
            f"{trace.get('evaluation_notes') or 'All rubric checks passed.'}"
        )


def _render_message_feedback(message_id: str, message_index: int) -> None:
    submitted_ids = st.session_state.get("feedback_submitted_ids", [])
    if message_id in submitted_ids:
        return

    feedback = st.feedback(
        "thumbs", key=f"fb_{message_id}_{message_index}"
    )
    if feedback is None:
        return

    response = requests.post(
        f"{API_BASE}/feedback",
        json={
            "message_id": message_id,
            "user_id": st.session_state.visitor_id,
            "rating": 1 if feedback == 1 else -1,
        },
        timeout=5,
    )
    if not response.ok:
        st.error(f"Could not save feedback: {response.text}")
        return

    st.session_state.feedback_submitted_ids = [*submitted_ids, message_id]
    st.toast("Feedback saved!")


docs = _fetch_docs()
quota = _fetch_quota()

with st.sidebar:
    st.markdown("### 📁 Document Knowledge Base")
    ready_docs = [d for d in docs if d.get("status") == "ready"]
    if ready_docs:
        doc_options = {f"{d['doc_name']} ({d['page_count']}p)": d["doc_id"] for d in ready_docs}
        labels = list(doc_options.keys())
        default_idx = 0
        if st.session_state.selected_doc_id in doc_options.values():
            default_idx = list(doc_options.values()).index(st.session_state.selected_doc_id)
        chosen_label = st.selectbox(
            "Active Reference Book",
            labels,
            index=default_idx,
            label_visibility="collapsed",
        )
        previous_doc_id = st.session_state.selected_doc_id
        st.session_state.selected_doc_id = doc_options[chosen_label]
        active_doc = next((d for d in ready_docs if d["doc_id"] == st.session_state.selected_doc_id), None)
        if active_doc:
            st.session_state.selected_doc_name = active_doc["doc_name"]
            st.session_state.selected_doc_pages = active_doc.get("page_count", "—")
            st.session_state.selected_doc_chunks = active_doc.get("chunk_count", "—")
            st.session_state.suggested_questions = active_doc.get(
                "suggested_questions", []
            )
            st.caption(f"✅ `{active_doc['chunk_count']}` child chunks indexed")
        if previous_doc_id != st.session_state.selected_doc_id:
            st.session_state.awaiting_scope = False
            st.session_state.active_video_url = None
            st.session_state.active_video_id = None
            st.session_state.active_book_banner = None
    else:
        st.info("No documents indexed yet. Upload a PDF on the main screen to start.")

    for pd in [d for d in docs if d.get("status") == "processing"]:
        st.info(f"⏳ Indexing `{pd['doc_name']}`...")

    if st.button("↻ Refresh documents", key="refresh_documents"):
        _fetch_docs(force_refresh=True)
        st.rerun()

    if st.button("＋ New Audit Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.active_video_url = None
        st.session_state.active_video_id = None
        st.session_state.show_inline_transcript = False
        st.session_state.awaiting_scope = False

    st.markdown("---")

    with st.expander("🔍 Indexed Textbook TOC", expanded=False):
        if ready_docs and st.session_state.selected_doc_id:
            active_doc = next((d for d in ready_docs if d["doc_id"] == st.session_state.selected_doc_id), None)
            if active_doc and active_doc.get("sections"):
                for sec in active_doc["sections"][:18]:
                    st.caption(f"• p.{sec['start_page']} — {sec['section_title']}")
        else:
            st.caption("Select or upload a reference book to inspect sections.")

    with st.expander("⚙️ Inference Engine & Free Trial", expanded=True):
        st.markdown("🟢 **Gemini 3.8 Flash + Pinecone Connected**")
        st.caption("Hybrid Retrieval: `BGE-base-en-v1.5` + `FlashRank` + `skills.md` Evaluator")
        st.progress(
            min(1.0, quota["used"] / max(1, quota["limit"])),
            text=f"Free Guest Audits: {quota['remaining']} of {quota['limit']} remaining",
        )
        st.caption("🔒 Server-side API keys secured via Cloudflare AI Gateway.")

    st.markdown("---")
    st.markdown("**Recent Audits**")
    for s in _fetch_sessions()[:6]:
        title = s.get("title") or s["session_id"][:8]
        if st.button(f"💬 {title[:26]}...", key=f"hist_{s['session_id']}", use_container_width=True):
            st.session_state.session_id = s["session_id"]
            loaded = requests.get(f"{API_BASE}/sessions/{s['session_id']}/messages", timeout=8).json()
            st.session_state.messages = loaded
            restored_url = None
            for m in loaded:
                for tok in (m.get("content") or "").split():
                    if "youtube.com/" in tok or "youtu.be/" in tok:
                        restored_url = tok.strip("()[]<>.,")
                        break
            st.session_state.active_video_url = restored_url

st.markdown(
    """
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
            <div>
                <div class="hero-title">⚡ Syllabus Gap Audit AI</div>
                <p class="hero-sub">
                    Full Multimodal Pipeline: PDF Parsing → Hierarchical Chunking → Pinecone Hybrid Search → FlashRank Reranking → Grounded Video Gap Audit.
                </p>
            </div>
            <span class="engine-pill">● Engine: Gemini 3.8 Flash + Pinecone</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_audit, tab_approvals, tab_metrics = st.tabs(
    ["💬 Syllabus Audit", "✅ Approvals Queue", "📊 Live Metrics"]
)

with tab_audit:
    selected_doc = next(
        (
            doc
            for doc in docs
            if doc.get("doc_id") == st.session_state.selected_doc_id
        ),
        None,
    )
    if selected_doc:
        st.session_state.selected_doc_name = selected_doc.get(
            "doc_name", st.session_state.selected_doc_name
        )
        st.session_state.selected_doc_pages = selected_doc.get("page_count", "—")
        st.session_state.selected_doc_chunks = selected_doc.get("chunk_count", "—")
        if selected_doc.get("status") == "ready":
            st.session_state.suggested_questions = selected_doc.get(
                "suggested_questions", []
            )

    active_name = (
        st.session_state.get("selected_doc_name") or "No PDF Selected"
    )
    active_pages = st.session_state.get("selected_doc_pages", "—")
    active_chunks = st.session_state.get("selected_doc_chunks", "—")
    doc_status = selected_doc.get("status") if selected_doc else None
    if doc_status == "ready":
        status_text = "● READY FOR AUDIT"
        status_color = "#34d399"
        status_background = "rgba(16,185,129,0.15)"
        status_border = "rgba(16,185,129,0.4)"
    elif doc_status == "failed":
        status_text = "● INDEXING FAILED"
        status_color = "#f87171"
        status_background = "rgba(239,68,68,0.15)"
        status_border = "rgba(239,68,68,0.4)"
    elif doc_status == "processing":
        status_text = "● INDEXING"
        status_color = "#fbbf24"
        status_background = "rgba(245,158,11,0.15)"
        status_border = "rgba(245,158,11,0.4)"
    else:
        status_text = "● SELECT A PDF"
        status_color = "#fbbf24"
        status_background = "rgba(245,158,11,0.15)"
        status_border = "rgba(245,158,11,0.4)"

    st.markdown(
        f"""
        <div style="background:#0f172a;border:1px solid rgba(56,189,248,0.35);border-radius:12px;
                    padding:12px 18px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="color:#f8fafc;font-weight:700;font-size:15px;">📄 {html.escape(str(active_name))}</span>
                <span style="margin-left:12px;background:#1e293b;color:#94a3b8;padding:3px 9px;border-radius:6px;font-size:12px;">
                    {html.escape(str(active_pages))} Pages
                </span>
                <span style="margin-left:6px;background:#1e293b;color:#94a3b8;padding:3px 9px;border-radius:6px;font-size:12px;">
                    {html.escape(str(active_chunks))} Chunks
                </span>
                <span style="margin-left:6px;background:#1e293b;color:#38bdf8;padding:3px 9px;border-radius:6px;font-size:12px;">
                    Pinecone Hybrid + FlashRank
                </span>
            </div>
            <span style="background:{status_background};color:{status_color};border:1px solid {status_border};
                         padding:4px 10px;border-radius:999px;font-size:11px;font-weight:700;">
                {status_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggested_questions = (
        list(st.session_state.get("suggested_questions") or [])
        if doc_status == "ready"
        else []
    )
    fallback_questions = [
        f"What are the core architectural components in {active_name}?",
        f"Which topics in {active_name} are most critical for production?",
        "How does this system manage state, interrupts, and recovery?",
        "Paste a YouTube URL to audit missing topics from this PDF",
    ]
    if selected_doc and doc_status == "ready":
        suggested_questions = (suggested_questions + fallback_questions)[:4]
    else:
        suggested_questions = []
    if suggested_questions:
        st.caption("💡 SUGGESTED QUESTIONS (DERIVED SPECIFICALLY FROM THIS DOCUMENT)")
        question_columns = st.columns(4)
        for idx, question in enumerate(suggested_questions):
            if question_columns[idx].button(
                f"💡 {question}",
                key=f"suggested_question_{selected_doc['doc_id']}_{idx}",
                use_container_width=True,
                disabled=quota["remaining"] <= 0,
            ):
                st.session_state.queued_pill_prompt = question
                st.rerun()

    with st.expander(
        "📥 Upload Reference PDF (Auto-Indexes on Drop) or Pick a Ready Sample Book",
        expanded=(
            len(ready_docs) == 0
            or bool(st.session_state.get("upload_success_message"))
        ),
    ):
        uploaded_pdf = st.file_uploader(
            "Drop your PDF textbook or syllabus — indexing starts automatically",
            type=["pdf"],
            key="auto_pdf_uploader",
        )
        if uploaded_pdf is not None:
            file_sig = f"{uploaded_pdf.name}:{uploaded_pdf.size}"
            if st.session_state.get("last_uploaded_sig") != file_sig:
                try:
                    with st.spinner(
                        f"Parsing & embedding **{uploaded_pdf.name}** into Pinecone..."
                    ):
                        files = {
                            "file": (
                                uploaded_pdf.name,
                                uploaded_pdf.getvalue(),
                                "application/pdf",
                            )
                        }
                        up_res = requests.post(
                            f"{API_BASE}/documents/upload",
                            files=files,
                            data={"uploaded_by": st.session_state.visitor_id},
                            timeout=120,
                        )
                        if up_res.ok:
                            doc_info = up_res.json()
                            doc_id = doc_info.get("doc_id")
                            if doc_info.get("status") != "ready" or not doc_id:
                                raise ValueError(
                                    "Upload did not return a ready document with an ID."
                                )
                            uploaded_doc = {**doc_info, "status": "ready"}
                            cached_docs = st.session_state.get("cached_docs")
                            if not isinstance(cached_docs, list):
                                cached_docs = []
                            st.session_state.cached_docs = [
                                uploaded_doc,
                                *(
                                    item
                                    for item in cached_docs
                                    if item.get("doc_id") != doc_id
                                ),
                            ]
                            st.session_state.last_uploaded_sig = file_sig
                            st.session_state.selected_doc_id = doc_id
                            st.session_state.selected_doc_name = doc_info.get(
                                "doc_name", uploaded_pdf.name
                            )
                            st.session_state.selected_doc_pages = doc_info.get(
                                "page_count", "—"
                            )
                            st.session_state.selected_doc_chunks = doc_info.get(
                                "chunk_count", "—"
                            )
                            st.session_state.suggested_questions = doc_info.get(
                                "suggested_questions", []
                            )
                            st.session_state.awaiting_scope = False
                            st.session_state.active_video_url = None
                            st.session_state.active_video_id = None
                            st.session_state.active_book_banner = None
                            st.session_state.upload_success_message = (
                                f"✅ Indexed **{st.session_state.selected_doc_name}** "
                                f"({st.session_state.selected_doc_chunks} child chunks "
                                "embedded in Pinecone) — now active!"
                            )
                            st.rerun()
                        else:
                            st.error(
                                f"PDF indexing failed ({up_res.status_code}): "
                                f"{up_res.text}"
                            )
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Upload error: {exc}")

        if ready_docs:
            st.caption("OR CLICK AN ALREADY-INDEXED SAMPLE BOOK BELOW TO ACTIVATE IMMEDIATELY")
            sample_cols = st.columns(min(3, len(ready_docs)))
            for idx, d in enumerate(ready_docs[:3]):
                clean_title = d["doc_name"].replace("%20", " ")
                with sample_cols[idx]:
                    if st.button(
                        f"📘 {clean_title[:28]} ({d['page_count']}p)",
                        key=f"sample_{d['doc_id']}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_doc_id = d["doc_id"]
                        st.session_state.selected_doc_name = d["doc_name"]
                        st.session_state.selected_doc_pages = d.get(
                            "page_count", "—"
                        )
                        st.session_state.selected_doc_chunks = d.get(
                            "chunk_count", "—"
                        )
                        st.session_state.suggested_questions = d.get(
                            "suggested_questions", []
                        )
                        st.session_state.awaiting_scope = False
                        st.session_state.active_video_url = None
                        st.session_state.active_video_id = None
                        st.session_state.active_book_banner = clean_title
                        st.rerun()

        if st.session_state.get("active_book_banner"):
            st.success(
                f"✅ **{st.session_state.active_book_banner}** is indexed and active! "
                "Paste a YouTube video URL below that you want to find the gap audit for."
            )
        if st.session_state.get("upload_success_message"):
            st.success(st.session_state.upload_success_message)
            st.session_state.upload_success_message = None

    # Render Chat Messages
    rendered_urls = set()
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            v_url = msg.get("video_url")
            if v_url and v_url not in rendered_urls:
                st.video(v_url)
                rendered_urls.add(v_url)

            if msg.get("trace"):
                _render_audit_trace(msg["trace"])

            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                _render_verified_sources(
                    msg.get("citations") or msg.get("sources") or []
                )

            if msg["role"] == "assistant" and msg.get("id"):
                _render_message_feedback(msg["id"], idx)

    if quota["remaining"] <= 0:
        st.warning(
            f"You have reached the free guest limit (**{quota['limit']} audits**). Start a new session or deploy with your own keys for unlimited audits."
        )
    if st.session_state.get("active_video_id"):
        t_lbl = (
            "🔼 Hide Current Video Transcript (25s Windows)"
            if st.session_state.show_inline_transcript
            else "📄 View Current Video Transcript (25s Windows)"
        )
        if st.button(t_lbl, key="btn_inline_tr"):
            st.session_state.show_inline_transcript = (
                not st.session_state.show_inline_transcript
            )

        if st.session_state.show_inline_transcript:
            v_detail = requests.get(
                f"{API_BASE}/videos/{st.session_state.active_video_id}", timeout=8
            )
            if v_detail.ok:
                vd = v_detail.json()
                with st.container(border=True):
                    st.subheader(f"🎙️ {vd.get('title', 'Current Video Transcript')}")
                    for window in vd.get("windows", []):
                        st.markdown(
                            f"**`[{window['timestamp']}]`** "
                            f"{window.get('speech_text', '')}"
                        )

with tab_approvals:
    st.subheader("Human-in-the-Loop (HITL) Escalation Queue")
    st.caption(
        "Borderline syllabus gaps (medium similarity or escalated after max retries) routed for human verification."
    )
    try:
        app_res = requests.get(f"{API_BASE}/approvals", timeout=8)
        items = app_res.json() if app_res.ok else []
    except Exception:
        items = []

    if not items:
        st.success("No pending escalations in the queue.")
    else:
        for item in items:
            with st.container(border=True):
                reason_txt = (
                    item.get("reason")
                    or item.get("escalation_reason")
                    or item.get("notes")
                    or f"Similarity score: {item.get('similarity_score', 'borderline')}"
                )
                st.markdown(
                    f"**Topic:** {item.get('topic_title', 'Untitled Topic')} "
                    f"*(Page {item.get('page_number', 'N/A')})*"
                )
                st.caption(
                    f"Reason: {reason_txt} | Closest Video Timestamp: "
                    f"`{item.get('closest_timestamp', 'N/A')}`"
                )
                c1, c2 = st.columns(2)
                if c1.button("✅ Confirm Missed Gap", key=f"appr_{item['id']}", use_container_width=True):
                    requests.post(
                        f"{API_BASE}/approvals/{item['id']}/resolve",
                        json={"decision": "approved", "reviewer_id": st.session_state.visitor_id},
                        timeout=8,
                    )
                    st.toast("Confirmed as missed gap!")
                if c2.button("❌ Dismiss (Covered)", key=f"rej_{item['id']}", use_container_width=True):
                    requests.post(
                        f"{API_BASE}/approvals/{item['id']}/resolve",
                        json={"decision": "rejected", "reviewer_id": st.session_state.visitor_id},
                        timeout=8,
                    )
                    st.toast("Dismissed!")

with tab_metrics:
    st.subheader("Real-Time Pipeline & Evaluator Telemetry")
    try:
        m_res = requests.get(f"{API_BASE}/metrics", timeout=8)
        m = m_res.json() if m_res.ok else {}
    except Exception:
        m = {}

    if m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Audits Run", m.get("total_runs", 0))
        c2.metric("Evaluator Pass Rate", f"{m.get('pass_rate_pct', 0)}%")
        c3.metric("Avg Latency", f"{round(m.get('avg_latency_ms', 0) / 1000, 1)}s")
        c4.metric("Pending HITL Reviews", m.get("pending_approvals", 0))
        st.markdown("---")
        st.json(m)


# ROOT-LEVEL PINNED CHAT INPUT (Always visible at the bottom of the screen)
queued_prompt = st.session_state.pop("queued_pill_prompt", None)
user_typed = st.chat_input(
    "Ask about this document, specify a chapter/section, or paste a YouTube URL...",
    disabled=quota["remaining"] <= 0,
)
active_prompt = queued_prompt or user_typed

if active_prompt and quota["remaining"] > 0:
    has_video_url = (
        "youtube.com/" in active_prompt.lower()
        or "youtu.be/" in active_prompt.lower()
    )
    is_second_video = has_video_url and bool(
        st.session_state.get("active_video_url")
    )
    session_id = str(uuid.uuid4()) if is_second_video else st.session_state.session_id
    awaiting_scope = (
        False if is_second_video else bool(st.session_state.awaiting_scope)
    )
    if is_second_video:
        st.session_state.messages = []
        st.session_state.active_video_url = None
        st.session_state.active_video_id = None
        st.session_state.awaiting_scope = False
    st.session_state.session_id = session_id
    st.session_state.awaiting_scope = awaiting_scope

    user_message = {"role": "user", "content": active_prompt}
    for token in active_prompt.split():
        if "youtube.com/" in token.lower() or "youtu.be/" in token.lower():
            clean_url = token.strip("()[]<>.,")
            st.session_state.active_video_url = clean_url
            user_message["video_url"] = clean_url
            break
    st.session_state.messages.append(user_message)

    with tab_audit:
        with st.chat_message("user"):
            if user_message.get("video_url"):
                st.video(user_message["video_url"])
            st.markdown(active_prompt)

        payload = {
            "session_id": session_id,
            "user_id": st.session_state.visitor_id,
            "doc_id": st.session_state.selected_doc_id or "",
            "message": active_prompt,
            "video_url": (
                st.session_state.active_video_url if awaiting_scope else None
            ),
            "prior_clarification_asked": awaiting_scope,
        }
        response_data = None
        stream_error = None
        with st.chat_message("assistant"):
            try:
                with st.status("Processing...", expanded=True) as progress:
                    with requests.post(
                        f"{API_BASE}/chat/turn/stream",
                        json=payload,
                        stream=True,
                        timeout=(15, 600),
                    ) as resp:
                        if not resp.ok:
                            stream_error = (
                                f"Chat request failed ({resp.status_code}): "
                                f"{resp.text}"
                            )
                            progress.update(
                                label="Chat request failed", state="error"
                            )
                        else:
                            for raw_line in resp.iter_lines(
                                decode_unicode=True
                            ):
                                if not raw_line:
                                    continue
                                line = (
                                    raw_line.decode("utf-8")
                                    if isinstance(raw_line, bytes)
                                    else raw_line
                                )
                                if not line.startswith("data:"):
                                    continue
                                try:
                                    event = json.loads(line[5:].strip())
                                except ValueError as exc:
                                    stream_error = f"Invalid progress event: {exc}"
                                    progress.update(
                                        label="Invalid progress event",
                                        state="error",
                                    )
                                    break

                                if event.get("type") == "stage":
                                    progress.update(
                                        label=str(
                                            event.get("text") or "Working..."
                                        ),
                                        state="running",
                                    )
                                elif event.get("type") == "done":
                                    response_data = event.get("payload")
                                    progress.update(
                                        label="Chat turn complete",
                                        state="complete",
                                    )
                                elif event.get("type") == "error":
                                    stream_error = str(
                                        event.get("text") or "Chat turn failed."
                                    )
                                    progress.update(
                                        label="Chat turn failed", state="error"
                                    )
                                    break
            except requests.RequestException as exc:
                stream_error = f"Could not reach backend: {exc}"

            if stream_error:
                st.error(stream_error)
            elif not isinstance(response_data, dict):
                st.error(
                    "The chat stream ended without a completed response."
                )
            else:
                if "awaiting_scope" in response_data:
                    st.session_state.awaiting_scope = bool(
                        response_data["awaiting_scope"]
                    )
                else:
                    st.session_state.awaiting_scope = awaiting_scope
                if response_data.get("video_id"):
                    st.session_state.active_video_id = response_data["video_id"]
                reply = response_data.get("reply")
                if not isinstance(reply, str) or not reply:
                    st.error("Backend returned no chat reply.")
                else:
                    assistant_message = {
                        "role": "assistant",
                        "content": reply,
                        "id": response_data.get("message_id"),
                    }
                    for key in ("citations", "trace", "video_url"):
                        if response_data.get(key):
                            assistant_message[key] = response_data[key]
                    st.session_state.messages.append(assistant_message)
                    if assistant_message.get("trace"):
                        _render_audit_trace(assistant_message["trace"])
                    st.markdown(reply)
                    _render_verified_sources(
                        assistant_message.get("citations", [])
                    )
                    if assistant_message.get("id"):
                        _render_message_feedback(
                            assistant_message["id"],
                            len(st.session_state.messages) - 1,
                        )

