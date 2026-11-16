import html
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
DEFAULT_FREE_LIMIT = 5

st.set_page_config(
    page_title="Syllabus Gap Audit AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1440px; padding-top: 1.25rem; }
    div[data-testid="stSidebar"] {
        background: #080d18;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #172554 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        color: #60a5fa;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 0.35rem;
    }
    .hero-sub { color: #cbd5e1; font-size: 0.95rem; margin: 0; }
    .engine-pill {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 999px;
        color: #34d399;
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.35rem 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_session_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "visitor_id": f"guest_{uuid.uuid4().hex[:12]}",
        "active_book_select": None,
        "messages": [],
        "active_video_url": None,
        "active_video_id": None,
        "show_inline_transcript": False,
        "awaiting_scope": False,
        "selected_doc_id": None,
        "job_started_at": None,
        "job_session_id": None,
        "job_error": None,
        "history_error": None,
        "refresh_data": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_json(
    path: str, timeout: int = 8, params: Optional[Dict[str, str]] = None
) -> Tuple[Optional[Any], Optional[str]]:
    try:
        response = requests.get(
            f"{API_BASE}{path}", params=params, timeout=timeout
        )
    except requests.RequestException as exc:
        return None, str(exc)
    if not response.ok:
        return None, f"HTTP {response.status_code}: {response.text}"
    try:
        return response.json(), None
    except ValueError as exc:
        return None, f"Invalid JSON response: {exc}"


def _restore_session(session_id: str) -> None:
    messages, error = _get_json(
        f"/sessions/{quote(session_id, safe='')}/messages", timeout=10
    )
    if error or not isinstance(messages, list):
        st.session_state.history_error = error or "Invalid session messages response."
        return

    status_data, status_error = _get_json(
        f"/sessions/{quote(session_id, safe='')}/status", timeout=5
    )
    if status_error or not isinstance(status_data, dict):
        st.session_state.history_error = status_error or "Invalid session status response."
        return

    st.session_state.history_error = None
    st.session_state.session_id = session_id
    st.session_state.messages = messages
    st.session_state.job_started_at = None
    st.session_state.job_session_id = None
    st.session_state.job_error = None
    st.session_state.show_inline_transcript = False

    restored_url = None
    for message in reversed(messages):
        restored_url = message.get("video_url") or None
        if not restored_url:
            for token in (message.get("content") or "").split():
                if "youtube.com/" in token.lower() or "youtu.be/" in token.lower():
                    restored_url = token.strip("()[]<>.,")
                    break
        if restored_url:
            break
    st.session_state.active_video_url = restored_url

    last_assistant = next(
        (
            message
            for message in reversed(messages)
            if message.get("role") == "assistant"
        ),
        None,
    )
    st.session_state.awaiting_scope = bool(
        last_assistant
        and "Would you like to analyze"
        in last_assistant.get("content", "")
    )

    if status_data.get("awaiting_scope") is not None:
        st.session_state.awaiting_scope = bool(status_data["awaiting_scope"])
    if status_data.get("video_url"):
        st.session_state.active_video_url = status_data["video_url"]
    st.session_state.active_video_id = status_data.get("video_id")
    if status_data.get("trace") and status_data.get("message_id"):
        for message in reversed(messages):
            if message.get("id") == status_data["message_id"]:
                message["trace"] = status_data["trace"]
                break

    if status_data.get("status") == "running":
        st.session_state.job_session_id = session_id
        st.session_state.job_started_at = time.perf_counter()
    elif status_data.get("status") == "error":
        st.session_state.job_error = status_data.get(
            "error", "The background audit failed."
        )


def _select_document(doc_id: str) -> None:
    st.session_state.selected_doc_id = doc_id
    st.session_state.active_book_select = doc_id


_init_session_state()

visitor_id = st.session_state.visitor_id
docs_data, docs_error = _get_json("/documents", timeout=8)
docs: List[Dict[str, Any]] = docs_data if isinstance(docs_data, list) else []
if docs_data is not None and not isinstance(docs_data, list):
    docs_error = "Invalid document list response."

quota_data, quota_error = _get_json(
    f"/quota/{quote(visitor_id, safe='')}", timeout=5
)
if (
    not isinstance(quota_data, dict)
    or not isinstance(quota_data.get("used"), int)
    or not isinstance(quota_data.get("limit"), int)
    or not isinstance(quota_data.get("remaining"), int)
):
    quota_error = quota_error or "Invalid quota response."
    quota_data = None

engine_data, engine_error = _get_json("/engine/status", timeout=5)
if engine_data is not None and not isinstance(engine_data, dict):
    engine_error = "Invalid engine status response."
    engine_data = None

sessions_data, sessions_error = _get_json(
    "/sessions", params={"user_id": visitor_id}, timeout=8
)
sessions: List[Dict[str, Any]] = (
    sessions_data if isinstance(sessions_data, list) else []
)
if sessions_data is not None and not isinstance(sessions_data, list):
    sessions_error = "Invalid recent audits response."

videos_data, videos_error = _get_json("/videos", timeout=8)
videos: List[Dict[str, Any]] = videos_data if isinstance(videos_data, list) else []
if videos_data is not None and not isinstance(videos_data, list):
    videos_error = "Invalid video list response."

ready_docs = [doc for doc in docs if doc.get("status") == "ready"]
docs_by_id = {doc["doc_id"]: doc for doc in ready_docs}
ready_doc_ids = list(docs_by_id)
if st.session_state.selected_doc_id not in docs_by_id:
    st.session_state.selected_doc_id = ready_doc_ids[0] if ready_doc_ids else None
if ready_doc_ids and st.session_state.active_book_select not in docs_by_id:
    st.session_state.active_book_select = st.session_state.selected_doc_id

with st.sidebar:
    st.markdown("### 📁 Document Knowledge Base")
    if docs_error:
        st.error(f"Could not load documents: {docs_error}")
    if ready_doc_ids:
        selected_doc_id = st.selectbox(
            "Active Reference Book",
            ready_doc_ids,
            index=None,
            format_func=lambda doc_id: (
                f"{docs_by_id[doc_id]['doc_name']} "
                f"({docs_by_id[doc_id].get('page_count', 0)}p)"
            ),
            key="active_book_select",
        )
        st.session_state.selected_doc_id = selected_doc_id
        active_doc = docs_by_id[selected_doc_id]
        st.caption(
            f"✅ `{active_doc.get('chunk_count', 0)}` child chunks indexed"
        )
    else:
        st.info("Upload a PDF in the main workspace to begin.")

    for processing_doc in (
        doc for doc in docs if doc.get("status") in {"processing", "queued"}
    ):
        st.info(f"⏳ Indexing `{processing_doc.get('doc_name', 'PDF')}`...")

    if st.button("＋ New Audit Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.active_video_url = None
        st.session_state.active_video_id = None
        st.session_state.show_inline_transcript = False
        st.session_state.awaiting_scope = False
        st.session_state.job_started_at = None
        st.session_state.job_session_id = None
        st.session_state.job_error = None
        st.rerun()

    if st.button("🔄 Refresh Knowledge Base", use_container_width=True):
        st.rerun()

    with st.expander("🔍 Inspect Chunks & 25s Transcripts", expanded=False):
        active_doc = docs_by_id.get(st.session_state.selected_doc_id)
        if active_doc and active_doc.get("sections"):
            st.markdown("**Indexed Textbook TOC**")
            for section in active_doc["sections"][:15]:
                st.caption(
                    f"• p.{section.get('start_page', 1)} — "
                    f"{section.get('section_title', 'Untitled section')}"
                )
        elif not docs_error:
            st.caption("Upload a document to inspect its indexed sections.")

        if videos_error:
            st.error(f"Could not load cached videos: {videos_error}")
        elif videos:
            st.markdown("**Cached 25-second Video Windows**")
            video_map = {
                video["video_id"]: video
                for video in videos
                if video.get("video_id")
            }
            video_ids = list(video_map)
            if video_ids:
                inspect_video_id = st.selectbox(
                    "Select video",
                    video_ids,
                    format_func=lambda video_id: (
                        f"{video_map[video_id].get('title', 'Video')[:28]} "
                        f"({video_map[video_id].get('window_count', 0)} windows)"
                    ),
                    key="inspect_video_select",
                )
                if st.button("Load 25s Windows", use_container_width=True):
                    st.session_state.active_video_id = inspect_video_id
                    st.session_state.show_inline_transcript = True
                    st.rerun()
        else:
            st.caption("No cached video transcripts yet.")

    with st.expander("⚙️ Inference Engine & Free Trial", expanded=True):
        if engine_error or not engine_data:
            st.warning(f"Engine status unavailable: {engine_error or 'No response'}")
        else:
            model_name = engine_data.get("auditor_model", "Unknown model")
            gemini_ready = bool(engine_data.get("gemini_configured"))
            pinecone_ready = bool(engine_data.get("pinecone_configured"))
            st.markdown(
                f"{'🟢' if gemini_ready else '🟠'} **Gemini `{model_name}` "
                f"{'configured' if gemini_ready else 'not configured'}**"
            )
            st.caption(
                f"Embeddings: `{engine_data.get('embedding_model', 'unknown')}` | "
                f"FlashRank: `{engine_data.get('reranker_model', 'unknown')}`"
            )
            st.caption(
                f"{'🟢' if pinecone_ready else '🟠'} Pinecone "
                f"{'configured' if pinecone_ready else 'not configured'}"
            )

        if quota_error or not quota_data:
            st.error(f"Free-trial quota unavailable: {quota_error or 'No response'}")
        else:
            limit = max(1, quota_data["limit"])
            used = max(0, quota_data["used"])
            remaining = max(0, quota_data["remaining"])
            st.progress(
                min(1.0, used / limit),
                text=f"Free Guest Audits: {remaining} of {quota_data['limit']} remaining",
            )
        st.caption("API credentials remain server-side.")

    st.markdown("---")
    st.markdown("**💬 Recent Audits**")
    if sessions_error:
        st.error(f"Could not load recent audits: {sessions_error}")
    for session in sessions[:8]:
        session_id = session.get("session_id")
        if not session_id:
            continue
        title = (
            session.get("video_title")
            or session.get("title")
            or session_id[:8]
        )
        if st.button(
            f"💬 {title[:28]}...",
            key=f"audit_{session_id}",
            use_container_width=True,
        ):
            _restore_session(session_id)
            st.rerun()
    if st.session_state.history_error:
        st.error(st.session_state.history_error)

engine_name = html.escape(
    str((engine_data or {}).get("auditor_model", "Engine status unavailable"))
)
engine_ready = bool(
    engine_data
    and engine_data.get("api_status") == "online"
    and engine_data.get("gemini_configured")
    and engine_data.get("pinecone_configured")
)

st.markdown(
    f"""
    <div class="hero-card">
      <div style="display:flex; justify-content:space-between; align-items:center;
                  flex-wrap:wrap; gap:0.75rem;">
        <div>
          <div class="hero-title">⚡ Syllabus Gap Audit AI</div>
          <p class="hero-sub">
            Compare a lecture with your reference book using transcript windows,
            slide OCR, hybrid retrieval, and a grounded evaluator.
          </p>
        </div>
        <span class="engine-pill">
          {'● API Online' if engine_ready else '● Check engine configuration'}
          · {engine_name} · Pinecone + FlashRank
        </span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("📥 Upload a Reference PDF or Choose a Ready Book")
    upload_col, button_col = st.columns([4, 1])
    uploaded_pdf = upload_col.file_uploader(
        "Upload PDF textbook or syllabus",
        type=["pdf"],
        label_visibility="collapsed",
        key="reference_pdf_upload",
    )
    if button_col.button(
        "Ingest & Index",
        type="primary",
        use_container_width=True,
        disabled=uploaded_pdf is None,
    ):
        try:
            upload_response = requests.post(
                f"{API_BASE}/documents/upload",
                files={
                    "file": (
                        uploaded_pdf.name,
                        uploaded_pdf.getvalue(),
                        "application/pdf",
                    )
                },
                data={"uploaded_by": visitor_id},
                timeout=60,
            )
        except requests.RequestException as exc:
            st.error(f"Could not upload the PDF: {exc}")
        else:
            if upload_response.ok:
                st.success(
                    f"**{uploaded_pdf.name}** queued. Indexing is running in the background."
                )
                st.rerun()
            else:
                st.error(f"Could not queue the PDF: {upload_response.text}")

    if ready_docs:
        st.caption("Or select an already-indexed reference book:")
        sample_columns = st.columns(min(3, len(ready_docs)))
        for index, doc in enumerate(ready_docs[:3]):
            with sample_columns[index]:
                st.button(
                    f"📘 {doc['doc_name'][:28]} "
                    f"({doc.get('page_count', 0)}p)",
                    key=f"sample_{doc['doc_id']}",
                    use_container_width=True,
                    on_click=_select_document,
                    args=(doc["doc_id"],),
                )

if docs_error:
    st.warning(f"Document list is unavailable: {docs_error}")
elif not ready_docs:
    st.info("Upload a PDF to create your reference knowledge base.")

if st.session_state.job_error:
    st.error(st.session_state.job_error)

rendered_urls = set()
for index, message in enumerate(st.session_state.messages):
    role = message.get("role", "assistant")
    with st.chat_message(role):
        video_url = message.get("video_url")
        if video_url and video_url not in rendered_urls:
            st.video(video_url)
            rendered_urls.add(video_url)

        trace = message.get("trace")
        if trace:
            timings = trace.get("step_timings", {})
            verdict = str(trace.get("evaluator_status", "failed")).upper()
            badge = "✅ PASSED" if verdict == "PASSED" else "❌ FAILED"
            total_seconds = round(trace.get("latency_ms", 0) / 1000, 1)
            with st.expander(
                f"Agent Execution Trace — {badge} "
                f"({total_seconds}s | Retries: {trace.get('retry_count', 0)})",
                expanded=False,
            ):
                st.markdown(
                    f"- **Video & transcription:** `{trace.get('windows_count', 0)}` "
                    f"windows | `{timings.get('total_ingestion_sec', 0.0)}s`\n"
                    f"- **Retrieval:** inventory "
                    f"`{timings.get('pinecone_inventory_sec', 0.0)}s` | embedding "
                    f"`{timings.get('bge_embedding_sec', 0.0)}s` | Pinecone fetch "
                    f"`{timings.get('pinecone_child_fetch_sec', 0.0)}s` | "
                    f"Bi-Encoder + FlashRank "
                    f"`{timings.get('bi_encoder_plus_flashrank_sec', 0.0)}s`\n"
                    f"- **Audit & evaluator:** `{trace.get('sections_checked', 0)}` "
                    f"sections | {trace.get('evaluation_notes') or 'Rubric checks passed.'}"
                )
                if trace.get("repair_notes"):
                    st.markdown(f"**Repair notes:** {trace['repair_notes']}")

        st.markdown(message.get("content", ""))
        citations = message.get("citations") or []
        if citations:
            with st.expander(f"📚 Retrieved Evidence ({len(citations)} chunks)"):
                for citation_index, citation in enumerate(citations, 1):
                    st.markdown(
                        f"**[{citation_index}] {citation.get('doc_name', 'Book')} — "
                        f"Page {citation.get('page_number', '?')} | "
                        f"{citation.get('section_title', 'Section')}**"
                    )
                    st.caption(
                        f"Bi-Encoder: `{citation.get('bi_encoder_score', citation.get('score', 0.0))}` | "
                        f"FlashRank: `{citation.get('rerank_score', 0.0)}`"
                    )
                    st.write(citation.get("snippet", ""))

        message_id = message.get("id")
        if role == "assistant" and message_id:
            feedback = st.feedback(
                "thumbs", key=f"feedback_{message_id}_{index}"
            )
            if feedback is not None:
                try:
                    feedback_response = requests.post(
                        f"{API_BASE}/feedback",
                        json={
                            "message_id": message_id,
                            "user_id": visitor_id,
                            "rating": 1 if feedback == 1 else -1,
                        },
                        timeout=5,
                    )
                except requests.RequestException as exc:
                    st.error(f"Could not save feedback: {exc}")
                else:
                    if feedback_response.ok:
                        st.toast("Feedback saved.")
                    else:
                        st.error(f"Could not save feedback: {feedback_response.text}")

if st.session_state.active_video_id:
    transcript_label = (
        "🔼 Hide 25s Transcript Windows"
        if st.session_state.show_inline_transcript
        else "📄 View 25s Transcript Windows"
    )
    if st.button(transcript_label, key="toggle_inline_transcript"):
        st.session_state.show_inline_transcript = (
            not st.session_state.show_inline_transcript
        )
        st.rerun()

    if st.session_state.show_inline_transcript:
        video_detail, video_error = _get_json(
            f"/videos/{quote(st.session_state.active_video_id, safe='')}",
            timeout=10,
        )
        if video_error or not isinstance(video_detail, dict):
            st.error(video_error or "Invalid transcript response.")
        else:
            with st.container(border=True):
                st.subheader(
                    f"🎙️ {video_detail.get('title', 'Transcribed Windows')}"
                )
                windows = video_detail.get("windows", [])
                st.caption(f"Total 25-second windows: `{len(windows)}`")
                for window in windows:
                    with st.expander(
                        f"[{window.get('timestamp', '??:??')}] "
                        f"{window.get('start_sec', 0)}s–{window.get('end_sec', 0)}s"
                    ):
                        st.write(window.get("speech_text", ""))
                        if window.get("slide_ocr_text"):
                            st.caption(
                                f"🖼️ Slide OCR: {window['slide_ocr_text']}"
                            )


@st.fragment(run_every="1s")
def _live_audit_monitor() -> None:
    session_id = st.session_state.job_session_id
    if not session_id:
        return

    status_data, status_error = _get_json(
        f"/sessions/{quote(session_id, safe='')}/status", timeout=5
    )
    if status_error or not isinstance(status_data, dict):
        st.warning(f"Could not check audit status: {status_error or 'Invalid response'}")
        return

    job_status = status_data.get("status", "idle")
    if job_status == "running":
        if not st.session_state.job_started_at:
            st.session_state.job_started_at = time.perf_counter()
        elapsed = time.perf_counter() - st.session_state.job_started_at
        minutes, seconds = divmod(elapsed, 60)
        clock = (
            f"{int(minutes)}m {seconds:04.1f}s"
            if minutes >= 1
            else f"{seconds:.1f}s"
        )
        st.info(f"⏱️ **{clock}** — Audit running in the background.")
        return

    messages, messages_error = _get_json(
        f"/sessions/{quote(session_id, safe='')}/messages", timeout=10
    )
    if messages_error or not isinstance(messages, list):
        st.error(
            f"Could not refresh chat messages: {messages_error or 'Invalid response'}"
        )
        return

    if job_status == "idle":
        status_data["status"] = "error"
        status_data["error"] = (
            "The background job status is unavailable; it may have been lost "
            "when the API restarted."
        )
    if status_data.get("trace") and status_data.get("message_id"):
        for message in reversed(messages):
            if message.get("id") == status_data["message_id"]:
                message["trace"] = status_data["trace"]
                break

    st.session_state.messages = messages
    st.session_state.awaiting_scope = bool(
        status_data.get("awaiting_scope", False)
    )
    if status_data.get("video_url"):
        st.session_state.active_video_url = status_data["video_url"]
    if status_data.get("video_id"):
        st.session_state.active_video_id = status_data["video_id"]
    st.session_state.job_error = (
        status_data.get("error")
        if status_data.get("status") == "error"
        else None
    )
    st.session_state.job_started_at = None
    st.session_state.job_session_id = None
    st.rerun()


_live_audit_monitor()

quota_remaining = quota_data.get("remaining", 0) if quota_data else 0
quota_exhausted = quota_data is not None and quota_remaining <= 0
quota_unavailable = quota_error is not None or quota_data is None
job_running = bool(st.session_state.job_session_id)
if quota_exhausted:
    st.warning(
        f"The free-audit limit of {quota_data['limit']} completed audits has been reached."
    )
elif quota_unavailable:
    st.warning("Audit submission is disabled until the quota service is available.")

prompt = st.chat_input(
    "Paste a YouTube lecture URL or specify a chapter...",
    disabled=job_running or quota_exhausted or quota_unavailable,
)
if prompt:
    if not st.session_state.selected_doc_id:
        st.error("Choose an indexed reference book or upload a PDF first.")
        st.stop()

    has_new_url = (
        "youtube.com/" in prompt.lower() or "youtu.be/" in prompt.lower()
    ) and bool(st.session_state.messages)
    if has_new_url:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.active_video_url = None
        st.session_state.active_video_id = None
        st.session_state.show_inline_transcript = False
        st.session_state.awaiting_scope = False

    user_message: Dict[str, Any] = {"role": "user", "content": prompt}
    for token in prompt.split():
        if "youtube.com/" in token.lower() or "youtu.be/" in token.lower():
            clean_url = token.strip("()[]<>.,")
            st.session_state.active_video_url = clean_url
            user_message["video_url"] = clean_url
            break
    st.session_state.messages.append(user_message)

    payload = {
        "session_id": st.session_state.session_id,
        "user_id": visitor_id,
        "doc_id": st.session_state.selected_doc_id,
        "message": prompt,
        "video_url": (
            st.session_state.active_video_url
            if st.session_state.awaiting_scope
            else None
        ),
        "prior_clarification_asked": st.session_state.awaiting_scope,
    }
    try:
        response = requests.post(
            f"{API_BASE}/chat/turn/async", json=payload, timeout=15
        )
    except requests.RequestException as exc:
        st.session_state.messages.pop()
        st.error(f"Could not start the audit: {exc}")
        st.stop()
    if not response.ok:
        st.session_state.messages.pop()
        st.error(f"Could not start the audit: {response.text}")
        st.stop()

    st.session_state.awaiting_scope = not st.session_state.awaiting_scope
    st.session_state.job_session_id = st.session_state.session_id
    st.session_state.job_started_at = time.perf_counter()
    st.session_state.job_error = None
    st.rerun()
