import re
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import AuditGraphState
from app.gateway.client import get_fast_groq_llm

PROJECT_DOC_PATH = Path(__file__).resolve().parents[3] / "project_context.md"


def _select_relevant_md_sections(md_text: str, query: str) -> str:
    """Return only the Markdown sections relevant to the user's query."""
    sections = re.split(r"(?=^##\s+)", md_text, flags=re.M)
    query_lower = query.lower()
    routes = (
        (
            "stack",
            (
                "stack",
                "model",
                "pinecone",
                "database",
                "tech",
                "framework",
                "provider",
                "technology",
            ),
        ),
        (
            "start",
            (
                "start",
                "begin",
                "use",
                "usage",
                "workflow",
                "step",
                "upload",
                "video",
                "url",
                "audit",
            ),
        ),
        (
            "efficien",
            ("fast", "efficien", "latency", "cache", "optim"),
        ),
        (
            "overview",
            ("what", "about", "overview", "purpose", "architecture", "project", "system"),
        ),
    )
    selected_route = next(
        (
            section_route
            for section_route, keywords in routes
            if any(keyword in query_lower for keyword in keywords)
        ),
        "overview",
    )

    selected_sections = []
    for section in sections:
        header_match = re.search(r"^##\s+(.+)$", section, flags=re.M)
        if header_match and selected_route in header_match.group(1).lower():
            selected_sections.append(section.strip())

    if selected_sections:
        return "\n\n".join(selected_sections)

    fallback_section = next(
        (
            section.strip()
            for section in sections
            if re.search(r"^##\s+", section, re.M)
        ),
        "",
    )
    return fallback_section or md_text.split("\n\n", maxsplit=1)[0].strip()


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content)


def project_guide_node(state: AuditGraphState) -> Dict[str, Any]:
    clean_query = re.sub(
        r"\[Mode:[^\]]+\]\s*", "", state.get("user_query", "")
    ).strip()
    active_doc = state.get("doc_name")
    doc_status_hint = (
        f"Active reference PDF ready: **{active_doc}**."
        if active_doc
        else "No reference PDF is selected yet; remind the user to select a sample book or upload a PDF first."
    )

    if re.match(
        r"^(?:hi+|hello+|hey+|yo|good\s+(?:morning|afternoon|evening)|help|start)[\s!?.]*$",
        clean_query,
        re.I,
    ):
        return {
            "chat_response": (
                "Hi! 👋 **Syllabus Gap Audit AI** compares YouTube lectures against "
                "reference PDFs to find skipped topics.\n\n"
                "**Quick Start:**\n"
                "1. Upload a PDF or select an indexed sample book above.\n"
                "2. Paste a YouTube lecture URL here to run a timestamped gap audit.\n\n"
                f"👉 **Apply a YouTube video URL to try!** ({doc_status_hint})"
            ),
            "route_taken": "project_guide",
            "awaiting_video": True,
            "llm_used": False,
            "retrieved_evidence": [],
        }

    project_md = PROJECT_DOC_PATH.read_text(encoding="utf-8")
    relevant_md = _select_relevant_md_sections(project_md, clean_query)
    llm = get_fast_groq_llm(
        user_groq_key=state.get("user_groq_key"),
        user_gemini_key=state.get("user_gemini_key"),
    )
    resp = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are the interactive assistant for **Syllabus Gap Audit AI**. "
                    "Answer the user's question thoroughly and naturally using ONLY the relevant project documentation below. "
                    "Use clear bullet points and bold highlights for readability.\n\n"
                    "CRITICAL HANDOFF RULE:\n"
                    "Do NOT end the conversation passively. Always finish your response by inviting the user to test the live system "
                    "with a clear call-to-action asking them to **apply a YouTube video URL to try** (and mention the active PDF status).\n\n"
                    f"=== RELEVANT PROJECT DOCUMENTATION ===\n{relevant_md}"
                )
            ),
            HumanMessage(content=f"User message: {clean_query}\n{doc_status_hint}"),
        ]
    )

    answer = _extract_text(resp.content)
    if not answer:
        raise RuntimeError("The project guide model returned an empty response.")
    has_youtube_call_to_action = bool(
        re.search(
            r"\b(?:apply|paste|share|send)\b.{0,100}\byoutube\b"
            r"|\byoutube\b.{0,100}\b(?:apply|paste|share|send)\b",
            answer,
            re.I,
        )
    )
    mentions_reference_status = bool(
        re.search(r"\b(?:reference|pdf|document)\b", answer, re.I)
    )
    if not has_youtube_call_to_action or not mentions_reference_status:
        answer += (
            f"\n\n---\n👉 **Next step:** {doc_status_hint} "
            "Paste a YouTube lecture URL to run the audit."
        )

    return {
        "chat_response": answer,
        "route_taken": "project_guide",
        "awaiting_video": True,
        "llm_used": True,
        "retrieved_evidence": [],
    }
