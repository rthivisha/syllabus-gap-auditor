from typing import Any, Dict, List
from app.agents.state import AuditGraphState
from app.config import get_settings
from app.services.db.repository import enqueue_approval, log_audit_event, save_audit_run


def finalize_and_escalate_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Step 5 & 6 Finalizer:
      - Separates high-confidence missed topics from unresolved max-retry borderline items (0.42 - 0.55).
      - Routes borderline items to PostgreSQL `approvals` (status='needs_human_review').
      - Prioritizes verbatim chunks matching Confirmed Missed Topics pages first for verified sources.
      - Persists complete `audit_runs` & `audit_logs` records in Neon PostgreSQL.
    """
    settings = get_settings()
    draft = (
        state.get("drafted_gaps")
        or state.get("draft_missed_topics")
        or state.get("missed_topics")
        or []
    )
    verdict = state.get("evaluator_verdict", "FAIL")
    attempt = state.get("attempt", 0)
    flagged_set = {t.lower() for t in state.get("flagged_topics", [])}
    eval_passed = verdict == "PASS"
    final_verdict = "passed" if eval_passed else "failed"

    confirmed_missed: List[Dict[str, Any]] = []
    escalated_items: List[Dict[str, Any]] = []

    for item in draft:
        conf = float(item.get("confidence", 0.8))
        is_borderline = verdict == "FAIL" and (
            item["topic_title"].lower() in flagged_set
            or (settings.AMBIGUITY_LOW <= conf <= settings.AMBIGUITY_HIGH)
        )
        if is_borderline:
            escalated_items.append(item)
        else:
            confirmed_missed.append(item)

    missed_gaps = state.get("drafted_gaps") or state.get("missed_topics") or confirmed_missed or []
    all_chunks = state.get("retrieved_evidence") or state.get("cited_chunks") or []

    if missed_gaps and all_chunks:
        gap_pages = [int(g["page_number"]) for g in missed_gaps if g.get("page_number") is not None]
        page_order = {pg: idx for idx, pg in enumerate(dict.fromkeys(gap_pages))}
        matched = [c for c in all_chunks if int(c.get("page_number") or 0) in page_order]
        matched.sort(key=lambda c: page_order[int(c.get("page_number") or 0)])
        remaining = [c for c in all_chunks if c not in matched]
        prioritized_sources = (matched + remaining)[:6]
    else:
        prioritized_sources = all_chunks[:6]

    enriched_sources = []
    for c in prioritized_sources:
        c_dict = dict(c)
        txt = (
            c_dict.get("text")
            or c_dict.get("evidence_quote")
            or c_dict.get("snippet")
            or c_dict.get("parent_text")
            or c_dict.get("summary")
            or ""
        )
        c_dict["text"] = txt
        c_dict["snippet"] = txt[:600]
        c_dict["evidence_quote"] = c_dict.get("evidence_quote") or txt[:140]
        enriched_sources.append(c_dict)

    doc_name = state.get("doc_name") or "Uploaded Document"
    if len(enriched_sources) < 3 and missed_gaps:
        for idx, g in enumerate(missed_gaps):
            if len(enriched_sources) >= 6:
                break
            txt = (
                g.get("evidence_quote")
                or g.get("summary")
                or g.get("description")
                or g.get("explanation")
                or ""
            )
            if txt:
                enriched_sources.append(
                    {
                        "chunk_id": g.get("chunk_id", f"gap_{idx}"),
                        "doc_name": doc_name,
                        "section_title": g.get("section_title", ""),
                        "page_number": g.get("page_number", 1),
                        "text": txt,
                        "snippet": txt[:600],
                        "evidence_quote": txt[:140],
                        "score": 0.45,
                    }
                )
    prioritized_sources = enriched_sources

    raw_scoped = state.get("scoped_section_titles") or []
    scope_label = (
        ", ".join(raw_scoped)
        if raw_scoped
        else f"Full Document ({doc_name})"
    )

    lines = [
        f"### 📋 Gap Audit Results — `{scope_label}`",
        f"**Evaluator Status:** `{'PASSED' if eval_passed else 'FAILED'}` (Retries: `{attempt}`) · **Missed Topics Found:** `{len(missed_gaps)}`",
        "",
    ]

    for idx, gap in enumerate(missed_gaps, start=1):
        title = gap.get("topic_title") or gap.get("section_title") or f"Topic {idx}"
        sec = gap.get("section_title", "")
        pg = gap.get("page_number", "?")
        desc = gap.get("description") or gap.get("explanation") or gap.get("summary") or ""
        vid_note = gap.get("closest_timestamp") or gap.get("video_check_note") or "Not covered in video transcript"

        lines.append(f"**{idx}. {title}**  ·  `Page {pg}` *({sec})*")
        lines.append(f"- 📄 **Missing from Video:** {desc}")
        lines.append(f"- 🎥 **Closest Video Check:** {vid_note}")
        lines.append("")

    lines.append(
        "💡 *Would you like a quick breakdown of any missed topic above, or a ready-to-record script covering all of them?*"
    )
    chat_response = "\n".join(lines)

    run_id = save_audit_run(
        {
            "session_id": state["session_id"],
            "user_id": state.get("user_id", "default_user"),
            "doc_id": state["doc_id"],
            "video_id": state.get("video_id", ""),
            "video_url": state["video_url"],
            "video_title": state.get("video_title", ""),
            "target_scope": state.get("target_scope", "Entire Document"),
            "evaluator_status": final_verdict,
            "retry_count": attempt,
            "latency_ms": state.get("latency_ms", 0),
            "missed_topics_json": confirmed_missed,
            "cited_chunks_json": prioritized_sources,
            "evaluation_notes": state.get("evaluation_notes", ""),
            "repair_notes": state.get("repair_notes", ""),
        }
    )

    for esc in escalated_items:
        enqueue_approval(
            {
                "run_id": run_id,
                "doc_id": state["doc_id"],
                "topic_title": esc["topic_title"],
                "book_snippet": esc["summary"],
                "page_number": int(esc.get("page_number", 1)),
                "matched_timestamp": esc.get("closest_video_timestamp", ""),
                "video_snippet": esc.get("closest_video_snippet", ""),
                "similarity_score": float(esc.get("confidence", 0.5)),
                "evaluator_notes": state.get("evaluation_notes", ""),
                "status": "needs_human_review",
            }
        )

    log_audit_event(
        action="audit_completed",
        run_id=run_id,
        user_id=state.get("user_id", "default_user"),
        detail={
            "status": final_verdict,
            "retries": attempt,
            "confirmed_gaps": len(confirmed_missed),
            "escalated_hitl": len(escalated_items),
        },
    )

    return {
        "run_id": run_id,
        "chat_response": chat_response,
        "final_missed_topics": confirmed_missed,
        "escalated_items": escalated_items,
        "final_status": final_verdict,
        "route_taken": "audit_pipeline",
        "awaiting_video": False,
        "retrieved_sections": state.get("retrieved_sections", []),
        "retrieved_evidence": prioritized_sources,
        "cited_chunks": prioritized_sources,
    }


__all__ = ["finalize_and_escalate_node"]
