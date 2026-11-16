from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import desc, func, select
from app.services.db.database import get_db_session
from app.services.db.models import (
    ApprovalItem,
    AuditLog,
    AuditRun,
    ChatMessage,
    DocumentRecord,
    FeedbackRecord,
)


# --- Documents ---
def save_document_record(
    doc_id: str,
    doc_name: str,
    file_path: str,
    page_count: int = 0,
    chunk_count: int = 0,
    sections: Optional[List[Dict[str, Any]]] = None,
    uploaded_by: str = "default_user",
    status: str = "ready",
) -> DocumentRecord:
    with get_db_session() as db:
        existing = db.get(DocumentRecord, doc_id)
        if existing:
            existing.doc_name = doc_name
            existing.file_path = file_path
            existing.page_count = page_count
            existing.chunk_count = chunk_count
            existing.sections_json = sections or []
            existing.status = status
            db.flush()
            return existing

        rec = DocumentRecord(
            doc_id=doc_id,
            doc_name=doc_name,
            file_path=file_path,
            page_count=page_count,
            chunk_count=chunk_count,
            sections_json=sections or [],
            status=status,
            uploaded_by=uploaded_by,
        )
        db.add(rec)
        db.flush()
        return rec


def list_documents() -> List[Dict[str, Any]]:
    with get_db_session() as db:
        rows = db.scalars(
            select(DocumentRecord).order_by(desc(DocumentRecord.created_at))
        ).all()
        return [
            {
                "doc_id": r.doc_id,
                "doc_name": r.doc_name,
                "page_count": r.page_count,
                "chunk_count": r.chunk_count,
                "sections": r.sections_json or [],
                "status": r.status,
                "uploaded_by": r.uploaded_by,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    with get_db_session() as db:
        r = db.get(DocumentRecord, doc_id)
        if not r:
            return None
        return {
            "doc_id": r.doc_id,
            "doc_name": r.doc_name,
            "file_path": r.file_path,
            "page_count": r.page_count,
            "chunk_count": r.chunk_count,
            "sections": r.sections_json or [],
            "status": r.status,
        }


# --- Audit Runs & Immutable Logs ---
def log_audit_event(
    action: str,
    detail: Dict[str, Any],
    run_id: Optional[str] = None,
    user_id: str = "default_user",
) -> None:
    with get_db_session() as db:
        db.add(AuditLog(run_id=run_id, user_id=user_id, action=action, detail=detail))


def save_audit_run(payload: Dict[str, Any]) -> str:
    with get_db_session() as db:
        run = AuditRun(**payload)
        db.add(run)
        db.flush()
        return run.run_id


# --- Chat Messages & Session History ---
def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    user_id: str = "default_user",
    video_url: str = "",
    citations: Optional[List[Dict[str, Any]]] = None,
    run_id: Optional[str] = None,
) -> str:
    with get_db_session() as db:
        msg = ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            video_url=video_url,
            citations_json=citations or [],
            run_id=run_id,
        )
        db.add(msg)
        db.flush()
        return msg.id


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    with get_db_session() as db:
        rows = db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        ).all()
        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "video_url": r.video_url,
                "citations": r.citations_json or [],
                "run_id": r.run_id,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


def count_completed_audits(user_id: str) -> int:
    with get_db_session() as db:
        count = db.scalar(
            select(func.count(AuditLog.id)).where(
                AuditLog.user_id == user_id,
                AuditLog.action == "audit_completed",
            )
        )
        return int(count or 0)


def list_recent_sessions(
    limit: int = 15, user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    with get_db_session() as db:
        stmt = select(AuditRun).order_by(desc(AuditRun.created_at))
        if user_id is not None:
            stmt = stmt.where(AuditRun.user_id == user_id)
        rows = db.scalars(stmt.limit(limit)).all()
        return [
            {
                "session_id": r.session_id,
                "run_id": r.run_id,
                "video_title": r.video_title or r.video_id,
                "target_scope": r.target_scope,
                "evaluator_status": r.evaluator_status,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


# --- HITL Approvals Queue ---
def enqueue_approval(item: Dict[str, Any]) -> str:
    with get_db_session() as db:
        rec = ApprovalItem(**item)
        db.add(rec)
        db.flush()
        return rec.id


def list_approvals(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_db_session() as db:
        stmt = select(ApprovalItem).order_by(desc(ApprovalItem.created_at))
        if status_filter:
            stmt = stmt.where(ApprovalItem.status == status_filter)
        rows = db.scalars(stmt).all()
        return [
            {
                "id": r.id,
                "run_id": r.run_id,
                "doc_id": r.doc_id,
                "topic_title": r.topic_title,
                "book_snippet": r.book_snippet,
                "page_number": r.page_number,
                "matched_timestamp": r.matched_timestamp,
                "video_snippet": r.video_snippet,
                "similarity_score": r.similarity_score,
                "evaluator_notes": r.evaluator_notes,
                "status": r.status,
                "reviewed_by": r.reviewed_by,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


def resolve_approval(
    approval_id: str, decision: str, reviewer: str = "teacher"
) -> bool:
    """decision: 'confirmed_gap' or 'marked_covered'"""
    with get_db_session() as db:
        rec = db.get(ApprovalItem, approval_id)
        if not rec:
            return False
        rec.status = decision
        rec.reviewed_by = reviewer
        rec.reviewed_at = datetime.now(timezone.utc)
        db.add(
            AuditLog(
                run_id=rec.run_id,
                user_id=reviewer,
                action="hitl_decision",
                detail={
                    "approval_id": approval_id,
                    "topic": rec.topic_title,
                    "decision": decision,
                },
            )
        )
        return True


# --- Feedback & Live Metrics ---
def save_feedback(
    message_id: str, rating: int, user_id: str = "default_user", comment: str = ""
) -> str:
    with get_db_session() as db:
        fb = FeedbackRecord(
            message_id=message_id, user_id=user_id, rating=rating, comment=comment
        )
        db.add(fb)
        db.flush()
        return fb.id


def get_system_metrics() -> Dict[str, Any]:
    with get_db_session() as db:
        runs = db.scalars(select(AuditRun)).all()
        approvals = db.scalars(select(ApprovalItem)).all()
        feedbacks = db.scalars(select(FeedbackRecord)).all()

        total_runs = len(runs)
        first_pass = sum(
            1 for r in runs if r.evaluator_status == "passed" and r.retry_count == 0
        )
        repaired = sum(
            1
            for r in runs
            if r.retry_count > 0 and r.evaluator_status in {"passed", "repaired"}
        )
        escalated_run_ids = {approval.run_id for approval in approvals}
        escalated = sum(
            1
            for r in runs
            if r.evaluator_status == "escalated" or r.run_id in escalated_run_ids
        )
        avg_latency = (
            round(sum(r.latency_ms for r in runs) / total_runs, 1)
            if total_runs
            else 0.0
        )

        helpful = sum(1 for f in feedbacks if f.rating > 0)
        flagged = sum(1 for f in feedbacks if f.rating < 0)

        return {
            "total_audits": total_runs,
            "first_pass_count": first_pass,
            "repaired_count": repaired,
            "escalated_count": escalated,
            "avg_latency_ms": avg_latency,
            "pending_approvals": sum(
                1 for a in approvals if a.status == "needs_human_review"
            ),
            "resolved_approvals": sum(
                1 for a in approvals if a.status != "needs_human_review"
            ),
            "helpful_feedback": helpful,
            "flagged_feedback": flagged,
        }
