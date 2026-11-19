import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.services.db.database import Base

# Use cross-database JSON type (JSONB on Postgres, JSON on SQLite)
JsonType = JSON().with_variant(JSONB(), "postgresql")
JSON_TYPE = JsonType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class DocumentRecord(Base):
    """Uploaded reference PDFs/books indexed into Pinecone Serverless."""

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    doc_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    index_version: Mapped[int] = mapped_column(
        Integer, default=1, server_default=text("1"), nullable=False
    )
    sections_json: Mapped[list] = mapped_column(
        JSON_TYPE, default=list
    )  # Extracted TOC / section titles + page spans
    suggested_questions_json: Mapped[list] = mapped_column(
        JsonType, default=list, server_default=text("'[]'")
    )
    status: Mapped[str] = mapped_column(String(32), default="ready")
    uploaded_by: Mapped[str] = mapped_column(String(100), default="default_user")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class AuditRun(Base):
    """High-level record of each LangGraph Syllabus/Book Gap Audit run."""

    __tablename__ = "audit_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(100), default="default_user")
    doc_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.doc_id"), nullable=False
    )
    video_id: Mapped[str] = mapped_column(String(64), nullable=False)
    video_url: Mapped[str] = mapped_column(Text, nullable=False)
    video_title: Mapped[str] = mapped_column(String(255), default="")
    target_scope: Mapped[str] = mapped_column(String(255), default="full_document")

    evaluator_status: Mapped[str] = mapped_column(
        String(32), default="passed"
    )  # passed | failed
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    missed_topics_json: Mapped[list] = mapped_column(JSON_TYPE, default=list)
    cited_chunks_json: Mapped[list] = mapped_column(JSON_TYPE, default=list)
    evaluation_notes: Mapped[str] = mapped_column(Text, default="")
    repair_notes: Mapped[str] = mapped_column(Text, default="")
    langsmith_run_url: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class AuditLog(Base):
    """Step-by-step immutable audit trail (ingestion, retrieval, evaluator, repair, HITL)."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String(64), index=True, nullable=True)
    user_id: Mapped[str] = mapped_column(String(100), default="default_user")
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class ChatMessage(Base):
    """Persistent conversational history (user prompts, assistant replies, inline video & citations)."""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(100), default="default_user")
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    video_url: Mapped[str] = mapped_column(Text, default="")
    citations_json: Mapped[list] = mapped_column(JSON_TYPE, default=list)
    run_id: Mapped[str] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class ApprovalItem(Base):
    """HITL Escalation Queue for Max-Retry Low-Confidence borderline matches (0.42 - 0.55)."""

    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("audit_runs.run_id"), index=True
    )
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False)
    topic_title: Mapped[str] = mapped_column(String(255), nullable=False)
    book_snippet: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=1)
    matched_timestamp: Mapped[str] = mapped_column(String(16), default="")
    video_snippet: Mapped[str] = mapped_column(Text, default="")
    similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    evaluator_notes: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(
        String(32), default="needs_human_review"
    )  # needs_human_review | confirmed_gap | marked_covered
    reviewed_by: Mapped[str] = mapped_column(String(100), default="")
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class FeedbackRecord(Base):
    """User thumbs-up / thumbs-down feedback logged from the chat UI."""

    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    message_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chat_messages.id"), index=True
    )
    user_id: Mapped[str] = mapped_column(String(100), default="default_user")
    rating: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # 1 = Helpful, -1 = Flagged
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
