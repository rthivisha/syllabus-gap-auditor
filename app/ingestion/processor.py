import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.config import get_settings
from app.ingestion.chunking.splitter import build_hierarchical_chunks
from app.ingestion.loaders.pdf_loader import load_pdf_sections
from app.ingestion.loaders.video_loader import extract_video_id, parse_video_source
from app.services.db.repository import log_audit_event, save_document_record
from app.services.retrieval.pinecone_service import PineconeService

PDF_INDEX_VERSION = 2


def _compute_doc_id(file_path: Path) -> str:
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() else "-" for ch in file_path.stem.lower()).strip(
        "-"
    )
    return f"{stem[:36]}-{digest}"


def ingest_pdf_document(
    pdf_path: Path,
    uploaded_by: str = "default_user",
    suggested_questions_generator: Optional[
        Callable[[str, List[Dict[str, Any]]], List[str]]
    ] = None,
) -> Dict[str, Any]:
    """
    Step 0 Instant PDF Ingestion:
      1. Writes immediate `status='processing'` row so UI shows it right away.
      2. Parses sections & page spans via PyMuPDF.
      3. Builds Hierarchical Parent-Child chunks & upserts to Pinecone Serverless.
      4. Updates Neon PostgreSQL row to `status='ready'` (or `'failed'` with error details).
    """
    settings = get_settings()
    doc_id = _compute_doc_id(pdf_path)
    doc_name = pdf_path.name

    save_document_record(
        doc_id=doc_id,
        doc_name=doc_name,
        file_path=str(pdf_path),
        uploaded_by=uploaded_by,
        status="processing",
        index_version=PDF_INDEX_VERSION,
    )
    print(f"[INGEST START] {doc_name} (doc_id={doc_id})")

    try:
        parsed = load_pdf_sections(pdf_path)
        sections = parsed["sections"]
        page_count = parsed["page_count"]

        child_chunks = build_hierarchical_chunks(doc_id=doc_id, sections=sections)
        section_summaries = [
            {
                "section_idx": s["section_idx"],
                "section_title": s["section_title"],
                "start_page": s["start_page"],
                "end_page": s["end_page"],
                "summary": s["summary"],
            }
            for s in sections
        ]

        pc_service = PineconeService()
        vector_count = pc_service.upsert_document_vectors(
            doc_id=doc_id,
            doc_name=doc_name,
            child_chunks=child_chunks,
            section_summaries=section_summaries,
        )
        suggested_questions = (
            suggested_questions_generator(doc_name, child_chunks)
            if suggested_questions_generator
            else []
        )

        out_json = settings.PROCESSED_BOOKS_DIR / f"{doc_id}.json"
        out_json.write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "index_version": PDF_INDEX_VERSION,
                    "page_count": page_count,
                    "sections": section_summaries,
                    "chunk_count": len(child_chunks),
                    "suggested_questions": suggested_questions,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        save_document_record(
            doc_id=doc_id,
            doc_name=doc_name,
            file_path=str(pdf_path),
            page_count=page_count,
            chunk_count=len(child_chunks),
            sections=section_summaries,
            suggested_questions=suggested_questions,
            uploaded_by=uploaded_by,
            status="ready",
            index_version=PDF_INDEX_VERSION,
        )
        log_audit_event(
            action="document_ingested",
            detail={
                "doc_id": doc_id,
                "doc_name": doc_name,
                "pages": page_count,
                "vectors": vector_count,
            },
            user_id=uploaded_by,
        )
        print(
            f"[INGEST READY] {doc_name} -> {page_count} pages, {vector_count} vectors"
        )

        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "index_version": PDF_INDEX_VERSION,
            "page_count": page_count,
            "section_count": len(sections),
            "chunk_count": len(child_chunks),
            "vector_count": vector_count,
            "sections": section_summaries,
            "suggested_questions": suggested_questions,
            "status": "ready",
        }
    except Exception as exc:
        print(f"[INGEST ERROR] {doc_name}: {exc}")
        save_document_record(
            doc_id=doc_id,
            doc_name=doc_name,
            file_path=str(pdf_path),
            sections=[
                {"section_title": f"ERROR: {exc}", "start_page": 0, "end_page": 0}
            ],
            uploaded_by=uploaded_by,
            status="failed",
            index_version=PDF_INDEX_VERSION,
        )
        raise

def get_or_parse_video(source: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Step 2 Multimodal Video Ingestion backed by fast local SQLite cache."""
    settings = get_settings()
    video_id = extract_video_id(source)

    with sqlite3.connect(settings.CACHE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_cache (
                video_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        if not force_refresh:
            row = conn.execute(
                "SELECT payload_json FROM video_cache WHERE video_id = ?", (video_id,)
            ).fetchone()
            if row:
                data = json.loads(row[0])
                cached_windows = data.get("windows") or []
                cached_chars = sum(
                    len(str(w.get("combined_text") or "").strip()) for w in cached_windows
                )
                if not data.get("passed_guard") or cached_chars >= 120:
                    data["from_cache"] = True
                    return data

    parsed = parse_video_source(source)
    parsed["from_cache"] = False

    windows = parsed.get("windows") or []
    transcript_chars = sum(
        len(str(w.get("combined_text") or "").strip()) for w in windows
    )
    if parsed.get("passed_guard") and transcript_chars < 120:
        parsed["passed_guard"] = False
        parsed["guard_reason"] = (
            "YouTube captions could not be fetched from this server (captions disabled or cloud IP blocked). "
            "Try a video with English captions enabled."
        )

    if parsed.get("passed_guard"):
        payload_str = json.dumps(parsed, indent=2)
        with sqlite3.connect(settings.CACHE_DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO video_cache (video_id, payload_json) VALUES (?, ?)",
                (video_id, payload_str),
            )
        (settings.PROCESSED_VIDEOS_DIR / f"video_{video_id}.json").write_text(
            payload_str, encoding="utf-8"
        )

    return parsed
