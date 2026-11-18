from typing import Any, Dict, List, Optional
from app.services.db.repository import save_document_record


def register_document(
    doc_id: str,
    filename: str,
    page_count: int = 0,
    chunk_count: int = 0,
    section_titles: Optional[List[str]] = None,
    status: str = "ready",
    uploaded_by: str = "default_user",
    suggested_questions: Optional[List[str]] = None,
) -> Any:
    sections = [
        {"section_title": title, "title": title, "start_page": 1, "page_start": 1}
        for title in (section_titles or [])
    ]
    return save_document_record(
        doc_id=doc_id,
        doc_name=filename,
        file_path=filename,
        page_count=page_count,
        chunk_count=chunk_count,
        sections=sections,
        uploaded_by=uploaded_by,
        status=status,
        suggested_questions=suggested_questions,
    )


__all__ = ["register_document"]
