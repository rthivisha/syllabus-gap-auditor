from typing import Any, Dict, Iterable, List, Optional, Tuple


CitationKey = Tuple[str, str, int]


def _page_number(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_citation_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: Dict[CitationKey, Dict[str, Any]] = {}
    for record in records:
        chunk_id = str(record.get("chunk_id") or record.get("id") or "")
        section_title = str(record.get("section_title") or "")
        page_number = _page_number(record.get("page_number"))
        if not chunk_id or not section_title or page_number is None:
            continue

        key = (chunk_id, section_title, page_number)
        existing = normalized.get(key, {})
        if len(str(record.get("parent_text") or record.get("summary") or "")) >= len(
            str(existing.get("parent_text") or existing.get("summary") or "")
        ):
            normalized[key] = {
                **existing,
                **record,
                "chunk_id": chunk_id,
                "section_title": section_title,
                "page_number": page_number,
            }
    return list(normalized.values())


def match_citation(
    item: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    *,
    allow_correction: bool = False,
) -> Optional[Dict[str, Any]]:
    chunk_id = str(item.get("chunk_id") or "")
    section_title = str(item.get("section_title") or "")
    page_number = _page_number(item.get("page_number"))

    if allow_correction and chunk_id:
        matches = [record for record in evidence if record["chunk_id"] == chunk_id]
        if len(matches) == 1:
            return matches[0]

    matches = [
        record
        for record in evidence
        if record["chunk_id"] == chunk_id
        and record["section_title"] == section_title
        and record["page_number"] == page_number
    ]
    if len(matches) == 1:
        return matches[0]

    if allow_correction:
        matches = [
            record
            for record in evidence
            if record["section_title"] == section_title
            and record["page_number"] == page_number
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def state_citation_records(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for key in ("retrieved_evidence", "retrieved_sections", "candidate_gaps", "cited_chunks"):
        records.extend(state.get(key) or [])
    return normalize_citation_records(records)
