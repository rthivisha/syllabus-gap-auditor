import re
from typing import Any, Dict, List, Optional


def _split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(])", cleaned)
        if s.strip()
    ]


def _extract_page_blocks(text: str, max_atomic_chars: int = 2800) -> List[Dict[str, Any]]:
    """
    Extracts atomic blocks from page text:
    - [TABLE]...[/TABLE] blocks are kept atomic (only force-split if exceeding max_atomic_chars)
    - [OCR_DIAGRAM_TEXT]... blocks are kept atomic (only force-split if exceeding max_atomic_chars)
    - Normal paragraphs are split on \\n\\n
    """
    if not text:
        return []

    # Regex to split on [TABLE]...[/TABLE] blocks while keeping them intact
    table_pattern = re.compile(r"(\[TABLE\][\s\S]*?\[/TABLE\])", re.IGNORECASE)
    parts = table_pattern.split(text)
    blocks: List[Dict[str, Any]] = []

    for part in parts:
        raw_part = part.strip()
        if not raw_part:
            continue

        if raw_part.upper().startswith("[TABLE]") and raw_part.upper().endswith("[/TABLE]"):
            # Table block: indivisible unit
            if len(raw_part) <= max_atomic_chars:
                blocks.append({
                    "text": raw_part,
                    "is_table": True,
                    "is_ocr": "[OCR_" in raw_part,
                })
            else:
                # Force-split oversized table exceeding 2,800 chars line-by-line
                lines = raw_part.splitlines()
                cur_t = ""
                for ln in lines:
                    if cur_t and (len(cur_t) + len(ln) + 1 > max_atomic_chars):
                        blocks.append({"text": cur_t.strip(), "is_table": True, "is_ocr": "[OCR_" in cur_t})
                        cur_t = ln
                    else:
                        cur_t = f"{cur_t}\n{ln}" if cur_t else ln
                if cur_t:
                    blocks.append({"text": cur_t.strip(), "is_table": True, "is_ocr": "[OCR_" in cur_t})
        else:
            # Regular text: split paragraphs on \n\n
            paras = [p.strip() for p in re.split(r"\n\s*\n", raw_part) if p.strip()]
            for p in paras:
                if len(p) <= max_atomic_chars:
                    blocks.append({
                        "text": p,
                        "is_table": "[TABLE]" in p,
                        "is_ocr": "[OCR_" in p,
                    })
                else:
                    # Force-split oversized paragraph on sentences
                    sents = _split_sentences(p)
                    cur_p = ""
                    for s in sents:
                        if cur_p and (len(cur_p) + len(s) + 1 > max_atomic_chars):
                            blocks.append({"text": cur_p.strip(), "is_table": False, "is_ocr": "[OCR_" in cur_p})
                            cur_p = s
                        else:
                            cur_p = f"{cur_p} {s}" if cur_p else s
                    if cur_p:
                        blocks.append({"text": cur_p.strip(), "is_table": False, "is_ocr": "[OCR_" in cur_p})

    return blocks


def _coalesce_child_chunks(
    child_chunks: List[Dict[str, Any]], max_chars: int = 2800
) -> List[Dict[str, Any]]:
    """Coalesces small adjacent chunks without slicing tables or exceeding max_chars."""
    merged: List[Dict[str, Any]] = []
    for chunk in child_chunks:
        if not merged:
            merged.append(dict(chunk))
            continue
        prev = merged[-1]
        # Merge adjacent small chunks from same section & page if under max_chars
        if (
            prev.get("section_title") == chunk.get("section_title")
            and prev.get("page_number") == chunk.get("page_number")
            and len(prev.get("child_text", "")) + len(chunk.get("child_text", "")) + 2 <= max_chars
            and len(prev.get("child_text", "")) < 1200
            and not (prev.get("has_tables") and chunk.get("has_tables"))
        ):
            combined = f"{prev['child_text']}\n\n{chunk['child_text']}"
            prev["child_text"] = combined
            prev["text"] = combined
            prev["has_tables"] = bool(prev.get("has_tables") or chunk.get("has_tables"))
            prev["ocr_applied"] = bool(prev.get("ocr_applied") or chunk.get("ocr_applied"))
        else:
            merged.append(dict(chunk))
    return merged


def build_hierarchical_chunks(
    doc_id: str,
    sections: List[Dict[str, Any]],
    parent_max_words: int = 650,
    child_max_words: int = 140,
    child_overlap_sentences: int = 1,
) -> List[Dict[str, Any]]:
    """
    Produces Hierarchical Parent-Child Chunks with atomic table and OCR preservation:
      - Parent (~650 words / ~900 tokens): Full concept context fed to Auditor LLM.
      - Child (1.5k–2.8k chars): Granular vector chunks embedded in Pinecone with parent_text attached.
      - Keeps [TABLE]...[/TABLE] and [OCR_DIAGRAM_TEXT]... blocks indivisible.
      - Preserves has_tables and ocr_applied provenance flags in metadata.
    """
    child_records: List[Dict[str, Any]] = []
    parent_counter = 0

    for sec in sections:
        sec_title = sec.get("section_title", "Overview")
        chapter = sec.get("chapter", sec_title)
        pages = sec.get("pages", [])
        if not pages:
            if sec.get("text"):
                pages = [{
                    "page_number": sec.get("start_page", 1),
                    "text": sec["text"],
                    "has_tables": "[TABLE]" in sec["text"],
                    "ocr_applied": "[OCR_" in sec["text"],
                }]
            else:
                continue

        # Extract atomic blocks across pages while preserving provenance
        block_stream: List[Dict[str, Any]] = []
        for p in pages:
            page_num = p.get("page_number", 1)
            page_has_tables = bool(p.get("has_tables", False))
            page_ocr = bool(p.get("ocr_applied", False))
            p_blocks = _extract_page_blocks(p.get("text", ""))
            for blk in p_blocks:
                block_stream.append({
                    "page_number": page_num,
                    "text": blk["text"],
                    "is_table": blk["is_table"] or page_has_tables or ("[TABLE]" in blk["text"]),
                    "is_ocr": blk["is_ocr"] or page_ocr or ("[OCR_" in blk["text"]),
                })

        if not block_stream:
            continue

        # 1. Group atomic block stream into Parent Chunks (~3,500-5,000 chars / ~650 words)
        parents: List[Dict[str, Any]] = []
        cur_parent_blocks: List[Dict[str, Any]] = []
        cur_parent_chars = 0
        parent_start_page = block_stream[0]["page_number"]

        for blk in block_stream:
            blk_len = len(blk["text"])
            if cur_parent_blocks and (cur_parent_chars + blk_len > 4000):
                parent_counter += 1
                parent_text = f"[{sec_title} | Page {parent_start_page}]\n" + "\n\n".join(
                    b["text"] for b in cur_parent_blocks
                )
                parents.append({
                    "parent_id": f"{doc_id}-p{parent_counter:04d}",
                    "section_title": sec_title,
                    "chapter": chapter,
                    "page_number": parent_start_page,
                    "parent_text": parent_text,
                    "blocks": list(cur_parent_blocks),
                })
                cur_parent_blocks = [blk]
                cur_parent_chars = blk_len
                parent_start_page = blk["page_number"]
            else:
                cur_parent_blocks.append(blk)
                cur_parent_chars += blk_len

        if cur_parent_blocks:
            parent_counter += 1
            parent_text = f"[{sec_title} | Page {parent_start_page}]\n" + "\n\n".join(
                b["text"] for b in cur_parent_blocks
            )
            parents.append({
                "parent_id": f"{doc_id}-p{parent_counter:04d}",
                "section_title": sec_title,
                "chapter": chapter,
                "page_number": parent_start_page,
                "parent_text": parent_text,
                "blocks": list(cur_parent_blocks),
            })

        # 2. Split each Parent Chunk into 1.5k–2.8k char Child Chunks keeping atomic blocks intact
        for p_obj in parents:
            p_blocks = p_obj["blocks"]
            c_idx = 0
            cur_child_blocks: List[Dict[str, Any]] = []
            cur_child_chars = 0

            for blk in p_blocks:
                blk_len = len(blk["text"])
                # If adding blk exceeds 2,500 chars and we already have content, flush child chunk
                if cur_child_blocks and (cur_child_chars + blk_len > 2500):
                    c_idx += 1
                    child_page = cur_child_blocks[0]["page_number"]
                    child_text = f"{sec_title}: " + "\n\n".join(b["text"] for b in cur_child_blocks)
                    has_tbl = any(b["is_table"] for b in cur_child_blocks)
                    has_ocr = any(b["is_ocr"] for b in cur_child_blocks)
                    child_records.append({
                        "chunk_id": f"{p_obj['parent_id']}-c{c_idx:02d}",
                        "parent_id": p_obj["parent_id"],
                        "doc_id": doc_id,
                        "section_title": sec_title,
                        "chapter": chapter,
                        "page_number": child_page,
                        "child_text": child_text,
                        "text": child_text,
                        "parent_text": p_obj["parent_text"],
                        "has_tables": has_tbl,
                        "ocr_applied": has_ocr,
                    })
                    cur_child_blocks = [blk]
                    cur_child_chars = blk_len
                else:
                    cur_child_blocks.append(blk)
                    cur_child_chars += blk_len

            if cur_child_blocks:
                c_idx += 1
                child_page = cur_child_blocks[0]["page_number"]
                child_text = f"{sec_title}: " + "\n\n".join(b["text"] for b in cur_child_blocks)
                has_tbl = any(b["is_table"] for b in cur_child_blocks)
                has_ocr = any(b["is_ocr"] for b in cur_child_blocks)
                child_records.append({
                    "chunk_id": f"{p_obj['parent_id']}-c{c_idx:02d}",
                    "parent_id": p_obj["parent_id"],
                    "doc_id": doc_id,
                    "section_title": sec_title,
                    "chapter": chapter,
                    "page_number": child_page,
                    "child_text": child_text,
                    "text": child_text,
                    "parent_text": p_obj["parent_text"],
                    "has_tables": has_tbl,
                    "ocr_applied": has_ocr,
                })

    return _coalesce_child_chunks(child_records)


def format_mmss(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def merge_multimodal_windows(
    transcript_segments: List[Dict[str, Any]],
    ocr_slides: List[Dict[str, Any]],
    window_sec: float = 25.0,
) -> List[Dict[str, Any]]:
    """
    Merges short Whisper speech segments and overlapping Slide OCR frames
    into coherent ~25-second multimodal windows ([MM:SS]).
    """
    if not transcript_segments:
        return []

    windows: List[Dict[str, Any]] = []
    buf_speech: List[str] = []
    buf_lps: List[float] = []
    win_start: Optional[float] = None
    win_end: float = 0.0

    def _flush(w_start: float, w_end: float, texts: List[str], lps: List[float]):
        slide_texts = [
            s["text"]
            for s in ocr_slides
            if (w_start - 3.0) <= float(s.get("sec", 0.0)) <= (w_end + 3.0)
        ]
        # Deduplicate identical slide strings inside window
        unique_slides = list(dict.fromkeys(slide_texts))
        speech_str = " ".join(texts).strip()
        slide_str = " | ".join(unique_slides).strip()

        combined = speech_str
        if slide_str:
            combined += f"\n[On-Screen Slide OCR]: {slide_str}"

        windows.append(
            {
                "start_sec": round(w_start, 2),
                "end_sec": round(w_end, 2),
                "timestamp": format_mmss(w_start),
                "speech_text": speech_str,
                "slide_ocr_text": slide_str,
                "combined_text": combined,
                "avg_logprob": round(sum(lps) / len(lps), 4) if lps else -0.25,
            }
        )

    for seg in transcript_segments:
        s_start = float(seg["start_sec"])
        s_end = float(seg["end_sec"])
        if win_start is None:
            win_start = s_start
        buf_speech.append(seg["text"])
        buf_lps.append(float(seg.get("avg_logprob", -0.25)))
        win_end = s_end

        if (win_end - win_start) >= window_sec:
            _flush(win_start, win_end, buf_speech, buf_lps)
            buf_speech, buf_lps, win_start = [], [], None

    if buf_speech and win_start is not None:
        _flush(win_start, win_end, buf_speech, buf_lps)

    return windows
