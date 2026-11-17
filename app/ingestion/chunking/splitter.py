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


def _coalesce_child_chunks(
    child_chunks: List[Dict[str, Any]], max_chars: int = 1500
) -> List[Dict[str, Any]]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive.")

    merged: List[Dict[str, Any]] = []
    for chunk in child_chunks:
        text = str(chunk.get("child_text") or "").strip()
        if not text:
            continue

        parts: List[str] = []
        while len(text) > max_chars:
            split_at = text.rfind(" ", 0, max_chars + 1)
            if split_at <= 0:
                split_at = max_chars
            parts.append(text[:split_at].strip())
            text = text[split_at:].strip()
        if text:
            parts.append(text)

        for part_idx, part in enumerate(parts):
            part_chunk = dict(chunk)
            if part_idx:
                part_chunk["chunk_id"] = f"{chunk['chunk_id']}-part{part_idx + 1}"
            part_chunk["child_text"] = part

            previous = merged[-1] if merged else None
            combined = (
                f"{previous['child_text']}\n{part}" if previous is not None else part
            )
            if (
                previous is not None
                and previous["section_title"] == part_chunk["section_title"]
                and previous["page_number"] == part_chunk["page_number"]
                and len(combined) <= max_chars
            ):
                previous["child_text"] = combined
                if previous["parent_id"] != part_chunk["parent_id"]:
                    previous["parent_text"] = combined
            else:
                merged.append(part_chunk)

    return merged


def build_hierarchical_chunks(
    doc_id: str,
    sections: List[Dict[str, Any]],
    parent_max_words: int = 650,
    child_max_words: int = 140,
    child_overlap_sentences: int = 1,
) -> List[Dict[str, Any]]:
    """
    Produces Hierarchical Parent-Child Chunks:
      - Parent (~650 words / ~900 tokens): Full concept/example context fed to the Auditor LLM.
      - Child (~140 words / ~180 tokens): Granular vector chunk embedded in Pinecone with parent_text attached.
    """
    child_records: List[Dict[str, Any]] = []
    parent_counter = 0

    for sec in sections:
        sec_title = sec["section_title"]
        pages = sec.get("pages", [])
        if not pages:
            continue

        # Flatten sentences across pages of this section while preserving exact page numbers
        sentence_stream: List[tuple[int, str]] = []
        for p in pages:
            for sent in _split_sentences(p["text"]):
                sentence_stream.append((p["page_number"], sent))

        if not sentence_stream:
            continue

        # 1. Group sentence stream into Parent Chunks (~650 words)
        parents: List[Dict[str, Any]] = []
        cur_sents: List[tuple[int, str]] = []
        cur_words = 0
        start_page = sentence_stream[0][0]

        for page_num, sent in sentence_stream:
            w = len(sent.split())
            if cur_sents and (cur_words + w) > parent_max_words:
                parent_counter += 1
                parents.append(
                    {
                        "parent_id": f"{doc_id}-p{parent_counter:04d}",
                        "section_title": sec_title,
                        "page_number": start_page,
                        "parent_text": f"[{sec_title} | Page {start_page}]\n"
                        + " ".join(sentence for _, sentence in cur_sents),
                        "sentences": list(cur_sents),
                    }
                )
                cur_sents = [(page_num, sent)]
                cur_words = w
                start_page = page_num
            else:
                cur_sents.append((page_num, sent))
                cur_words += w

        if cur_sents:
            parent_counter += 1
            parents.append(
                {
                    "parent_id": f"{doc_id}-p{parent_counter:04d}",
                    "section_title": sec_title,
                    "page_number": start_page,
                    "parent_text": f"[{sec_title} | Page {start_page}]\n"
                    + " ".join(sentence for _, sentence in cur_sents),
                    "sentences": list(cur_sents),
                }
            )

        # 2. Split each Parent Chunk into overlapping Child Chunks (~140 words)
        for p_obj in parents:
            sents = p_obj["sentences"]
            idx = 0
            c_idx = 0
            while idx < len(sents):
                chunk_sents: List[tuple[int, str]] = []
                w_count = 0
                j = idx
                while j < len(sents):
                    if chunk_sents and sents[j][0] != chunk_sents[0][0]:
                        break
                    sw = len(sents[j][1].split())
                    if chunk_sents and (w_count + sw) > child_max_words:
                        break
                    chunk_sents.append(sents[j])
                    w_count += sw
                    j += 1

                c_idx += 1
                child_text = f"{sec_title}: " + " ".join(
                    sentence for _, sentence in chunk_sents
                )
                child_records.append(
                    {
                        "chunk_id": f"{p_obj['parent_id']}-c{c_idx:02d}",
                        "parent_id": p_obj["parent_id"],
                        "section_title": sec_title,
                        "page_number": chunk_sents[0][0],
                        "child_text": child_text,
                        "parent_text": p_obj["parent_text"],
                    }
                )

                if j >= len(sents):
                    break
                idx = max(idx + 1, j - child_overlap_sentences)

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
