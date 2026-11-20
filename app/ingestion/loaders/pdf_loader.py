import io
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF


def _detect_recurring_boilerplate(raw_pages: List[str], threshold_ratio: float = 0.25) -> set[str]:
    """Identifies running headers/footers repeated across >= 25% of pages."""
    if len(raw_pages) < 4:
        return set()
    line_counts: Counter[str] = Counter()
    for text in raw_pages:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        edge_lines = set(lines[:2] + lines[-2:])
        for ln in edge_lines:
            if len(ln) <= 80 and not ln.isdigit():
                line_counts[ln.lower()] += 1
    min_hits = max(3, int(len(raw_pages) * threshold_ratio))
    return {ln for ln, cnt in line_counts.items() if cnt >= min_hits}


def _extract_markdown_tables(page: fitz.Page) -> Tuple[str, List[fitz.Rect]]:
    """Deterministically extracts native PDF tables as Markdown and returns their bounding boxes."""
    md_blocks: List[str] = []
    bboxes: List[fitz.Rect] = []
    try:
        finder = page.find_tables()
        for tbl in finder.tables:
            md = tbl.to_markdown().strip()
            if md and len(md) >= 25:
                md_blocks.append(f"\n[TABLE]\n{md}\n[/TABLE]\n")
                bboxes.append(fitz.Rect(tbl.bbox))
    except Exception:
        pass
    return "\n".join(md_blocks), bboxes


def _ocr_single_page_png(png_bytes: bytes) -> str:
    """Runs CPU-bound Tesseract OCR on a pre-rendered 150-DPI grayscale page PNG."""
    try:
        import pytesseract
        from PIL import Image

        windows_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.name == "nt" and Path(windows_tesseract).exists():
            pytesseract.pytesseract.tesseract_cmd = windows_tesseract

        img = Image.open(io.BytesIO(png_bytes))
        txt = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
        return re.sub(r"\n{3,}", "\n\n", txt).strip()
    except Exception:
        return ""


def load_pdf_sections(pdf_path: Path, max_ocr_pages: int = 25) -> Dict[str, Any]:
    """
    3-Tier Deterministic PDF Extraction:
    1. Fast PyMuPDF native text + recurring boilerplate stripping.
    2. Deterministic Markdown table extraction via page.find_tables().
    3. Parallel Tesseract OCR strictly on image-only / low-text pages (<180 chars).
    """
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)

    raw_texts = [doc[i].get_text("text") or "" for i in range(page_count)]
    boilerplate_lines = _detect_recurring_boilerplate(raw_texts)

    page_records: List[Dict[str, Any]] = []
    ocr_candidates: List[Tuple[int, bytes]] = []

    for idx in range(page_count):
        page = doc[idx]
        tables_md, table_rects = _extract_markdown_tables(page)

        # Extract text blocks outside table bounding boxes to avoid duplicate rows
        if table_rects:
            blocks = page.get_text("blocks")
            non_table_lines: List[str] = []
            for b in blocks:
                b_rect = fitz.Rect(b[:4])
                if any(b_rect.intersects(t_rect) for t_rect in table_rects):
                    continue
                non_table_lines.extend(str(b[4]).splitlines())
            lines = non_table_lines
        else:
            lines = raw_texts[idx].splitlines()

        cleaned_lines = [
            ln.strip()
            for ln in lines
            if ln.strip()
            and not ln.strip().isdigit()
            and ln.strip().lower() not in boilerplate_lines
        ]
        body_text = "\n".join(cleaned_lines).strip()
        combined = f"{body_text}\n{tables_md}".strip() if tables_md else body_text

        page_records.append(
            {
                "page_number": idx + 1,
                "text": combined,
                "has_tables": bool(tables_md),
                "ocr_applied": False,
            }
        )

        # Queue for selective parallel OCR only if page has images and minimal native text
        if (
            len(combined) < 180
            and len(page.get_images(full=False)) > 0
            and len(ocr_candidates) < max_ocr_pages
        ):
            pix = page.get_pixmap(dpi=150, colorspace=fitz.csGRAY, alpha=False)
            ocr_candidates.append((idx, pix.tobytes("png")))

    doc.close()

    # Run selective OCR in parallel threads (releases GIL in Tesseract C++)
    if ocr_candidates:
        with ThreadPoolExecutor(max_workers=min(4, len(ocr_candidates))) as pool:
            futures = {
                pool.submit(_ocr_single_page_png, png): page_idx
                for page_idx, png in ocr_candidates
            }
            for fut, page_idx in futures.items():
                ocr_txt = fut.result()
                if ocr_txt:
                    existing = page_records[page_idx]["text"]
                    page_records[page_idx]["text"] = (
                        f"{existing}\n[OCR_DIAGRAM_TEXT]\n{ocr_txt}".strip()
                        if existing
                        else ocr_txt
                    )
                    page_records[page_idx]["ocr_applied"] = True

    # Group pages into structured sections for build_hierarchical_chunks()
    sections: List[Dict[str, Any]] = []
    current_title = "Overview"
    current_pages: List[Dict[str, Any]] = []

    for rec in page_records:
        txt = rec["text"]
        if len(txt) < 60:
            continue

        first_lines = [
            ln.strip()
            for ln in txt.splitlines()[:5]
            if not ln.startswith("[TABLE]") and not ln.startswith("[OCR_")
        ]
        for candidate in first_lines:
            if (
                5 <= len(candidate) <= 85
                and not candidate.endswith(".")
                and "table of contents" not in candidate.lower()
            ):
                if current_pages and candidate != current_title:
                    sections.append(
                        {
                            "section_title": current_title,
                            "chapter": current_title,
                            "start_page": current_pages[0]["page_number"],
                            "end_page": current_pages[-1]["page_number"],
                            "pages": current_pages,
                            "text": "\n\n".join(p["text"] for p in current_pages),
                        }
                    )
                    current_pages = []
                current_title = candidate
                break

        current_pages.append(rec)

    if current_pages:
        sections.append(
            {
                "section_title": current_title,
                "chapter": current_title,
                "start_page": current_pages[0]["page_number"],
                "end_page": current_pages[-1]["page_number"],
                "pages": current_pages,
                "text": "\n\n".join(p["text"] for p in current_pages),
            }
        )

    return {
        "page_count": page_count,
        "ocr_pages_processed": len(ocr_candidates),
        "sections": sections,
    }
