import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and Path(WINDOWS_TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT_PATH

HEADING_RE = re.compile(
    r"^(unit\s+\d+|chapter\s+\d+|section\s+\d+|\d+(\.\d+){0,2}\s+[A-Z]).{2,90}$",
    re.I,
)


def _extract_page_text(page: fitz.Page) -> str:
    """Extracts clean digital text via PyMuPDF; safely falls back to English Tesseract OCR if installed."""
    raw = page.get_text("text").strip()
    if len(raw) >= 40:
        return unicodedata.normalize("NFKC", raw)

    try:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_txt = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
        return unicodedata.normalize("NFKC", ocr_txt).strip()
    except Exception:
        return unicodedata.normalize("NFKC", raw).strip()


def load_pdf_sections(pdf_path: Path) -> Dict[str, Any]:
    """
    Parses any English PDF into structured sections with exact page spans:
      1. Uses embedded PDF TOC bookmarks when present.
      2. Falls back to font-size + numbered heading detection across pages.
    """
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)

    # Extract all page texts first
    page_texts: List[Dict[str, Any]] = []
    for idx in range(page_count):
        txt = _extract_page_text(doc[idx])
        if txt:
            page_texts.append({"page_number": idx + 1, "text": txt})

    toc = doc.get_toc()  # [[level, title, page_num], ...]
    doc.close()

    sections: List[Dict[str, Any]] = []

    # Strategy A: Use embedded PDF Table of Contents (levels 1 & 2)
    filtered_toc = [
        (lvl, title.strip(), max(1, int(pg)))
        for lvl, title, pg in toc
        if lvl <= 2 and title.strip()
    ]
    if len(filtered_toc) >= 2:
        for i, (_, title, start_pg) in enumerate(filtered_toc):
            end_pg = (
                filtered_toc[i + 1][2] if (i + 1) < len(filtered_toc) else page_count
            )
            end_pg = max(start_pg, end_pg)
            sec_pages = [
                p for p in page_texts if start_pg <= p["page_number"] <= end_pg
            ]
            if sec_pages:
                combined = " ".join(p["text"] for p in sec_pages)
                sections.append(
                    {
                        "section_idx": len(sections) + 1,
                        "section_title": title[:140],
                        "start_page": start_pg,
                        "end_page": end_pg,
                        "pages": sec_pages,
                        "summary": combined[:900],
                    }
                )
        if sections:
            return {"page_count": page_count, "sections": sections}

    # Strategy B: Heading-line detection for textbooks without embedded bookmarks
    current_title = "Introduction & Overview"
    current_pages: List[Dict[str, Any]] = []
    start_pg = 1

    for p in page_texts:
        p_num = p["page_number"]
        lines = [ln.strip() for ln in p["text"].splitlines() if ln.strip()]
        detected_heading = None
        for ln in lines[:8]:
            if HEADING_RE.match(ln) and not ln.endswith("."):
                detected_heading = ln
                break

        if detected_heading and current_pages:
            combined = " ".join(x["text"] for x in current_pages)
            sections.append(
                {
                    "section_idx": len(sections) + 1,
                    "section_title": current_title[:140],
                    "start_page": start_pg,
                    "end_page": current_pages[-1]["page_number"],
                    "pages": current_pages,
                    "summary": combined[:900],
                }
            )
            current_title = detected_heading
            current_pages = [p]
            start_pg = p_num
        else:
            if not current_pages:
                start_pg = p_num
            current_pages.append(p)

    if current_pages:
        combined = " ".join(x["text"] for x in current_pages)
        sections.append(
            {
                "section_idx": len(sections) + 1,
                "section_title": current_title[:140],
                "start_page": start_pg,
                "end_page": current_pages[-1]["page_number"],
                "pages": current_pages,
                "summary": combined[:900],
            }
        )

    return {"page_count": page_count, "sections": sections}
