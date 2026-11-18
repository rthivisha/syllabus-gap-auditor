import streamlit as st
from typing import Any, Optional


def render_live_ingestion_board(
    active_step: int = 0,
    doc_name: Optional[str] = None,
    page_count: Any = None,
    vector_count: Any = None,
    elapsed_sec: Any = None,
    placeholder: Any = None,
) -> None:
    """
    Renders a live, step-by-step Part A Ingestion Pipeline (Stages 01 -> 05):
      - active_step = 0: Standby (all grey/dim)
      - active_step = 1..5: Stage `active_step` glows cyan (RUNNING), stages < `active_step` light up bright emerald green (PASSED)
      - active_step = 6: All 5 stages lit emerald green (INGESTED IN PINECONE)
    """
    doc_label = doc_name or "Awaiting PDF"
    pages_txt = f"{page_count} Pages" if page_count else "PyMuPDF (fitz)"
    vecs_txt = f"{vector_count} Vectors" if vector_count else "768-d Pinecone"
    time_txt = f" · {elapsed_sec}s" if elapsed_sec else ""

    stages = [
        ("STAGE 01", "📥", "PDF Extraction", pages_txt, "PARSED"),
        ("STAGE 02", "🧹", "Noise Filtering", "TOC & Boilerplate", "CLEANED"),
        ("STAGE 03", "✂️", "Adaptive Chunking", "1.5k–2.8k Chars", "CHUNKED"),
        ("STAGE 04", "🧬", "Dense Embeddings", "BGE-base · 768-d", "EMBEDDED"),
        ("STAGE 05", "🗄️", "Pinecone Index", vecs_txt, "INDEXED ✓"),
    ]

    cards_html = []
    for idx, (st_code, icon, title, tech, done_badge) in enumerate(stages, start=1):
        if active_step > idx:
            card_cls = "ing-card done"
            pill_cls = "ing-pill done"
            pill_txt = done_badge
        elif active_step == idx:
            card_cls = "ing-card running"
            pill_cls = "ing-pill running"
            pill_txt = "RUNNING..."
        else:
            card_cls = "ing-card pending"
            pill_cls = "ing-pill pending"
            pill_txt = "PENDING"

        cards_html.append(
            f'<div class="{card_cls}">'
            f'<div class="ing-stage">{st_code}</div>'
            f'<div class="ing-icon">{icon}</div>'
            f'<div class="ing-title">{title}</div>'
            f'<div class="ing-tech">{tech}</div>'
            f'<span class="{pill_cls}">{pill_txt}</span>'
            f"</div>"
        )
        if idx < len(stages):
            arrow_cls = "ing-arrow lit" if active_step > idx else "ing-arrow"
            cards_html.append(f'<div class="{arrow_cls}">→</div>')

    status_header = (
        f"INGESTED & SYNCED IN PINECONE ({doc_label}{time_txt})"
        if active_step >= 6
        else (f"LIVE INGESTION PIPELINE — {doc_label}" if active_step > 0 else "RAG INGESTION PIPELINE (READY)")
    )

    html = f"""
    <style>
    .ing-wrap {{
        background: linear-gradient(180deg, #0b1220 0%, #090e1a 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 16px 18px;
        margin: 8px 0 14px 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .ing-header {{
        color: #94a3b8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}
    .ing-row {{
        display: flex;
        align-items: stretch;
        gap: 8px;
        overflow-x: auto;
    }}
    .ing-card {{
        flex: 1;
        min-width: 145px;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        transition: all 0.25s ease;
    }}
    .ing-card.pending {{
        background: #0b111e;
        border: 1px solid #1e293b;
        opacity: 0.55;
    }}
    .ing-card.running {{
        background: #0d1f33;
        border: 1.5px solid #38bdf8;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
        opacity: 1;
    }}
    .ing-card.done {{
        background: #0a221d;
        border: 1.5px solid #10b981;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.25);
        opacity: 1;
    }}
    .ing-stage {{ color: #64748b; font-size: 10px; font-weight: 700; letter-spacing: 1px; margin-bottom: 6px; }}
    .ing-icon {{ font-size: 22px; margin-bottom: 6px; }}
    .ing-title {{ color: #f8fafc; font-size: 13px; font-weight: 700; margin-bottom: 4px; }}
    .ing-tech {{ color: #7dd3fc; font-size: 11px; margin-bottom: 8px; }}
    .ing-pill {{
        display: inline-block;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.6px;
    }}
    .ing-pill.pending {{ background: rgba(100,116,139,0.15); color: #64748b; border: 1px solid #334155; }}
    .ing-pill.running {{ background: rgba(56,189,248,0.18); color: #38bdf8; border: 1px solid #38bdf8; }}
    .ing-pill.done {{ background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid #10b981; }}
    .ing-arrow {{ display: flex; align-items: center; color: #334155; font-size: 16px; font-weight: 700; }}
    .ing-arrow.lit {{ color: #10b981; }}
    </style>
    <div class="ing-wrap">
        <div class="ing-header">{status_header}</div>
        <div class="ing-row">{''.join(cards_html)}</div>
    </div>
    """

    target = placeholder if placeholder is not None else st
    target.html(html)


def render_system_architecture_board(
    doc_name: Optional[str] = None,
    page_count: Any = None,
    vector_count: Any = None,
    last_trace: Optional[Any] = None,
    placeholder: Any = None,
    **kwargs: Any,
) -> None:
    """Backward compatibility alias delegating to render_live_ingestion_board."""
    render_live_ingestion_board(
        active_step=6 if doc_name else 0,
        doc_name=doc_name,
        page_count=page_count,
        vector_count=vector_count,
        elapsed_sec=(last_trace or {}).get("elapsed_sec") if isinstance(last_trace, dict) else None,
        placeholder=placeholder,
    )
