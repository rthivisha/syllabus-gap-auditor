import json
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import yt_dlp
from app.config import get_settings
from app.guardrails.rails import check_media_guard, sanitize_transcript_text
from app.ingestion.chunking.splitter import format_mmss, merge_multimodal_windows


def extract_video_id(source: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([A-Za-z0-9_-]{11})", source)
    return match.group(1) if match else Path(source).stem


def _parse_timestamp_to_sec(ts: str) -> float:
    """Converts MM:SS or HH:MM:SS into float seconds."""
    parts = [int(p) for p in ts.strip().split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    return 0.0


def _fetch_youtube_captions_fast(
    video_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Tier 1 (~1.5s, free): Fetch English captions via youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        proxy_user = os.getenv("WEBSHARE_PROXY_USER")
        proxy_pass = os.getenv("WEBSHARE_PROXY_PASS")
        if proxy_user and proxy_pass:
            from youtube_transcript_api.proxies import WebshareProxyConfig
            ytt_api = YouTubeTranscriptApi(
                proxy_config=WebshareProxyConfig(
                    proxy_username=proxy_user,
                    proxy_password=proxy_pass,
                )
            )
        else:
            ytt_api = YouTubeTranscriptApi()

        try:
            transcript = ytt_api.fetch(
                video_id, languages=["en", "en-US", "en-GB", "en-IN"]
            )
        except Exception:
            t_list = ytt_api.list(video_id)
            chosen = next(iter(t_list), None)
            if chosen is None:
                return None
            transcript = (
                chosen.fetch()
                if chosen.language_code.startswith("en")
                else chosen.translate("en").fetch()
            )

        segs: List[Dict[str, Any]] = []
        for entry in transcript:
            raw_txt = (
                entry.get("text", "")
                if isinstance(entry, dict)
                else getattr(entry, "text", "")
            )
            txt = sanitize_transcript_text(str(raw_txt or "")).strip()
            start = float(
                entry.get("start", 0.0)
                if isinstance(entry, dict)
                else getattr(entry, "start", 0.0)
            )
            dur = float(
                entry.get("duration", 2.0)
                if isinstance(entry, dict)
                else getattr(entry, "duration", 2.0)
            )
            if txt:
                segs.append(
                    {
                        "start_sec": start,
                        "end_sec": start + dur,
                        "text": txt,
                        "avg_logprob": -0.10,
                    }
                )
        if segs:
            print(f"[CAPTIONS FAST] Fetched {len(segs)} segments via youtube-transcript-api")
            return segs
    except Exception as exc:
        print(f"[CAPTIONS WARN] youtube-transcript-api skipped ({exc}); using Gemini video fallback.")
    return None


def _extract_windows_via_gemini_youtube(
    source_url: str, window_sec: int = 25
) -> Optional[List[Dict[str, Any]]]:
    """
    Tier 2 Fallback (Cloud-Safe, Zero Caption Requirement):
    Passes the public YouTube URL directly to Gemini 2.5 Flash via file_uri,
    extracts chronological [MM:SS] topic/explanation intervals, and maps them
    directly into the 25-second windows schema used by BGE-base + Pinecone.
    """
    settings = get_settings()
    api_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
    if not api_key:
        print("[GEMINI VIDEO WARN] GEMINI_API_KEY not configured for YouTube fallback.")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        canonical_url = f"https://www.youtube.com/watch?v={extract_video_id(source_url)}"

        prompt = (
            "Extract a timestamped chronological outline of every topic, definition, "
            "formula, example, and concept taught in this lecture in 25-second intervals "
            "([MM:SS] topic and explanation).\n\n"
            "Formatting rules:\n"
            "- Output one line per interval starting strictly with [MM:SS] (or [HH:MM:SS] if over 1 hour).\n"
            "- Include specific technical terms, theorem names, equations, and explanations spoken or shown on slides.\n"
            "- Do not include markdown headers, bullet points, or intro/outro commentary."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=canonical_url)
                    ),
                    types.Part(text=prompt),
                ]
            ),
        )
        raw_text = (response.text or "").strip()
        if not raw_text:
            return None

        line_pattern = re.compile(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–:]?\s*(.+)$")
        parsed_entries: List[Dict[str, Any]] = []

        for line in raw_text.splitlines():
            clean_line = unicodedata.normalize("NFKC", line).strip()
            if not clean_line:
                continue
            m = line_pattern.match(clean_line)
            if m:
                ts_raw, body = m.group(1), sanitize_transcript_text(m.group(2)).strip()
                if not body:
                    continue
                start_sec = _parse_timestamp_to_sec(ts_raw)
                parsed_entries.append(
                    {
                        "start_sec": start_sec,
                        "end_sec": start_sec + float(window_sec),
                        "timestamp": format_mmss(start_sec),
                        "combined_text": body,
                        "transcript_text": body,
                        "ocr_text": "",
                    }
                )
            elif parsed_entries:
                # Append continuation text to previous timestamp window
                extra = sanitize_transcript_text(clean_line).strip()
                if extra:
                    parsed_entries[-1]["combined_text"] += f" {extra}"
                    parsed_entries[-1]["transcript_text"] += f" {extra}"

        if parsed_entries:
            print(f"[GEMINI VIDEO FALLBACK] Extracted {len(parsed_entries)} timestamped windows from YouTube URL.")
            return parsed_entries
    except Exception as exc:
        print(f"[GEMINI VIDEO ERROR] Gemini YouTube URL extraction failed: {exc}")

    return None


def parse_video_source(
    source: str, browser_for_cookies: Optional[str] = None
) -> Dict[str, Any]:
    """
    2-Tier Cloud-Safe Video Ingestion:
    1. Fast `youtube-transcript-api` (~1.5s, 0 LLM tokens) -> 25s merged windows.
    2. Automatic Fallback to Gemini 2.5 Flash native YouTube `file_uri` -> [MM:SS] 25s windows.
    """
    t_total = time.perf_counter()
    timings: Dict[str, float] = {}

    settings = get_settings()
    video_id = extract_video_id(source)
    is_url = source.startswith(("http://", "https://"))
    title = "YouTube Lecture"
    duration = 0.0

    # Step 1: Lightweight Metadata Check (8s timeout, skip_download)
    t0 = time.perf_counter()
    if is_url:
        try:
            base_opts: Dict[str, Any] = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "socket_timeout": 8,
                "retries": 1,
            }
            if browser_for_cookies:
                base_opts["cookiesfrombrowser"] = (browser_for_cookies,)
            with yt_dlp.YoutubeDL(base_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                duration = float(info.get("duration") or 0)
                title = info.get("title", "YouTube Lecture")
        except Exception as exc:
            print(f"[METADATA WARN] Metadata lookup skipped ({exc}); proceeding to captions/Gemini.")
    timings["metadata_fetch_sec"] = round(time.perf_counter() - t0, 2)

    if is_url and duration > settings.MAX_VIDEO_DURATION_SEC:
        return {
            "video_id": video_id,
            "title": title,
            "duration_sec": duration,
            "passed_guard": False,
            "guard_reason": f"Duration {duration}s exceeds allowed limit ({settings.MAX_VIDEO_DURATION_SEC}s).",
            "windows": [],
            "ocr_slides": [],
            "step_timings": timings,
        }

    # Step 2: Tier 1 Fast Captions -> Tier 2 Gemini Native YouTube Fallback
    t0 = time.perf_counter()
    raw_segs = _fetch_youtube_captions_fast(video_id) if is_url else None
    ocr_slides: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    mean_lp = -0.10

    if raw_segs:
        if duration <= 0:
            duration = max((s["end_sec"] for s in raw_segs), default=600.0)
        guard = check_media_guard(duration, len(raw_segs), mean_lp)
        if not guard.passed:
            return {
                "video_id": video_id,
                "title": title,
                "duration_sec": duration,
                "mean_logprob": mean_lp,
                "passed_guard": False,
                "guard_reason": guard.reason,
                "windows": [],
                "ocr_slides": [],
                "step_timings": timings,
            }
        timings["transcription_sec"] = round(time.perf_counter() - t0, 2)
        timings["transcription_mode"] = "youtube_captions_fast"

        t_merge = time.perf_counter()
        windows = merge_multimodal_windows(
            raw_segs, ocr_slides, window_sec=settings.WINDOW_SECONDS
        )
        timings["window_merge_sec"] = round(time.perf_counter() - t_merge, 2)
    else:
        windows = _extract_windows_via_gemini_youtube(
            source, window_sec=settings.WINDOW_SECONDS
        ) or []
        timings["transcription_sec"] = round(time.perf_counter() - t0, 2)
        timings["transcription_mode"] = "gemini_2.5_flash_youtube_uri"

        if not windows:
            return {
                "video_id": video_id,
                "title": title,
                "duration_sec": duration,
                "passed_guard": False,
                "guard_reason": (
                    "Could not extract lecture content from this YouTube URL. "
                    "Ensure the video is public (not private or unlisted)."
                ),
                "windows": [],
                "ocr_slides": [],
                "step_timings": timings,
            }
        if duration <= 0:
            duration = max((w["end_sec"] for w in windows), default=600.0)

    timings["total_ingestion_sec"] = round(time.perf_counter() - t_total, 2)

    return {
        "video_id": video_id,
        "title": title,
        "duration_sec": duration,
        "mean_logprob": mean_lp,
        "mean_ocr_conf": 0.0,
        "passed_guard": True,
        "guard_reason": "Passed lecture ingestion quality gates.",
        "windows": windows,
        "ocr_slides": ocr_slides,
        "step_timings": timings,
    }


def parse_raw_transcript_text(
    transcript_text: str, title: str = "Pasted Lecture Transcript"
) -> Dict[str, Any]:
    cleaned = sanitize_transcript_text(transcript_text).strip()
    words = cleaned.split()
    # Chunk every ~60 words as a synthetic 25-second window
    segs = []
    for idx in range(0, len(words), 60):
        start_sec = (idx // 60) * 25.0
        segs.append({
            "start_sec": start_sec,
            "end_sec": start_sec + 25.0,
            "text": " ".join(words[idx : idx + 60]),
            "avg_logprob": -0.10,
        })
    windows = merge_multimodal_windows(segs, [], window_sec=25)
    return {
        "video_id": "pasted-transcript",
        "title": title,
        "duration_sec": len(segs) * 25.0,
        "passed_guard": len(cleaned) >= 120,
        "guard_reason": "Passed pasted transcript guard." if len(cleaned) >= 120 else "Transcript too short.",
        "windows": windows,
        "ocr_slides": [],
        "step_timings": {"transcription_mode": "manual_paste"},
    }
