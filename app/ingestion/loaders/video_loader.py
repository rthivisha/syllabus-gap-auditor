import json
import os
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None
import imagehash
import pytesseract
import yt_dlp
from groq import Groq
from PIL import Image
from app.config import get_settings
from app.guardrails.rails import check_media_guard, sanitize_transcript_text
from app.ingestion.chunking.splitter import format_mmss, merge_multimodal_windows

WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and Path(WINDOWS_TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT_PATH


def extract_video_id(source: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([A-Za-z0-9_-]{11})", source)
    return match.group(1) if match else Path(source).stem


def _extract_compressed_audio(video_file: Path, audio_out: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "48k",
        str(audio_out),
    ]
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )


def _extract_unique_slide_ocr(
    video_file: Path, min_conf: float = 0.65, sample_every_sec: int = 8
) -> tuple[List[Dict[str, Any]], float]:
    if cv2 is None:
        return [], 0.0
    cap = cv2.VideoCapture(str(video_file))
    slides: List[Dict[str, Any]] = []
    confs: List[float] = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(fps * sample_every_sec))
        last_hash = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                curr_hash = imagehash.phash(pil_img)

                if last_hash is not None and (curr_hash - last_hash) <= 10:
                    frame_idx += 1
                    continue
                last_hash = curr_hash

                data = pytesseract.image_to_data(
                    pil_img,
                    lang="eng",
                    config="--psm 6",
                    output_type=pytesseract.Output.DICT,
                )
                words: List[str] = []
                frame_confs: List[float] = []
                for txt, c in zip(data["text"], data["conf"]):
                    score = float(c) / 100.0 if float(c) >= 0 else 0.0
                    cleaned = unicodedata.normalize("NFKC", str(txt)).strip()
                    if score >= min_conf and len(cleaned) >= 2:
                        words.append(cleaned)
                        frame_confs.append(score)

                if words:
                    sec = frame_idx / fps
                    avg_c = round(sum(frame_confs) / len(frame_confs), 4)
                    confs.append(avg_c)
                    slides.append(
                        {
                            "sec": round(sec, 2),
                            "timestamp": format_mmss(sec),
                            "text": sanitize_transcript_text(" ".join(words)),
                            "confidence": avg_c,
                        }
                    )
            frame_idx += 1
    finally:
        cap.release()

    mean_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
    return slides, mean_conf


def _fetch_youtube_captions_fast(
    video_id: str, source_url: str
) -> Optional[List[Dict[str, Any]]]:
    """Fetch English captions via youtube-transcript-api (with optional proxy), then fast yt-dlp JSON3 subtitles."""
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
            # Fallback: grab any available transcript and translate to English if needed
            t_list = ytt_api.list(video_id)
            chosen = next(iter(t_list), None)
            if chosen is None:
                raise RuntimeError("No caption tracks found.")
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
        print(f"[CAPTIONS WARN] youtube-transcript-api failed ({exc}), trying yt-dlp subtitles...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-US", "en-GB", "en-IN", "en-orig"],
                "subtitlesformat": "json3",
                "socket_timeout": 8,
                "retries": 1,
                "outtmpl": str(Path(tmpdir) / "%(id)s.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(sub_opts) as ydl:
                ydl.download([source_url])
            json3_files = sorted(Path(tmpdir).glob("*.json3"))
            if json3_files:
                payload = json.loads(json3_files[0].read_text(encoding="utf-8"))
                segs = []
                for event in payload.get("events", []):
                    start_ms = event.get("tStartMs")
                    if start_ms is None or "segs" not in event:
                        continue
                    text = "".join(
                        segment.get("utf8", "") for segment in event["segs"]
                    ).replace("\n", " ").strip()
                    text = sanitize_transcript_text(text)
                    if text:
                        start_sec = start_ms / 1000.0
                        duration_sec = (event.get("dDurationMs") or 2000) / 1000.0
                        segs.append(
                            {
                                "start_sec": start_sec,
                                "end_sec": start_sec + duration_sec,
                                "text": text,
                                "avg_logprob": -0.12,
                            }
                        )
                if segs:
                    print(f"[CAPTIONS FAST] Fetched {len(segs)} segments via yt-dlp json3 subtitles")
                    return segs
    except Exception as exc:
        print(f"[CAPTIONS WARN] yt-dlp subtitles failed ({exc})")

    return None


def parse_video_source(
    source: str, browser_for_cookies: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fast Caption-Only Ingestion (~1.5s) with strict empty-transcript guardrails.
    Never downloads heavy video/audio streams on cloud servers.
    """
    import time

    t_total = time.perf_counter()
    timings: Dict[str, float] = {}

    settings = get_settings()
    video_id = extract_video_id(source)
    is_url = source.startswith(("http://", "https://"))
    title = "YouTube Lecture"
    duration = 0.0

    # Step 1: Fast Metadata Check (8s timeout)
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
            print(f"[METADATA WARN] yt-dlp metadata lookup skipped ({exc}); proceeding directly to captions.")
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

    # Step 2: Fast Captions Extraction (Never falls back to slow audio download on YouTube URLs)
    t0 = time.perf_counter()
    raw_segs = _fetch_youtube_captions_fast(video_id, source) if is_url else None
    ocr_slides: List[Dict[str, Any]] = []
    mean_ocr_conf = 0.0

    if not raw_segs:
        return {
            "video_id": video_id,
            "title": title,
            "duration_sec": duration,
            "passed_guard": False,
            "guard_reason": (
                "Could not fetch English captions for this YouTube video (captions may be disabled "
                "or YouTube blocked this cloud server IP). Please use a video with English captions enabled."
            ),
            "windows": [],
            "ocr_slides": [],
            "step_timings": timings,
        }

    if duration <= 0:
        duration = max((s["end_sec"] for s in raw_segs), default=600.0)

    timings["transcription_sec"] = round(time.perf_counter() - t0, 2)
    timings["transcription_mode"] = "youtube_captions_fast"
    mean_lp = -0.10

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

    # Step 3: Merge into 25-second windows
    t0 = time.perf_counter()
    windows = merge_multimodal_windows(
        raw_segs, ocr_slides, window_sec=settings.WINDOW_SECONDS
    )
    timings["window_merge_sec"] = round(time.perf_counter() - t0, 2)
    timings["total_ingestion_sec"] = round(time.perf_counter() - t_total, 2)

    return {
        "video_id": video_id,
        "title": title,
        "duration_sec": duration,
        "mean_logprob": mean_lp,
        "mean_ocr_conf": mean_ocr_conf,
        "passed_guard": True,
        "guard_reason": "Passed English caption quality gates.",
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



