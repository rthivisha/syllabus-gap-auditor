import json
import os
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
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
    """Fetch English captions via youtube-transcript-api, then yt-dlp subtitles."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(
            video_id, languages=["en", "en-US", "en-GB", "en-IN"]
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
            print(
                f"[CAPTIONS FAST] Fetched {len(segs)} segments via youtube-transcript-api"
            )
            return segs
        print(
            "[CAPTIONS WARN] youtube-transcript-api returned no English segments, "
            "trying yt-dlp subtitles..."
        )
    except Exception as exc:
        print(
            f"[CAPTIONS WARN] youtube-transcript-api failed ({exc}), "
            "trying yt-dlp subtitles..."
        )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-US", "en-orig"],
                "subtitlesformat": "json3",
                "outtmpl": str(Path(tmpdir) / "%(id)s.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(sub_opts) as ydl:
                ydl.download([source_url])
            json3_files = sorted(Path(tmpdir).glob("*.json3"))
            if not json3_files:
                print("[CAPTIONS WARN] yt-dlp did not produce an English JSON3 subtitle file.")
            else:
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
                    print(
                        f"[CAPTIONS FAST] Fetched {len(segs)} segments via "
                        "yt-dlp json3 subtitles"
                    )
                    return segs
                print("[CAPTIONS WARN] yt-dlp JSON3 subtitles contained no usable segments.")
    except Exception as exc:
        print(
            f"[CAPTIONS WARN] yt-dlp subtitles failed ({exc}), "
            "falling back to Whisper audio..."
        )

    return None


def parse_video_source(
    source: str, browser_for_cookies: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fast Multimodal Ingestion with per-step timing telemetry:
      1. Metadata & Caption Fast-Path (~1.5s) -> falls back to Audio-Only Groq Whisper (~15s).
      2. Window Merging (25s chunks) & Guardrails.
    """
    import time

    t_total = time.perf_counter()
    timings: Dict[str, float] = {}

    settings = get_settings()
    video_id = extract_video_id(source)
    is_url = source.startswith(("http://", "https://"))
    title = "YouTube Lecture"
    duration = 0.0

    # Step 1: Metadata + Fast YouTube Captions Check
    t0 = time.perf_counter()
    if is_url:
        base_opts: Dict[str, Any] = {"quiet": True, "no_warnings": True}
        if browser_for_cookies:
            base_opts["cookiesfrombrowser"] = (browser_for_cookies,)
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(source, download=False)
            duration = float(info.get("duration") or 0)
            title = info.get("title", "YouTube Lecture")
    timings["metadata_fetch_sec"] = round(time.perf_counter() - t0, 2)

    if is_url and (duration <= 0 or duration > settings.MAX_VIDEO_DURATION_SEC):
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

    # Step 2: Fast Captions Path vs. Audio-Only Groq Whisper Fallback
    t0 = time.perf_counter()
    raw_segs = _fetch_youtube_captions_fast(video_id, source) if is_url else None
    ocr_slides: List[Dict[str, Any]] = []
    mean_ocr_conf = 0.0

    if raw_segs:
        timings["transcription_sec"] = round(time.perf_counter() - t0, 2)
        timings["transcription_mode"] = "youtube_captions_fast"
        mean_lp = -0.10
    else:
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in .env")
        groq_client = Groq(api_key=settings.GROQ_API_KEY)
        with tempfile.TemporaryDirectory() as tmpdir:
            if is_url:
                audio_dl = Path(tmpdir) / f"{video_id}.m4a"
                dl_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "format": "bestaudio[ext=m4a]/bestaudio/best",
                    "outtmpl": str(audio_dl),
                }
                with yt_dlp.YoutubeDL(dl_opts) as ydl:
                    ydl.download([source])
                timings["audio_download_sec"] = round(time.perf_counter() - t0, 2)
                source_media = audio_dl
            else:
                source_media = Path(source)
                title = source_media.stem

            t_asr = time.perf_counter()
            audio_file = Path(tmpdir) / f"{video_id}.mp3"
            _extract_compressed_audio(source_media, audio_file)
            with open(audio_file, "rb") as af:
                resp = groq_client.audio.transcriptions.create(
                    file=(audio_file.name, af.read()),
                    model=settings.GROQ_WHISPER_MODEL,
                    language="en",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    temperature=0.0,
                )

            raw_segs = []
            logprobs: List[float] = []
            for seg in getattr(resp, "segments", []) or []:
                txt = (
                    seg.get("text", "")
                    if isinstance(seg, dict)
                    else getattr(seg, "text", "")
                ).strip()
                if not txt:
                    continue
                s_sec = float(
                    seg.get("start", 0.0)
                    if isinstance(seg, dict)
                    else getattr(seg, "start", 0.0)
                )
                e_sec = float(
                    seg.get("end", s_sec)
                    if isinstance(seg, dict)
                    else getattr(seg, "end", s_sec)
                )
                lp = float(
                    seg.get("avg_logprob", -0.25)
                    if isinstance(seg, dict)
                    else getattr(seg, "avg_logprob", -0.25)
                )
                raw_segs.append(
                    {
                        "start_sec": s_sec,
                        "end_sec": e_sec,
                        "text": sanitize_transcript_text(txt),
                        "avg_logprob": lp,
                    }
                )
                logprobs.append(lp)
            mean_lp = round(sum(logprobs) / len(logprobs), 4) if logprobs else -9.9
            timings["whisper_asr_sec"] = round(time.perf_counter() - t_asr, 2)
            timings["transcription_mode"] = "groq_whisper_audio"

    guard = check_media_guard(duration or 600.0, len(raw_segs), mean_lp)
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
        "guard_reason": "Passed English ASR quality gates.",
        "windows": windows,
        "ocr_slides": ocr_slides,
        "step_timings": timings,
    }


