import re
from dataclasses import dataclass
from pathlib import Path
from app.config import get_settings

YOUTUBE_URL_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/(watch\?v=|shorts/|live/)|youtu\.be/)[A-Za-z0-9_-]{11}"
)

INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions|system\s*:\s*you\s+are|mark\s+all\s+topics\s+as\s+covered|override\s+syllabus\s+audit)",
    re.I,
)


@dataclass
class GuardCheck:
    passed: bool
    reason: str


def check_input_guard(source: str, grade: int = 1, chapter: int = 1) -> GuardCheck:
    """Validates YouTube URL or local video file format."""
    src = source.strip()
    if src.startswith(("http://", "https://")):
        if not YOUTUBE_URL_RE.match(src):
            return GuardCheck(False, "Invalid YouTube URL format.")
        return GuardCheck(True, "Valid YouTube URL.")

    p = Path(src)
    if not p.exists() or p.suffix.lower() not in {".mp4", ".mkv", ".webm", ".mov"}:
        return GuardCheck(False, f"Video file not found or unsupported format: {src}")
    return GuardCheck(True, "Valid local video file.")


def check_media_guard(
    duration_sec: float, segment_count: int, mean_logprob: float
) -> GuardCheck:
    """Stage A Media Guard: verifies duration cap and English ASR confidence."""
    s = get_settings()
    if duration_sec <= 0 or duration_sec > s.MAX_VIDEO_DURATION_SEC:
        return GuardCheck(
            False,
            f"Duration {duration_sec}s out of bounds (1s-{s.MAX_VIDEO_DURATION_SEC}s).",
        )
    if segment_count == 0 or mean_logprob < s.MIN_MEAN_LOGPROB:
        return GuardCheck(
            False,
            f"Low English ASR quality (mean_logprob={mean_logprob} < {s.MIN_MEAN_LOGPROB}).",
        )
    return GuardCheck(True, "Passed media quality checks.")


def sanitize_transcript_text(text: str) -> str:
    """Strips embedded prompt-injection strings from untrusted video audio/slide OCR."""
    return INJECTION_RE.sub("[REDACTED]", text)
