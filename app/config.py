from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Project Paths ---
    DATA_DIR: Path = Path("DATA/textbooks")
    PROCESSED_BOOKS_DIR: Path = Path("processed_data/books")
    PROCESSED_VIDEOS_DIR: Path = Path("processed_data/videos")
    CACHE_DB_PATH: Path = Path("cache/audit_cache.db")
    AGENTS_MD_PATH: Path = Path("skills/agents.md")
    SKILLS_MD_PATH: Path = Path("skills/skills.md")

    # --- Neon PostgreSQL ---
    DATABASE_URL: str
    DATABASE_URL_POOLED: Optional[str] = None

    # LLM Models (used by app/gateway/client.py)
    EXTRACTION_MODEL: str = "gemini-2.5-flash"
    AUDITOR_MODEL: str = "gemini-2.5-flash"

    # --- Pinecone Serverless (768d) ---
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "syllabus-gap-audit-vec"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # --- Embeddings & Cross-Encoder Reranker ---
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM: int = 768
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"

    # --- Cloudflare AI Gateway + Model Providers ---
    CF_ACCOUNT_ID: Optional[str] = None
    CF_GATEWAY_ID: Optional[str] = None
    CF_AIG_TOKEN: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # Model Routing
    FAST_MODEL: str = "llama-3.1-8b-instant"  # Groq: scope routing & fast checks
    AUDITOR_MODEL: str = "gemini-2.5-flash"  # Gemini: Worker, Evaluator & Repair
    GROQ_WHISPER_MODEL: str = (
        "whisper-large-v3-turbo"  # Groq: 216x real-time English ASR
    )

    # --- Quality, Similarity & Loop Bounds ---
    LANGUAGE: str = "en"
    WINDOW_SECONDS: float = 25.0
    MAX_VIDEO_DURATION_SEC: int = 5400
    MIN_MEAN_LOGPROB: float = -0.80
    MIN_OCR_CONFIDENCE: float = 0.65
    AMBIGUITY_LOW: float = 0.42
    AMBIGUITY_HIGH: float = 0.55
    MAX_CRITIC_RETRIES: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    for p in (
        s.DATA_DIR,
        s.PROCESSED_BOOKS_DIR,
        s.PROCESSED_VIDEOS_DIR,
        s.CACHE_DB_PATH.parent,
    ):
        p.mkdir(parents=True, exist_ok=True)
    return s
