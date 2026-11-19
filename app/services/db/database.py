from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import NullPool
from app.config import get_settings


def _normalize_pg_url(raw_url: str) -> str:
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+psycopg://", 1)
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw_url


class Base(DeclarativeBase):
    pass


_settings = get_settings()
_direct_url = _normalize_pg_url(_settings.DATABASE_URL or "sqlite:///cache/audit_cache.db")
_runtime_url = _normalize_pg_url(
    _settings.DATABASE_URL_POOLED or _settings.DATABASE_URL or "sqlite:///cache/audit_cache.db"
)

# Direct engine for DDL / table creation; NullPool runtime engine for Neon PgBouncer
_connect_args = {"check_same_thread": False} if "sqlite" in _direct_url else {}
direct_engine = create_engine(_direct_url, pool_pre_ping=True, connect_args=_connect_args)
runtime_engine = (
    create_engine(_runtime_url, poolclass=NullPool, pool_pre_ping=True, connect_args=_connect_args)
    if _settings.DATABASE_URL_POOLED
    else direct_engine
)

SessionLocal = sessionmaker(bind=runtime_engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Creates tables and applies additive document-metadata upgrades."""
    from app.services.db import models  # noqa: F401

    Base.metadata.create_all(bind=direct_engine)
    if "postgres" in _direct_url:
        try:
            with direct_engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE documents "
                        "ADD COLUMN IF NOT EXISTS suggested_questions_json "
                        "JSONB NOT NULL DEFAULT '[]'::jsonb"
                    )
                )
                connection.execute(
                    text(
                        "ALTER TABLE documents "
                        "ADD COLUMN IF NOT EXISTS index_version "
                        "INTEGER NOT NULL DEFAULT 1"
                    )
                )
        except Exception:
            pass


@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
