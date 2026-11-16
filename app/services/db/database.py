from contextlib import contextmanager
from sqlalchemy import create_engine
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
_direct_url = _normalize_pg_url(_settings.DATABASE_URL)
_runtime_url = _normalize_pg_url(
    _settings.DATABASE_URL_POOLED or _settings.DATABASE_URL
)

# Direct engine for DDL / table creation; NullPool runtime engine for Neon PgBouncer
direct_engine = create_engine(_direct_url, pool_pre_ping=True)
runtime_engine = (
    create_engine(_runtime_url, poolclass=NullPool, pool_pre_ping=True)
    if _settings.DATABASE_URL_POOLED
    else direct_engine
)

SessionLocal = sessionmaker(bind=runtime_engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Creates all PostgreSQL tables idempotently on startup."""
    from app.services.db import models  # noqa: F401

    Base.metadata.create_all(bind=direct_engine)


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
