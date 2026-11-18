import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Generator, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from app.config import get_settings

_USER_GEMINI_KEY: ContextVar[Optional[str]] = ContextVar(
    "user_gemini_key", default=None
)
_USER_GROQ_KEY: ContextVar[Optional[str]] = ContextVar(
    "user_groq_key", default=None
)


@contextmanager
def use_user_api_keys(
    gemini_key: Optional[str] = None, groq_key: Optional[str] = None
) -> Generator[None, None, None]:
    """Make optional BYOK credentials available only for the current request."""
    gemini_token = _USER_GEMINI_KEY.set((gemini_key or "").strip() or None)
    groq_token = _USER_GROQ_KEY.set((groq_key or "").strip() or None)
    try:
        yield
    finally:
        _USER_GEMINI_KEY.reset(gemini_token)
        _USER_GROQ_KEY.reset(groq_token)


def _cf_default_headers() -> Optional[Dict[str, str]]:
    settings = get_settings()
    if settings.CF_AIG_TOKEN:
        return {"cf-aig-authorization": f"Bearer {settings.CF_AIG_TOKEN}"}
    return None


def _cf_gemini_client_options() -> Optional[Dict[str, Any]]:
    settings = get_settings()
    if settings.CF_ACCOUNT_ID and settings.CF_GATEWAY_ID:
        return {
            "api_endpoint": (
                f"https://gateway.ai.cloudflare.com/v1/"
                f"{settings.CF_ACCOUNT_ID}/{settings.CF_GATEWAY_ID}/google-ai-studio"
            )
        }
    return None


def _cf_groq_base_url() -> Optional[str]:
    settings = get_settings()
    if settings.CF_ACCOUNT_ID and settings.CF_GATEWAY_ID:
        # ChatGroq automatically appends /openai/v1/chat/completions — do NOT add /openai/v1 here!
        return (
            f"https://gateway.ai.cloudflare.com/v1/"
            f"{settings.CF_ACCOUNT_ID}/{settings.CF_GATEWAY_ID}/groq"
        )
    return None


def _build_groq_fallback(
    temperature: float = 0.1,
    model: Optional[str] = None,
    override_key: Optional[str] = None,
) -> Optional[ChatGroq]:
    settings = get_settings()
    user_key = (override_key or "").strip() or _USER_GROQ_KEY.get()
    api_key = user_key or settings.GROQ_API_KEY
    if not api_key:
        print("[WARN] GROQ_API_KEY is missing in .env — Groq fallback disabled!")
        return None
    kwargs: Dict[str, Any] = {
        "model": model or settings.FALLBACK_MODEL,
        "api_key": api_key,
        "temperature": temperature,
        "max_retries": 0 if model else 2,
        "timeout": 10.0 if model else 45.0,
    }
    if not user_key:
        cf_url = _cf_groq_base_url()
        if cf_url:
            kwargs["base_url"] = cf_url
        headers = _cf_default_headers()
        if headers:
            kwargs["default_headers"] = headers
    return ChatGroq(**kwargs)


def _build_clean_gemini(
    override_key: Optional[str] = None,
) -> Optional[ChatGoogleGenerativeAI]:
    settings = get_settings()
    user_key = (override_key or "").strip() or _USER_GEMINI_KEY.get()
    api_key = user_key or settings.GEMINI_API_KEY
    if not api_key:
        return None
    kwargs: Dict[str, Any] = {
        "model": settings.AUDITOR_MODEL,  # gemini-2.5-flash or gemini-3.5-flash-lite
        "google_api_key": api_key,
        "temperature": None,
        "thinking_budget": 0,
        "max_retries": 0,  # Fail fast on 429 quota exhaustion so Groq takes over immediately
        "timeout": 20.0,
    }
    if not user_key:
        kwargs["client_options"] = _cf_gemini_client_options()
        headers = _cf_default_headers()
        if headers:
            kwargs["additional_headers"] = headers
    return ChatGoogleGenerativeAI(**kwargs)


def get_fast_groq_llm(
    user_groq_key: Optional[str] = None, user_gemini_key: Optional[str] = None
) -> Any:
    """Sub-second Groq LLM primary for Project Q&A, PDF Q&A, and intent routing, with Gemini fallback."""
    groq = _build_groq_fallback(temperature=0.1, override_key=user_groq_key)
    gemini = _build_clean_gemini(override_key=user_gemini_key)
    if groq is None:
        return gemini
    if gemini is None:
        return groq

    def _invoke(messages: Any) -> Any:
        try:
            return groq.invoke(messages)
        except Exception:
            return gemini.invoke(messages)

    return RunnableLambda(_invoke)


def _ensure_json_instruction(messages: Any, schema: Any) -> List[BaseMessage]:
    """Guarantees Groq's json_mode requirement ('JSON' keyword + schema fields in prompt)."""
    msg_list = list(messages) if isinstance(messages, (list, tuple)) else [messages]
    schema_json = (
        json.dumps(schema.model_json_schema(), indent=2)
        if hasattr(schema, "model_json_schema")
        else ""
    )
    json_guard = SystemMessage(
        content=(
            "You MUST respond ONLY with a valid JSON object matching this exact schema:\n"
            f"{schema_json}"
        )
    )
    return [json_guard, *msg_list]


def get_extraction_llm(model: Optional[str] = None) -> BaseChatModel:
    """Return the default extractor or a direct Groq model when one is requested."""
    settings = get_settings()
    if model and (_USER_GROQ_KEY.get() or settings.GROQ_API_KEY):
        fast_groq = _build_groq_fallback(temperature=0.0, model=model)
        if fast_groq:
            return fast_groq

    fallback = _build_groq_fallback(temperature=0.0)
    primary = _build_clean_gemini()
    if primary is None and fallback:
        return fallback
    if primary is None:
        raise RuntimeError("No Gemini or Groq API key is configured for this request.")
    if fallback:
        return primary.with_fallbacks([fallback], exceptions_to_handle=(Exception,))
    return primary


def get_auditor_llm() -> BaseChatModel:
    """Primary auditor model with immediate Gemini -> Groq failover."""
    fallback = _build_groq_fallback(temperature=0.1)
    primary = _build_clean_gemini()
    if primary is None and fallback:
        return fallback
    if primary is None:
        raise RuntimeError("No Gemini or Groq API key is configured for this request.")
    if fallback:
        return primary.with_fallbacks([fallback], exceptions_to_handle=(Exception,))
    return primary


def get_structured_extraction_llm(schema: Any) -> Any:
    """Structured extraction runnable with Gemini -> Groq failover."""
    settings = get_settings()
    fallback = _build_groq_fallback(temperature=0.0)
    groq_structured = (
        fallback.with_structured_output(schema, method="json_mode")
        if fallback
        else None
    )

    primary = _build_clean_gemini()
    primary_structured = (
        primary.with_structured_output(schema) if primary else None
    )
    if primary_structured is None and groq_structured:
        return RunnableLambda(
            lambda messages: groq_structured.invoke(
                _ensure_json_instruction(messages, schema)
            )
        )
    if primary_structured is None:
        raise RuntimeError("No Gemini or Groq API key is configured for this request.")

    def _invoke_with_failover(messages: Any) -> Any:
        try:
            return primary_structured.invoke(messages)
        except Exception as exc:
            if groq_structured is None:
                raise
            print(
                f"[LLM Failover] Gemini extraction failed ({type(exc).__name__}), "
                f"switching to Groq {settings.FALLBACK_MODEL}..."
            )
            return groq_structured.invoke(_ensure_json_instruction(messages, schema))

    return RunnableLambda(_invoke_with_failover)


def get_structured_auditor_llm(schema: Any) -> Any:
    """
    Structured output runnable with verified Gemini -> Groq failover.
    Injects schema JSON guard on Groq failover so json_mode never throws a 400 error.
    """
    settings = get_settings()
    fallback = _build_groq_fallback(temperature=0.1)
    groq_structured = (
        fallback.with_structured_output(schema, method="json_mode")
        if fallback
        else None
    )

    primary = _build_clean_gemini()
    primary_structured = (
        primary.with_structured_output(schema) if primary else None
    )
    if primary_structured is None and groq_structured:
        return RunnableLambda(
            lambda msgs: groq_structured.invoke(_ensure_json_instruction(msgs, schema))
        )
    if primary_structured is None:
        raise RuntimeError("No Gemini or Groq API key is configured for this request.")

    def _invoke_with_failover(messages: Any) -> Any:
        try:
            return primary_structured.invoke(messages)
        except Exception as exc:
            if groq_structured is None:
                raise
            print(
                f"[LLM Failover] Gemini failed ({type(exc).__name__}), "
                f"switching to Groq {settings.FALLBACK_MODEL}..."
            )
            return groq_structured.invoke(_ensure_json_instruction(messages, schema))

    return RunnableLambda(_invoke_with_failover)
