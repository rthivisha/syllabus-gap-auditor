from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import get_settings


def build_gemini_chat(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    thinking_budget: int = 0,
) -> ChatGoogleGenerativeAI:
    """
    Creates a LangChain ChatGoogleGenerativeAI client routed through
    Cloudflare AI Gateway (with caching, rate-limiting & telemetry).
    """
    settings = get_settings()
    target_model = model_name or settings.AUDITOR_MODEL

    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing in .env")

    kwargs: dict = {
        "model": target_model,
        "google_api_key": settings.GEMINI_API_KEY,
        "temperature": temperature,
        "thinking_budget": thinking_budget,
        "timeout": 180.0,
        "max_retries": 3,
    }

    if settings.CF_ACCOUNT_ID and settings.CF_GATEWAY_ID and settings.CF_AIG_TOKEN:
        cf_endpoint = (
            f"https://gateway.ai.cloudflare.com/v1/"
            f"{settings.CF_ACCOUNT_ID}/{settings.CF_GATEWAY_ID}/google-ai-studio"
        )
        kwargs["client_options"] = {"api_endpoint": cf_endpoint}
        kwargs["additional_headers"] = {
            "cf-aig-authorization": f"Bearer {settings.CF_AIG_TOKEN}"
        }

    return ChatGoogleGenerativeAI(**kwargs)


def get_extraction_llm() -> ChatGoogleGenerativeAI:
    """Fast, low-cost model for batched proposition & video topic extraction."""
    settings = get_settings()
    return build_gemini_chat(
        model_name=settings.EXTRACTION_MODEL,
        temperature=0.0,
        thinking_budget=0,
    )


def get_auditor_llm() -> ChatGoogleGenerativeAI:
    """High-accuracy reasoning model for syllabus coverage & gap verdicts."""
    settings = get_settings()
    return build_gemini_chat(
        model_name=settings.AUDITOR_MODEL,
        temperature=0.0,
        thinking_budget=0,
    )
