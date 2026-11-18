_PROJECT_PHRASES = (
    "this project",
    "about the project",
    "the project",
    "about this app",
    "this app",
    "the app",
    "this application",
    "the application",
    "how does this work",
    "how it works",
    "this system",
    "the system",
    "what do you do",
    "tech stack",
    "pipeline",
    "syllabus gap audit",
)


def is_explicit_project_query(message: str) -> bool:
    """Recognize unambiguous project questions before semantic PDF matching."""
    msg = (message or "").lower().strip()
    return any(phrase in msg for phrase in _PROJECT_PHRASES) or msg.rstrip(
        "!?. "
    ) in {
        "hi",
        "hello",
        "hey",
        "help",
        "good morning",
        "good afternoon",
        "good evening",
        "what is this",
    }


def classify_intent(
    message: str,
    has_active_doc: bool,
    prior_clarification: bool = False,
) -> str:
    """Route in priority order: audit, project guidance, then PDF content Q&A."""
    msg = (message or "").lower().strip()

    if prior_clarification or "youtube.com/" in msg or "youtu.be/" in msg:
        return "audit"

    if is_explicit_project_query(msg):
        return "project_guide"

    pdf_phrases = (
        "in the pdf",
        "in pdf",
        "from the pdf",
        "document",
        "textbook",
        "book",
        "chapter",
        "contents",
        "topics",
        "syllabus",
        "section",
        "summarize",
        "explain",
    )
    if has_active_doc and any(phrase in msg for phrase in pdf_phrases):
        return "pdf_qa"

    return "project_guide"
