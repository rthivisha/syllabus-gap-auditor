import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.config import get_settings
from app.gateway.client import get_structured_extraction_llm


class ScopeDecision(BaseModel):
    needs_clarification: bool = Field(
        description="True ONLY if the user has not yet specified whether to check a specific chapter/section or the entire book."
    )
    is_full_book: bool = Field(
        description="True if the user explicitly requested scanning the entire book/document."
    )
    requested_keywords: List[str] = Field(
        default_factory=list,
        description="Chapter numbers, unit names, or section keywords mentioned by the user (e.g. ['Chapter 1', 'Measurements']).",
    )
    clarifying_question: str = Field(
        default="",
        description="A single polite clarifying question asking which chapter/section to check or whether to scan the whole book.",
    )


def load_agents_contract() -> str:
    path = get_settings().AGENTS_MD_PATH
    return path.read_text(encoding="utf-8") if path.exists() else ""


def resolve_audit_scope(
    user_message: str,
    doc_name: str,
    doc_sections: List[Dict[str, Any]],
    prior_clarification_asked: bool = False,
) -> Dict[str, Any]:
    """
    Conversational Scope Router:
      - If user asks a broad question on Turn 1 ("Help me find what I missed"), asks 1 clarifying question.
      - Once answered (or if scope is already explicit), maps request to exact section_title list.
    """
    available_titles = [s["section_title"] for s in doc_sections]
    toc_preview = "\n".join(
        f"- {s['section_title']} (Pages {s.get('start_page', 1)}-{s.get('end_page', 1)})"
        for s in doc_sections[:30]
    )

    llm = get_structured_extraction_llm(ScopeDecision)
    sys_prompt = (
        f"{load_agents_contract()}\n\n"
        "You are the Conversational Scope Router.\n"
        f"Active Document: '{doc_name}'\n"
        f"Document Sections:\n{toc_preview}\n\n"
        f"Clarification already asked in this session: {prior_clarification_asked}.\n"
        "Rules:\n"
        "- If clarification was NOT yet asked AND the user did not mention any chapter, unit, page, or 'whole/entire book', set needs_clarification=True and ask one brief question listing 2-3 example chapters/sections from the document.\n"
        "- If clarification WAS already asked or the user specified a chapter/section or 'entire/whole book', set needs_clarification=False."
    )

    decision: ScopeDecision = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=user_message)]
    )

    if decision.needs_clarification and not prior_clarification_asked:
        q = decision.clarifying_question or (
            f"Got it! Which chapter or section of **{doc_name}** should I check this video against "
            f"(for example: *{available_titles[0] if available_titles else 'Chapter 1'}*), "
            f"or would you like me to scan across the **entire book**?"
        )
        return {
            "needs_clarification": True,
            "clarifying_question": q,
            "target_scope": "",
            "scoped_section_titles": [],
        }

    if decision.is_full_book or not decision.requested_keywords:
        return {
            "needs_clarification": False,
            "clarifying_question": "",
            "target_scope": "Entire Document",
            "scoped_section_titles": available_titles,
        }

    # Match requested chapter/section keywords against actual document section titles
    # Replace the chapter-matching loop in resolve_audit_scope() with:


    matched: List[str] = []
    for kw in decision.requested_keywords:
        kw_low = kw.lower().strip()
        num_match = re.search(r"\b(\d+)\b", kw_low)
        for title in available_titles:
            t_low = title.lower().strip()
            if num_match:
                ch_num = num_match.group(1)
                # Match "Chapter 1 ...", "Unit 1 ...", or sub-sections "1.1 ...", "1.2 ..." ONLY
                if re.match(rf"^(chapter|unit|module)\s+0?{ch_num}\b", t_low) or re.match(
                    rf"^0?{ch_num}\.\d+\b", t_low
                ):
                    if title not in matched:
                        matched.append(title)
            elif kw_low in t_low and title not in matched:
                matched.append(title)




    if not matched:
        matched = available_titles

    scope_label = ", ".join(decision.requested_keywords)
    return {
        "needs_clarification": False,
        "clarifying_question": "",
        "target_scope": scope_label,
        "scoped_section_titles": matched,
    }
