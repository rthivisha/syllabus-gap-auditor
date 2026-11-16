from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class MissedTopicItem(BaseModel):
    topic_title: str = Field(
        description="Exact section or concept title from the reference document."
    )
    summary: str = Field(
        description="Concise 1-2 sentence explanation of what the book covers that the video omitted."
    )
    section_title: str = Field(
        description="Parent section heading in the uploaded PDF."
    )
    page_number: int = Field(description="Starting page number in the uploaded PDF.")
    chunk_id: str = Field(description="Matched Pinecone child/section chunk ID.")
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 that this topic was genuinely missed."
    )
    closest_video_timestamp: str = Field(
        default="",
        description="Closest [MM:SS] timestamp if partially adjacent, else empty.",
    )
    closest_video_snippet: str = Field(
        default="",
        description="Closest spoken/OCR text if partially adjacent, else empty.",
    )


class WorkerDraftOutput(BaseModel):
    missed_topics: List[MissedTopicItem] = Field(default_factory=list)


class EvaluatorVerdict(BaseModel):
    verdict: str = Field(description="Either 'PASS' or 'FAIL'.")
    confidence_score: float = Field(
        description="Overall audit reliability confidence (0.0 to 1.0)."
    )
    evaluation_notes: str = Field(
        description="Detailed rule-by-rule audit findings and exact remediation instructions if FAIL."
    )
    flagged_topics: List[str] = Field(
        default_factory=list,
        description="Topic titles that failed verification or remain borderline.",
    )


class AuditGraphState(TypedDict, total=False):
    # Inputs
    session_id: str
    user_id: str
    doc_id: str
    doc_name: str
    video_url: str
    user_query: str
    target_scope: str  # e.g., "Chapter 1" or "full_document"
    scoped_section_titles: List[
        str
    ]  # Resolved exact section titles for Pinecone filter

    # Step 2: Multimodal Video Artifacts
    video_id: str
    video_title: str
    duration_sec: float
    video_windows: List[Dict[str, Any]]
    ocr_slides: List[Dict[str, Any]]
    guard_blocked: bool
    guard_reason: str

    # Step 3: Scoped Ground-Truth Retrieval Artifacts
    retrieved_sections: List[Dict[str, Any]]
    candidate_gaps: List[Dict[str, Any]]
    cited_chunks: List[Dict[str, Any]]
    step_timings: Dict[str, Any]

    # Step 4: Bounded Worker -> Evaluator -> Repair Loop
    attempt: int
    max_attempts: int
    draft_missed_topics: List[Dict[str, Any]]
    evaluator_verdict: str  # "PASS" | "FAIL"
    evaluator_confidence: float
    evaluation_notes: str
    flagged_topics: List[str]
    repair_notes: str

    # Step 5 & 6: Final Output, HITL Escalations & Persistence
    final_missed_topics: List[Dict[str, Any]]
    escalated_items: List[Dict[str, Any]]
    final_status: str  # "passed" | "failed" | "blocked"
    run_id: str
    latency_ms: int
    step_timings: Dict[str, Any]
