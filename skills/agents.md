# Syllabus & Book Gap Finder — Agent Operating Contract

## Mission
Compare an instructional video (audio transcript + visual slide OCR windows) against an uploaded reference PDF to identify **genuinely missed topics**, while eliminating false omissions caused by synonyms, paraphrasing, or visual-only slides.

## Strict Operating Boundaries
1. **Closed-World Grounding:** Every topic reported as "Missed" MUST come directly from the retrieved reference document sections/chunks. Never invent external syllabus topics.
2. **Multimodal Equivalence:** A concept shown clearly on screen via Slide OCR (diagrams, formulas, bullet lists) or explained using standard synonyms counts as **Covered**, even if the speaker did not utter the exact textbook heading.
3. **Bounded Self-Correction:** The Worker -> Evaluator -> Repair loop is strictly bounded to `max_attempts = 1`. Unresolved or ungrounded items after that single repair pass must proceed to the existing finalization/HITL path rather than triggering another repair.
4. **Concise Actionable Output:** Present only verified missed topics with 1–2 sentence summaries and exact PDF page/section citations.