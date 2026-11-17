# Syllabus Gap Audit — Verification & Self-Correction Policy (`skills.md`)

## 1. Core Objective: Missing-Topic Detection
- Identify concrete concepts, architectural rules, and mechanisms present in retrieved textbook chunks that the video skipped or only mentioned without explaining how they work.
- Never invent topics outside `retrieved_evidence`. Every missed topic must cite the exact `chunk_id`, `section_title`, and `page_number` from retrieved evidence.
- Treat explanations on slides and clear spoken synonyms as covered; a bare mention without an explanation does not establish coverage.

## 2. Evaluator Verification Gates (`evaluator_node`)
1. **Citation grounding:** Every item in `drafted_gaps` must exactly match a `chunk_id`, `section_title`, and `page_number` in `retrieved_evidence`.
2. **Completeness / anti-leniency:** If at least three evidence records were retrieved and `drafted_gaps` is empty, fail the audit and instruct `repair_node` to re-examine the retrieved sections for specific mechanisms omitted from the video. An empty draft is a failure even when fewer than three records were retrieved.
3. **Fast pass:** When the draft is non-empty and every citation exactly matches retrieved evidence, pass without an evaluator LLM call.

## 3. Single Targeted Repair Pass (`repair_node`)
- When `evaluator_passed = False`, run at most one repair pass.
- Read the evaluator's `evaluation_notes`, the original retrieved evidence, and this policy. Fix only flagged citations and add only genuine missed concepts supported by retrieved evidence.
- Preserve unflagged valid items, correct citations only when the evidence identifies the exact chunk, and discard any item that cannot be grounded. Never use a new retrieval result or outside knowledge as a substitute for the original evidence.
- Write the same repaired list to `drafted_gaps` and `missed_topics` (and the pipeline's `draft_missed_topics` alias).
- Do not re-enter repair after this pass. If a concern remains, leave it for the existing finalization/HITL path.

## 4. Output Evidence
- Each missed topic must include `chunk_id`, `section_title`, `topic_title`, `page_number`, an explanation/description, and the closest checked video timestamp.
- Keep each explanation concise and distinguish an unexplained mention from a fully covered concept.
