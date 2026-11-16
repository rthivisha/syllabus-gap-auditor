# Evaluation Rubric (`skills.md`) — Gap Audit Quality Gate

Evaluate the Worker's drafted Missed Topics list against every criterion below. Return `PASS` only if all 5 rules succeed; otherwise return `FAIL` with specific, actionable `evaluation_notes` naming the exact topic and fix required.

## Rule 1: Synonym & Visual Slide OCR Verification (`RULE_01_MULTIMODAL_CHECK`)
- Cross-check every candidate "Missed" topic against BOTH the 25-second spoken transcript windows AND the extracted Slide OCR text.
- If a term, formula, or obvious synonym appears in Slide OCR or spoken audio with supporting context (cosine/rerank score >= 0.65), **FAIL** and instruct the Repair node to remove it from the Missed list.

## Rule 2: Document Grounding & Zero Hallucination (`RULE_02_GROUNDEDNESS`)
- Every listed missed topic must map to an actual retrieved chunk (`chunk_id`, `section_title`, `page_number`) from the target `doc_id` and user-requested scope.
- If any topic cannot be traced to retrieved document text, **FAIL** and instruct the Repair node to drop the ungrounded item.

## Rule 3: Borderline Ambiguity Escalation (`RULE_03_AMBIGUITY_BOUNDARY`)
- If a topic has weak/partial evidence (`score` between `0.42` and `0.55`) where coverage is genuinely ambiguous:
  - On `attempt < max_attempts`: **FAIL** and trigger a targeted keyword re-retrieval query against Pinecone + raw Slide OCR.
  - On `attempt == max_attempts`: Flag the item as `needs_human_review` for the `Approvals` HITL queue rather than forcing a binary guess.

## Rule 4: Output Focus & Conciseness (`RULE_04_FORMAT_FOCUS`)
- Output must list **only missed topics** (do not pad with long summaries of covered topics), each with a clear 1–2 sentence description, `section_title`, and `page_number`.

## Rule 5: Concept-Level Granularity (`RULE_05_CONCEPT_GRANULARITY`)
- Never excuse missing textbook mechanisms because the lecture is introductory.
- When a supplied book section teaches concrete implementation rules or architectural mechanisms, flag them if absent from both transcript and slide OCR; examples include Bash/Git safety constraints, concurrency-safe tool batching, and error/recovery invariants.
- If candidate gaps contain clearly missing mechanisms but the Worker returns no missed topics, fail the audit and identify the omitted mechanisms.