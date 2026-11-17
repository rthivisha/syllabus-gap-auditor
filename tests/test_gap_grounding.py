import unittest
from unittest.mock import patch

from app.agents.nodes.evaluator import evaluator_critic_node
from app.agents.nodes.gap_auditor import gap_auditor_worker_node
from app.agents.nodes.grounding import match_citation, normalize_citation_records
from app.agents.nodes.repair import BatchAuditResult, targeted_repair_node
from app.agents.nodes.syllabus_retriever import _select_gap_candidates
from app.agents.state import AuditedTopicItem, MissedTopicItem, WorkerDraftOutput


class CitationGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "chunk_id": "chunk-1",
            "section_title": "Concurrency",
            "page_number": 12,
            "parent_text": "The system batches concurrent tool calls.",
        }

    def test_exact_citation_matches_retrieved_evidence(self) -> None:
        evidence = normalize_citation_records([self.record])
        self.assertIsNotNone(
            match_citation(
                {
                    "chunk_id": "chunk-1",
                    "section_title": "Concurrency",
                    "page_number": 12,
                },
                evidence,
            )
        )

    def test_nonmatching_page_is_not_grounded(self) -> None:
        evidence = normalize_citation_records([self.record])
        self.assertIsNone(
            match_citation(
                {
                    "chunk_id": "chunk-1",
                    "section_title": "Concurrency",
                    "page_number": 13,
                },
                evidence,
            )
        )

    def test_candidate_filter_keeps_lowest_eight_below_threshold(self) -> None:
        candidates = [
            {"chunk_id": str(index), "best_video_similarity": score}
            for index, score in enumerate(
                [0.71, 0.1, 0.72, 0.5, 0.3, 0.6, 0.2, 0.4, 0.65, 0.7, 0.11]
            )
        ]
        result = _select_gap_candidates(candidates)
        expected = sorted(
            item["best_video_similarity"]
            for item in candidates
            if item["best_video_similarity"] < 0.72
        )[:8]
        self.assertEqual(len(result), 8)
        self.assertTrue(all(item["best_video_similarity"] < 0.72 for item in result))
        self.assertEqual(
            [item["best_video_similarity"] for item in result],
            expected,
        )

    def test_repair_can_correct_metadata_from_exact_chunk_id(self) -> None:
        evidence = normalize_citation_records([self.record])
        matched = match_citation(
            {
                "chunk_id": "chunk-1",
                "section_title": "Wrong section",
                "page_number": 99,
            },
            evidence,
            allow_correction=True,
        )
        self.assertEqual(matched["section_title"], "Concurrency")
        self.assertEqual(matched["page_number"], 12)

    @patch("app.agents.nodes.gap_auditor.set_job_stage")
    @patch("app.agents.nodes.gap_auditor._audit_single_batch")
    def test_worker_populates_all_gap_aliases_with_exact_citation(
        self, audit_batch, _set_job_stage
    ) -> None:
        audit_batch.return_value = WorkerDraftOutput(
            missed_topics=[
                AuditedTopicItem(
                    topic_title="Concurrent batching",
                    summary="Explains how concurrent calls are grouped.",
                    section_title="Concurrency",
                    page_number=12,
                    chunk_id="chunk-1",
                    confidence=0.9,
                    evidence_quote="batches concurrent tool calls",
                    video_check_note="01:20",
                )
            ]
        )
        result = gap_auditor_worker_node(
            {
                "candidate_gaps": [
                    {
                        **self.record,
                        "best_video_similarity": 0.3,
                        "closest_timestamp": "01:20",
                        "closest_video_text": "",
                    }
                ]
            }
        )
        self.assertIs(result["draft_missed_topics"], result["drafted_gaps"])
        self.assertIs(result["draft_missed_topics"], result["missed_topics"])
        item = result["drafted_gaps"][0]
        self.assertEqual(item["chunk_id"], "chunk-1")
        self.assertEqual(item["section_title"], "Concurrency")
        self.assertEqual(item["page_number"], 12)
        self.assertEqual(item["closest_timestamp"], "01:20")
        self.assertEqual(item["description"], item["explanation"])

    @patch("app.agents.nodes.repair.set_job_stage")
    @patch("app.agents.nodes.repair.get_structured_auditor_llm")
    def test_repair_preserves_unflagged_grounded_topics(
        self, get_llm, _set_job_stage
    ) -> None:
        class FakeLLM:
            def invoke(self, _messages):
                return BatchAuditResult(
                    repaired_missed_topics=[
                        MissedTopicItem(
                            topic_title="Flagged topic",
                            summary="A grounded explanation.",
                            section_title="Wrong section",
                            page_number=99,
                            chunk_id="chunk-1",
                            confidence=0.9,
                            evidence_quote="supported evidence",
                            video_check_note="01:20",
                        )
                    ],
                    repair_notes="Corrected the flagged citation.",
                )

        get_llm.return_value = FakeLLM()
        second_record = {
            "chunk_id": "chunk-2",
            "section_title": "Recovery",
            "page_number": 14,
            "summary": "The system recovers from errors.",
        }
        unflagged = {
            "topic_title": "Unflagged topic",
            "summary": "An existing grounded explanation.",
            "chunk_id": "chunk-2",
            "section_title": "Recovery",
            "page_number": 14,
        }
        flagged = {
            "topic_title": "Flagged topic",
            "summary": "Needs a citation correction.",
            "chunk_id": "chunk-1",
            "section_title": "Wrong section",
            "page_number": 99,
        }
        result = targeted_repair_node(
            {
                "attempt": 0,
                "evaluation_notes": "Correct the flagged citation.",
                "flagged_topics": ["Flagged topic"],
                "draft_missed_topics": [unflagged, flagged],
                "retrieved_evidence": [self.record, second_record],
            }
        )
        self.assertEqual(
            [item["topic_title"] for item in result["draft_missed_topics"]],
            ["Unflagged topic", "Flagged topic"],
        )
        self.assertEqual(result["draft_missed_topics"][0]["page_number"], 14)
        self.assertEqual(result["draft_missed_topics"][1]["page_number"], 12)

    @patch("app.agents.nodes.evaluator.set_job_stage")
    def test_grounded_draft_passes_without_repair(self, _set_job_stage) -> None:
        item = {
            "chunk_id": "chunk-1",
            "section_title": "Concurrency",
            "page_number": 12,
            "topic_title": "Concurrent batching",
        }
        result = evaluator_critic_node(
            {
                "retrieved_evidence": [self.record],
                "drafted_gaps": [item],
            }
        )
        self.assertTrue(result["evaluator_passed"])
        self.assertEqual(result["evaluator_verdict"], "PASS")
        self.assertEqual(result["step_timings"]["evaluator_sec"], 0.0)
        self.assertIs(result["draft_missed_topics"], result["missed_topics"])

    @patch("app.agents.nodes.evaluator.set_job_stage")
    def test_empty_draft_with_retrieved_sections_fails(self, _set_job_stage) -> None:
        result = evaluator_critic_node(
            {
                "retrieved_evidence": [
                    {**self.record, "chunk_id": f"chunk-{index}"}
                    for index in range(3)
                ],
                "drafted_gaps": [],
            }
        )
        self.assertFalse(result["evaluator_passed"])
        self.assertEqual(result["evaluator_verdict"], "FAIL")
        self.assertIn("specific mechanisms", result["evaluation_notes"])


if __name__ == "__main__":
    unittest.main()
