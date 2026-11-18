import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from pydantic import BaseModel

import app.agents.nodes.pdf_qa as pdf_qa
import app.agents.graph as graph
import app.gateway.client as gateway
import app.main as main
import app.services.db.repository as repository


class IntentRoutingTests(unittest.TestCase):
    def test_routes_pdf_questions_and_project_questions_separately(self) -> None:
        self.assertEqual(
            main._classify_intent("Tell me about this project", True, False),
            "project_guide",
        )
        self.assertEqual(
            graph.route_entry_intent(
                {"doc_id": "doc-1", "user_query": "Tell me about this project"}
            ),
            "project_guide",
        )
        self.assertEqual(
            graph.route_entry_intent(
                {"doc_id": "doc-1", "user_query": "What does the textbook say about testing?"}
            ),
            "pdf_qa",
        )
        self.assertEqual(
            graph.route_entry_intent(
                {"doc_id": "doc-1", "user_query": "How does the app use Pinecone?"}
            ),
            "project_guide",
        )
        self.assertEqual(
            graph.route_entry_intent(
                {
                    "doc_id": "doc-1",
                    "user_query": "https://youtu.be/example",
                }
            ),
            "media_ingestion",
        )

    def test_project_guide_drops_stale_pdf_sources(self) -> None:
        request = main.ChatTurnRequest(
            session_id="project-session",
            doc_id="doc-1",
            message="Tell me about this project",
        )
        stale_source = {"chunk_id": "old-chunk", "page_number": 2}
        document = {"doc_id": "doc-1", "doc_name": "book.pdf"}
        with (
            patch.object(main, "get_document", return_value=document),
            patch.object(main, "save_chat_message", return_value="message-id") as save,
            patch.object(
                main,
                "_invoke_graph_turn",
                return_value={
                    "route_taken": "project_guide",
                    "chat_response": "This project compares lectures and PDFs.",
                    "retrieved_evidence": [stale_source],
                    "llm_used": True,
                },
            ),
        ):
            result = main._process_chat_turn(request)

        self.assertEqual(result["reply"], "This project compares lectures and PDFs.")
        self.assertEqual(result["citations"], [])
        self.assertEqual(save.call_args.kwargs["citations"], [])


class ByokGatewayTests(unittest.TestCase):
    def test_user_keys_override_server_keys_and_bypass_gateway_routing(self) -> None:
        settings = SimpleNamespace(
            GEMINI_API_KEY="server-gemini",
            GROQ_API_KEY="server-groq",
            AUDITOR_MODEL="gemini-test",
            FALLBACK_MODEL="groq-test",
            CF_ACCOUNT_ID="account",
            CF_GATEWAY_ID="gateway",
            CF_AIG_TOKEN="gateway-token",
        )
        with (
            patch.object(gateway, "get_settings", return_value=settings),
            patch.object(gateway, "ChatGoogleGenerativeAI") as gemini,
            patch.object(gateway, "ChatGroq") as groq,
            gateway.use_user_api_keys("  user-gemini  ", "  user-groq  "),
        ):
            gateway.get_extraction_llm()

        self.assertEqual(gemini.call_args.kwargs["google_api_key"], "user-gemini")
        self.assertNotIn("client_options", gemini.call_args.kwargs)
        self.assertNotIn("additional_headers", gemini.call_args.kwargs)
        self.assertEqual(groq.call_args.kwargs["api_key"], "user-groq")
        self.assertNotIn("base_url", groq.call_args.kwargs)
        self.assertNotIn("default_headers", groq.call_args.kwargs)

    def test_structured_auditor_uses_request_provider_keys(self) -> None:
        class OutputSchema(BaseModel):
            answer: str

        settings = SimpleNamespace(
            GEMINI_API_KEY="server-gemini",
            GROQ_API_KEY="server-groq",
            AUDITOR_MODEL="gemini-test",
            FALLBACK_MODEL="groq-test",
            CF_ACCOUNT_ID="account",
            CF_GATEWAY_ID="gateway",
            CF_AIG_TOKEN="gateway-token",
        )
        with (
            patch.object(gateway, "get_settings", return_value=settings),
            patch.object(gateway, "ChatGoogleGenerativeAI") as gemini,
            patch.object(gateway, "ChatGroq") as groq,
            gateway.use_user_api_keys("user-gemini", "user-groq"),
        ):
            gateway.get_structured_auditor_llm(OutputSchema)

        self.assertEqual(gemini.call_args.kwargs["google_api_key"], "user-gemini")
        self.assertNotIn("client_options", gemini.call_args.kwargs)
        self.assertEqual(groq.call_args.kwargs["api_key"], "user-groq")
        self.assertNotIn("base_url", groq.call_args.kwargs)

    def test_project_guide_receives_request_key_without_checkpoint_state(self) -> None:
        settings = SimpleNamespace(
            GEMINI_API_KEY="server-gemini",
            GROQ_API_KEY=None,
            AUDITOR_MODEL="gemini-test",
            FALLBACK_MODEL="groq-test",
            CF_ACCOUNT_ID="account",
            CF_GATEWAY_ID="gateway",
            CF_AIG_TOKEN="gateway-token",
        )
        gemini_model = MagicMock()
        gemini_model.invoke.return_value = SimpleNamespace(
            content="Apply a YouTube URL and select a reference PDF."
        )
        state = {
            "user_query": "What is the project architecture?",
            "doc_id": "doc-1",
            "doc_name": "book.pdf",
        }
        with (
            patch.object(gateway, "get_settings", return_value=settings),
            patch.object(gateway, "ChatGoogleGenerativeAI", return_value=gemini_model) as gemini,
            patch.object(gateway, "ChatGroq", return_value=None),
            gateway.use_user_api_keys("request-gemini", None),
        ):
            workflow = graph.build_audit_graph()
            result = workflow.invoke(
                state,
                config={"configurable": {"thread_id": "byok-graph-test"}},
            )

        self.assertEqual(
            gemini.call_args.kwargs["google_api_key"], "request-gemini"
        )
        self.assertEqual(result["route_taken"], "project_guide")
        self.assertNotIn("user_gemini_key", result)
        self.assertNotIn("user_gemini_key", state)


class PdfQuestionAnswerTests(unittest.TestCase):
    def test_answers_with_retrieved_citations_and_follow_up(self) -> None:
        match = {
            "chunk_id": "chunk-7",
            "doc_name": "book.pdf",
            "section_title": "Testing",
            "page_number": 12,
            "parent_text": "The source describes a layered testing strategy.",
            "rerank_score": 0.91,
        }
        model = MagicMock()
        model.invoke.return_value = SimpleNamespace(
            content="The book recommends layered testing [Source 1]."
        )
        with (
            patch.object(
                pdf_qa.PineconeService,
                "search_and_rerank",
                return_value=[match],
            ) as search,
            patch.object(pdf_qa, "get_extraction_llm", return_value=model),
        ):
            result = pdf_qa.pdf_qa_node(
                {"doc_id": "doc-1", "doc_name": "book.pdf", "user_query": "How is testing described?"}
            )

        search.assert_called_once_with(
            doc_id="doc-1",
            query="How is testing described?",
            top_k_dense=18,
            top_n_final=5,
        )
        self.assertIn("[Source 1]", result["chat_response"])
        self.assertIn("YouTube lecture", result["chat_response"])
        self.assertEqual(result["retrieved_evidence"][0]["page_number"], 12)
        self.assertEqual(result["route_taken"], "pdf_qa")
        self.assertTrue(result["llm_used"])

    def test_empty_retrieval_returns_without_calling_model(self) -> None:
        with (
            patch.object(pdf_qa.PineconeService, "search_and_rerank", return_value=[]),
            patch.object(pdf_qa, "get_extraction_llm") as get_llm,
        ):
            result = pdf_qa.pdf_qa_node(
                {"doc_id": "doc-1", "user_query": "What does it say?"}
            )

        get_llm.assert_not_called()
        self.assertFalse(result["llm_used"])
        self.assertEqual(result["retrieved_evidence"], [])


class UnifiedQuotaTests(unittest.TestCase):
    def test_error_redaction_handles_trimmed_and_blank_keys(self) -> None:
        request = main.ChatTurnRequest(
            session_id="redaction-session",
            message="test",
            user_gemini_key="  private-key  ",
            user_groq_key="   ",
        )

        self.assertEqual(
            main._redact_sensitive_text("provider rejected private-key", request),
            "provider rejected [REDACTED]",
        )

    def test_successful_guest_llm_turn_is_logged_without_request_secrets(self) -> None:
        request = main.ChatTurnRequest(
            session_id="quota-session",
            user_id="quota-user",
            message="Explain this document",
        )
        with (
            patch.object(main, "_request_may_use_llm", return_value=True),
            patch.object(main, "_reserve_llm_quota", return_value="reservation") as reserve,
            patch.object(
                main,
                "_process_chat_turn",
                return_value={
                    "type": "pdf_qa",
                    "_llm_used": True,
                    "_route_taken": "pdf_qa",
                },
            ),
            patch.object(main, "log_audit_event") as log_event,
        ):
            result = main._process_chat_turn_with_quota(request)

        reserve.assert_called_once_with("quota-user")
        log_event.assert_called_once_with(
            action="llm_turn_completed",
            run_id=None,
            user_id="quota-user",
            detail={"route": "pdf_qa"},
        )
        self.assertNotIn("_llm_used", result)

    def test_byok_skips_guest_quota_and_completion_logging(self) -> None:
        request = main.ChatTurnRequest(
            session_id="byok-session",
            user_id="byok-user",
            message="Explain this document",
            user_gemini_key="private-key",
        )
        with (
            patch.object(main, "_request_may_use_llm", return_value=True),
            patch.object(main, "_reserve_llm_quota") as reserve,
            patch.object(
                main,
                "_process_chat_turn",
                return_value={"type": "chat", "_llm_used": True},
            ),
            patch.object(main, "log_audit_event") as log_event,
        ):
            main._process_chat_turn_with_quota(request)

        reserve.assert_not_called()
        log_event.assert_not_called()

    def test_five_completed_turns_reject_another_guest_llm_call(self) -> None:
        request = main.ChatTurnRequest(
            session_id="limited-session",
            user_id="limited-user",
            message="Explain this document",
        )
        with (
            patch.object(main, "_request_may_use_llm", return_value=True),
            patch.object(main, "_get_total_llm_turns", return_value=5),
            patch.object(main, "_active_user_reservations", return_value=0),
            patch.object(main, "_process_chat_turn") as process_turn,
        ):
            with self.assertRaises(HTTPException) as error:
                main._process_chat_turn_with_quota(request)

        self.assertEqual(error.exception.status_code, 429)
        process_turn.assert_not_called()

    def test_quota_counter_includes_legacy_audits_and_other_llm_turns(self) -> None:
        db = MagicMock()
        db.scalar.side_effect = [3, 2]
        with patch.object(repository, "get_db_session", return_value=contextlib.nullcontext(db)):
            count = repository.count_completed_llm_turns("user-1")

        self.assertEqual(count, 5)
        self.assertEqual(db.scalar.call_count, 2)


if __name__ == "__main__":
    unittest.main()
