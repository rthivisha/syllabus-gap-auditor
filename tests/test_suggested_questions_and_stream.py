import asyncio
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import UploadFile
from streamlit.testing.v1 import AppTest

import app.gateway.client as gateway
from app.ingestion.chunking.splitter import (
    _coalesce_child_chunks,
    build_hierarchical_chunks,
)
import app.ingestion.processor as processor
import app.main as main
from app.services.retrieval.pinecone_service import PineconeService


class ChunkCoalescingTests(unittest.TestCase):
    def test_coalesces_1074_short_chunks_into_one_chunk_per_page(self) -> None:
        raw_chunks = [
            {
                "chunk_id": f"chunk-{page}-{chunk}",
                "parent_id": f"parent-{chunk}",
                "section_title": "Design",
                "page_number": page,
                "child_text": "x" * 100,
                "parent_text": "x" * 100,
            }
            for page in range(1, 84)
            for chunk in range(13 if page < 83 else 8)
        ]

        coalesced = _coalesce_child_chunks(raw_chunks)

        self.assertEqual(len(raw_chunks), 1074)
        self.assertEqual(len(coalesced), 83)
        self.assertEqual({chunk["page_number"] for chunk in coalesced}, set(range(1, 84)))
        self.assertTrue(all(len(chunk["child_text"]) <= 1500 for chunk in coalesced))

    def test_builder_keeps_child_citations_on_their_source_page(self) -> None:
        page_text = "A sample page contains enough words for chunking. " * 20
        chunks = build_hierarchical_chunks(
            "doc",
            [
                {
                    "section_title": "Design",
                    "pages": [
                        {"page_number": 1, "text": page_text},
                        {"page_number": 2, "text": page_text},
                    ],
                }
            ],
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk["page_number"] for chunk in chunks], [1, 2])
        self.assertTrue(all(len(chunk["child_text"]) <= 1500 for chunk in chunks))


class PineconeBatchingTests(unittest.TestCase):
    def test_compacted_document_replaces_old_vectors_in_one_upsert(self) -> None:
        service = PineconeService.__new__(PineconeService)
        service.index = MagicMock()
        child_chunks = [
            {
                "chunk_id": f"doc-chunk-{idx}",
                "section_title": "Design",
                "page_number": idx + 1,
                "parent_id": f"doc-parent-{idx}",
                "child_text": f"Chunk {idx}",
                "parent_text": f"Parent {idx}",
            }
            for idx in range(84)
        ]
        section_summaries = [
            {
                "section_idx": 0,
                "section_title": "Design",
                "start_page": 1,
                "summary": "A design section.",
            }
        ]

        def fake_embed_texts(texts, is_query=False):
            self.assertFalse(is_query)
            return [[0.1, 0.2] for _ in texts]

        with patch(
            "app.services.retrieval.pinecone_service.embed_texts",
            side_effect=fake_embed_texts,
        ):
            vector_count = service.upsert_document_vectors(
                "doc", "design.pdf", child_chunks, section_summaries
            )

        self.assertEqual(vector_count, 85)
        service.index.delete.assert_called_once_with(
            filter={"doc_id": {"$eq": "doc"}}
        )
        service.index.upsert.assert_called_once()
        self.assertEqual(
            len(service.index.upsert.call_args.kwargs["vectors"]),
            85,
        )
        self.assertEqual(
            [call[0] for call in service.index.method_calls],
            ["delete", "upsert"],
        )


class SuggestedPromptTests(unittest.TestCase):
    def test_fast_extraction_uses_configured_groq_model_without_retries(self) -> None:
        settings = SimpleNamespace(
            GROQ_API_KEY="test-key",
            FALLBACK_MODEL="fallback-model",
            CF_ACCOUNT_ID=None,
            CF_GATEWAY_ID=None,
            CF_AIG_TOKEN=None,
        )
        with (
            patch.object(gateway, "get_settings", return_value=settings),
            patch.object(gateway, "ChatGroq") as chat_groq,
        ):
            model = gateway.get_extraction_llm(model="llama-3.1-8b-instant")

        self.assertIs(model, chat_groq.return_value)
        self.assertEqual(
            chat_groq.call_args.kwargs["model"], "llama-3.1-8b-instant"
        )
        self.assertEqual(chat_groq.call_args.kwargs["max_retries"], 0)
        self.assertEqual(chat_groq.call_args.kwargs["timeout"], 10.0)

    def test_generates_four_short_document_grounded_prompts(self) -> None:
        response = SimpleNamespace(
            content='["Question one?", "Question two?", "Question three?", "Question four?"]'
        )
        with patch.object(main, "get_extraction_llm") as get_llm:
            get_llm.return_value.invoke.return_value = response
            questions = main._generate_suggested_prompts(
                "sample.pdf",
                [
                    {
                        "page_number": 2,
                        "section_title": "System Design",
                        "child_text": "The excerpt describes system design.",
                    }
                ],
            )

        get_llm.assert_called_once_with()
        self.assertEqual(len(questions), 4)
        self.assertTrue(all(0 < len(question) < 85 for question in questions))

    def test_invalid_model_output_uses_four_bounded_fallbacks(self) -> None:
        with patch.object(main, "get_extraction_llm") as get_llm:
            get_llm.return_value.invoke.return_value = SimpleNamespace(content="not JSON")
            questions = main._generate_suggested_prompts(
                "reference.pdf", [{"section_title": "Architecture"}]
            )

        self.assertEqual(len(questions), 4)
        self.assertTrue(all(0 < len(question) < 85 for question in questions))
        self.assertIn("Architecture", questions[2])


class SuggestedPromptIngestionTests(unittest.TestCase):
    def test_upload_returns_only_after_ingestion_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            upload = UploadFile(
                filename="reference.pdf", file=io.BytesIO(b"%PDF-1.4 test")
            )
            ingestion_result = {
                "status": "ready",
                "doc_id": "reference-id",
                "doc_name": "reference.pdf",
                "page_count": 3,
                "chunk_count": 7,
                "sections": [{"section_title": "Introduction"}],
                "suggested_questions": ["Q1?", "Q2?", "Q3?", "Q4?"],
            }
            with (
                patch.object(
                    main, "get_settings", return_value=SimpleNamespace(DATA_DIR=data_dir)
                ),
                patch.object(main, "_compute_doc_id", return_value="reference-id"),
                patch.object(
                    main,
                    "get_document",
                    return_value={"status": "ready", "index_version": 1},
                ),
                patch.object(
                    main, "ingest_pdf_document", return_value=ingestion_result
                ) as ingest,
            ):
                response = asyncio.run(
                    main.api_upload_document(
                        upload,
                        uploaded_by="test-user",
                    )
                )
                uploaded_bytes = ingest.call_args.args[0].read_bytes()

        self.assertEqual(response["status"], "ready")
        self.assertEqual(response["doc_id"], "reference-id")
        self.assertEqual(response["page_count"], 3)
        self.assertEqual(response["chunk_count"], 7)
        self.assertEqual(response["sections"], ingestion_result["sections"])
        self.assertEqual(response["suggested_questions"], ingestion_result["suggested_questions"])
        self.assertGreaterEqual(response["elapsed_sec"], 0)
        ingest.assert_called_once()
        self.assertEqual(uploaded_bytes, b"%PDF-1.4 test")
        self.assertEqual(ingest.call_args.args[1], "test-user")
        self.assertIs(ingest.call_args.args[2], main._generate_suggested_prompts)

    def test_ready_document_reupload_skips_reindexing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            upload = UploadFile(
                filename="reference.pdf", file=io.BytesIO(b"%PDF-1.4 test")
            )
            existing_document = {
                "status": "ready",
                "doc_id": "reference-id",
                "doc_name": "reference.pdf",
                "index_version": processor.PDF_INDEX_VERSION,
                "page_count": 3,
                "chunk_count": 7,
                "sections": [],
                "suggested_questions": ["Q1?", "Q2?", "Q3?", "Q4?"],
            }
            with (
                patch.object(
                    main, "get_settings", return_value=SimpleNamespace(DATA_DIR=data_dir)
                ),
                patch.object(main, "_compute_doc_id", return_value="reference-id"),
                patch.object(main, "get_document", return_value=existing_document),
                patch.object(main, "ingest_pdf_document") as ingest,
            ):
                response = asyncio.run(
                    main.api_upload_document(upload, uploaded_by="test-user")
                )

        self.assertEqual(response["status"], "ready")
        self.assertEqual(response["suggested_questions"], existing_document["suggested_questions"])
        ingest.assert_not_called()

    def test_prompts_are_saved_with_completed_document_metadata(self) -> None:
        sections = [
            {
                "section_idx": 0,
                "section_title": "Introduction",
                "start_page": 1,
                "end_page": 1,
                "summary": "Overview",
                "pages": [{"page_number": 1, "text": "Overview text."}],
            }
        ]
        chunks = [
            {
                "chunk_id": "doc-c1",
                "section_title": "Introduction",
                "page_number": 1,
                "child_text": "Overview text.",
            }
        ]
        questions = ["Question one?", "Question two?", "Question three?", "Question four?"]

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "reference.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 test")
            processed_dir = Path(temp_dir) / "processed"
            processed_dir.mkdir()
            settings = SimpleNamespace(PROCESSED_BOOKS_DIR=processed_dir)
            save_record = MagicMock()
            vector_service = MagicMock()
            vector_service.upsert_document_vectors.return_value = 2
            generator = MagicMock(return_value=questions)
            with (
                patch.object(processor, "get_settings", return_value=settings),
                patch.object(processor, "load_pdf_sections", return_value={"sections": sections, "page_count": 1}),
                patch.object(processor, "build_hierarchical_chunks", return_value=chunks),
                patch.object(processor, "PineconeService", return_value=vector_service),
                patch.object(processor, "save_document_record", save_record),
                patch.object(processor, "log_audit_event"),
            ):
                result = processor.ingest_pdf_document(
                    pdf_path,
                    "test-user",
                    generator,
                )

            generator.assert_called_once_with("reference.pdf", chunks)
            self.assertEqual(result["suggested_questions"], questions)
            self.assertEqual(result["index_version"], processor.PDF_INDEX_VERSION)
            self.assertEqual(
                save_record.call_args.kwargs["suggested_questions"], questions
            )
            self.assertEqual(
                save_record.call_args.kwargs["index_version"],
                processor.PDF_INDEX_VERSION,
            )
            saved_json = json.loads(
                (processed_dir / f"{result['doc_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved_json["suggested_questions"], questions)


class ChatStreamTests(unittest.TestCase):
    def test_stream_emits_stage_updates_and_final_payload(self) -> None:
        request = main.ChatTurnRequest(
            session_id="stream-session-test",
            user_id="stream-user-test",
            message="hello",
        )

        def finish_turn(_request):
            main.set_job_stage(request.session_id, "2/4 Retrieving evidence...")
            time.sleep(0.35)
            return {"type": "chat", "reply": "hello", "awaiting_scope": False}

        with patch.object(main, "api_chat_turn", side_effect=finish_turn):
            response = main.api_chat_turn_stream(request)

            async def collect_events():
                return [
                    chunk async for chunk in response.body_iterator
                ]

            chunks = asyncio.run(collect_events())

        events = [
            json.loads(line[6:])
            for chunk in chunks
            for line in (chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk).splitlines()
            if line.startswith("data: ")
        ]
        self.assertIn(
            {"type": "stage", "text": "2/4 Retrieving evidence..."}, events
        )
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["payload"]["reply"], "hello")


class StreamlitStreamingUITests(unittest.TestCase):
    def test_uploaded_document_metadata_populates_header_and_question_pills(self) -> None:
        class ApiResponse:
            def __init__(self, data, status=200):
                self._data = data
                self.status_code = status
                self.ok = status < 400
                self.text = ""

            def json(self):
                return self._data

        document = {
            "doc_id": "uploaded-doc",
            "doc_name": "uploaded.pdf",
            "page_count": 9,
            "chunk_count": 14,
            "status": "ready",
            "sections": [],
            "suggested_questions": [
                "Question 1?",
                "Question 2?",
                "Question 3?",
                "Question 4?",
            ],
        }
        upload_state = {
            "posted": False,
            "post_count": 0,
            "document_fetch_count": 0,
        }

        def fake_get(url, **_kwargs):
            if url.endswith("/documents"):
                upload_state["document_fetch_count"] += 1
                return ApiResponse([document] if upload_state["posted"] else [])
            if "/quota/" in url:
                return ApiResponse({"used": 0, "limit": 3, "remaining": 3})
            if url.endswith("/sessions"):
                return ApiResponse([])
            if url.endswith("/approvals"):
                return ApiResponse([])
            if url.endswith("/metrics"):
                return ApiResponse({})
            return ApiResponse({"status": "idle"})

        def fake_post(url, **_kwargs):
            self.assertTrue(url.endswith("/documents/upload"))
            upload_state["posted"] = True
            upload_state["post_count"] += 1
            return ApiResponse(
                {
                    "status": "ready",
                    "doc_id": "uploaded-doc",
                    "doc_name": "uploaded.pdf",
                    "page_count": document["page_count"],
                    "chunk_count": document["chunk_count"],
                    "suggested_questions": document["suggested_questions"],
                }
            )

        with (
            patch("requests.get", side_effect=fake_get),
            patch("requests.post", side_effect=fake_post),
        ):
            ui_script = Path(__file__).resolve().parents[1] / "ui" / "app.py"
            app = AppTest.from_file(str(ui_script), default_timeout=30).run()
            app.file_uploader[0].upload(
                "uploaded.pdf", b"%PDF-1.4 test", "application/pdf"
            ).run()
            refresh_button = next(
                button for button in app.button if button.label == "↻ Refresh documents"
            )
            refresh_button.click().run()

        self.assertEqual(upload_state["post_count"], 1)
        self.assertEqual(upload_state["document_fetch_count"], 2)
        self.assertEqual(app.session_state["selected_doc_id"], "uploaded-doc")
        self.assertEqual(app.session_state["selected_doc_pages"], 9)
        self.assertEqual(app.session_state["selected_doc_chunks"], 14)
        self.assertEqual(
            app.session_state["suggested_questions"], document["suggested_questions"]
        )
        self.assertEqual(
            len([button for button in app.button if button.label.startswith("💡 Question")]),
            4,
        )
        self.assertTrue(any("READY FOR AUDIT" in item.value for item in app.markdown))
        self.assertFalse(app.exception, [item.message for item in app.exception])

    def test_suggested_prompt_streams_and_renders_verified_source(self) -> None:
        class ApiResponse:
            def __init__(self, data, status=200):
                self._data = data
                self.status_code = status
                self.ok = status < 400
                self.text = ""

            def json(self):
                return self._data

        class StreamResponse:
            status_code = 200
            ok = True

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def iter_lines(self, decode_unicode=True):
                yield (
                    'data: {"type":"done","payload":{"type":"audit_report",'
                    '"reply":"Verified result","message_id":"message-1",'
                    '"awaiting_scope":false,"citations":[{"page_number":2,'
                    '"section_title":"Design","chunk_id":"chunk-42",'
                    '"snippet":"Verbatim evidence","bi_encoder_score":0.92}]}}'
                )

        document = {
            "doc_id": "streamlit-doc",
            "doc_name": "book.pdf",
            "page_count": 3,
            "chunk_count": 4,
            "status": "ready",
            "sections": [],
            "suggested_questions": [
                "Question 1?",
                "Question 2?",
                "Question 3?",
                "Question 4?",
            ],
        }

        def fake_get(url, **_kwargs):
            if url.endswith("/documents"):
                return ApiResponse([document])
            if "/quota/" in url:
                return ApiResponse({"used": 0, "limit": 3, "remaining": 3})
            if url.endswith("/sessions"):
                return ApiResponse([])
            if url.endswith("/approvals"):
                return ApiResponse([])
            if url.endswith("/metrics"):
                return ApiResponse({})
            return ApiResponse({"status": "idle"})

        def fake_post(url, **kwargs):
            self.assertTrue(url.endswith("/chat/turn/stream"))
            self.assertTrue(kwargs.get("stream"))
            return StreamResponse()

        with (
            patch("requests.get", side_effect=fake_get) as get_request,
            patch("requests.post", side_effect=fake_post),
        ):
            ui_script = Path(__file__).resolve().parents[1] / "ui" / "app.py"
            app = AppTest.from_file(str(ui_script), default_timeout=30).run()
            app.session_state["selected_doc_id"] = "streamlit-doc"
            app.session_state["selected_doc_name"] = "book.pdf"
            app.run()
            question_buttons = [
                button
                for button in app.button
                if button.label.startswith("💡 Question")
            ]
            self.assertEqual(len(question_buttons), 4)
            next(
                button for button in question_buttons if button.label == "💡 Question 1?"
            ).click().run()

        self.assertTrue(
            any(message.get("content") == "Verified result" for message in app.session_state["messages"])
        )
        self.assertTrue(any("READY FOR AUDIT" in item.value for item in app.markdown))
        self.assertTrue(
            any(
                "Match 92%" in item.value and "Chunk: chunk-42" in item.value
                for item in app.markdown
            )
        )
        self.assertFalse(
            any(call.args[0].endswith("/status") for call in get_request.call_args_list)
        )
        self.assertFalse(app.exception, [item.message for item in app.exception])


if __name__ == "__main__":
    unittest.main()
