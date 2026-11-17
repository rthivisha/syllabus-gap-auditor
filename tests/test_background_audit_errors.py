import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from langgraph.types import Command

import app.main as main


class BackgroundAuditErrorTests(unittest.TestCase):
    def test_startup_preloads_local_models(self) -> None:
        with (
            patch.object(main, "init_db") as init_db,
            patch.object(main, "get_embedder") as get_embedder,
            patch.object(main, "get_flashranker") as get_flashranker,
        ):
            main.on_startup()
        init_db.assert_called_once_with()
        get_embedder.assert_called_once_with()
        get_flashranker.assert_called_once_with()

    def test_scope_prompt_uses_requested_active_document(self) -> None:
        request = main.ChatTurnRequest(
            session_id="scope-session",
            user_id="scope-user",
            doc_id="uploaded-doc",
            message="https://youtu.be/dQw4w9WgXcQ",
        )
        document = {
            "doc_id": "uploaded-doc",
            "doc_name": "harness-engineering.pdf",
            "page_count": 24,
            "sections": [
                {"section_title": "Harness Architecture"},
                {"section_title": "Agent Evaluation"},
            ],
        }
        with (
            patch.object(main, "get_document", return_value=document) as get_doc,
            patch.object(main, "save_chat_message", return_value="message-id"),
        ):
            result = main._process_chat_turn(request)
        get_doc.assert_called_once_with("uploaded-doc")
        self.assertIn("harness-engineering.pdf", result["reply"])
        self.assertIn("Harness Architecture", result["reply"])
        self.assertTrue(result["awaiting_scope"])

    def test_graph_resume_requires_paused_guide_node(self) -> None:
        turn_input = {"message": "continue"}
        config = {"configurable": {"thread_id": "session-1"}}
        with (
            patch.object(main.audit_graph, "get_state", return_value=None),
            patch.object(main.audit_graph, "invoke", return_value={}) as invoke,
        ):
            main._invoke_graph_turn(turn_input, "session-1")
        invoke.assert_called_once_with(turn_input, config=config)

        with (
            patch.object(
                main.audit_graph,
                "get_state",
                return_value=SimpleNamespace(next=("await_user_input",)),
            ),
            patch.object(main.audit_graph, "invoke", return_value={}) as invoke,
        ):
            main._invoke_graph_turn(turn_input, "session-1")
        command = invoke.call_args.args[0]
        self.assertIsInstance(command, Command)
        self.assertEqual(command.resume, turn_input)
        self.assertEqual(invoke.call_args.kwargs, {"config": config})

    def test_background_exception_is_reported_to_chat_and_job_stage(self) -> None:
        request = main.ChatTurnRequest(
            session_id="session-error-test",
            user_id="user-error-test",
            message="https://youtu.be/example",
            doc_id="doc-error-test",
        )
        job_id = "job-error-test"
        with main.ACTIVE_JOBS_LOCK:
            main.ACTIVE_JOB_IDS[request.session_id] = job_id
            main.ACTIVE_JOB_RESULTS[request.session_id] = {}

        output = io.StringIO()
        try:
            with (
                patch.object(
                    main,
                    "_process_chat_turn",
                    side_effect=ValueError("test failure detail"),
                ),
                patch.object(main, "save_chat_message") as save_message,
                patch.object(main, "set_job_stage") as set_stage,
                contextlib.redirect_stdout(output),
            ):
                main._run_audit_background(request, job_id)

            error_message = save_message.call_args.kwargs["content"]
            self.assertIn("ValueError", error_message)
            self.assertIn("test failure detail", error_message)
            self.assertIn("[BACKGROUND AUDIT ERROR]", output.getvalue())
            self.assertIn("ValueError: test failure detail", output.getvalue())
            self.assertEqual(
                set_stage.call_args,
                call(request.session_id, error_message, status="error"),
            )
        finally:
            with main.ACTIVE_JOBS_LOCK:
                main.ACTIVE_JOB_IDS.pop(request.session_id, None)
                main.ACTIVE_JOB_RESULTS.pop(request.session_id, None)
                main.ACTIVE_JOBS.pop(request.session_id, None)
                main.ACTIVE_JOB_USERS.pop(request.session_id, None)


if __name__ == "__main__":
    unittest.main()
