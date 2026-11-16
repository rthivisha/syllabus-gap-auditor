import json
from pathlib import Path
from app.agents.graph import audit_graph
from app.config import get_settings
from app.services.db.repository import get_document
from evals.metrics import compute_gap_metrics


def run_eval_suite(dataset_path: str = "evals/golden_dataset.json") -> None:
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    settings = get_settings()

    for case in cases:
        doc = get_document(case["doc_id"])
        if not doc:
            print(f"[SKIP] doc_id={case['doc_id']} not found in Neon PostgreSQL.")
            continue

        sec_titles = [s["section_title"] for s in doc["sections"]]
        result = audit_graph.invoke(
            {
                "session_id": f"eval-{case['case_id']}",
                "user_id": "eval_runner",
                "doc_id": doc["doc_id"],
                "doc_name": doc["doc_name"],
                "video_url": case["video_url"],
                "user_query": f"Audit against {case['target_scope']}",
                "target_scope": case["target_scope"],
                "scoped_section_titles": sec_titles,
                "attempt": 0,
                "max_attempts": settings.MAX_CRITIC_RETRIES,
            }
        )

        predicted = [t["topic_title"] for t in result.get("final_missed_topics", [])]
        scores = compute_gap_metrics(predicted, case["expected_missed_topics"])
        print(f"\n=== Case: {case['case_id']} ===")
        print(f"Predicted Missed : {predicted}")
        print(f"Expected Missed  : {case['expected_missed_topics']}")
        print(f"Metrics          : {scores} | Status: {result.get('final_status')}")


if __name__ == "__main__":
    run_eval_suite()
