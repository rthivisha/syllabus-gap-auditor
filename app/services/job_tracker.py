import threading
from typing import Dict


ACTIVE_JOB_STAGES: Dict[str, Dict[str, str]] = {}
_ACTIVE_JOB_STAGES_LOCK = threading.Lock()


def set_job_stage(session_id: str, stage: str, status: str = "running") -> None:
    if not session_id:
        return
    with _ACTIVE_JOB_STAGES_LOCK:
        ACTIVE_JOB_STAGES[session_id] = {"status": status, "stage": stage}


def get_job_state(session_id: str) -> Dict[str, str]:
    with _ACTIVE_JOB_STAGES_LOCK:
        return dict(
            ACTIVE_JOB_STAGES.get(session_id, {"status": "idle", "stage": ""})
        )
