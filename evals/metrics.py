import re
from typing import Dict, List, Set


def _normalize_tokens(items: List[str]) -> Set[str]:
    tokens: Set[str] = set()
    for item in items:
        for tok in re.findall(r"[a-z0-9]+", item.lower()):
            if len(tok) > 2:
                tokens.add(tok)
    return tokens


def compute_gap_metrics(
    predicted_topics: List[str], expected_topics: List[str]
) -> Dict[str, float]:
    """
    Zero-LLM-cost deterministic metrics comparing predicted missed topics
    against golden ground-truth omissions.
    """
    pred_set = {p.strip().lower() for p in predicted_topics if p.strip()}
    exp_set = {e.strip().lower() for e in expected_topics if e.strip()}

    # Token-level Jaccard overlap (resilient to minor phrasing differences)
    pred_toks = _normalize_tokens(predicted_topics)
    exp_toks = _normalize_tokens(expected_topics)
    jaccard = len(pred_toks & exp_toks) / max(1, len(pred_toks | exp_toks))

    # Fuzzy topic-level Precision & Recall
    matched = sum(
        1 for exp in exp_set if any(exp in pred or pred in exp for pred in pred_set)
    )
    precision = (
        matched / max(1, len(pred_set)) if pred_set else (1.0 if not exp_set else 0.0)
    )
    recall = matched / max(1, len(exp_set)) if exp_set else 1.0
    false_omission_rate = 1.0 - recall

    return {
        "jaccard_overlap": round(jaccard, 4),
        "gap_precision": round(precision, 4),
        "gap_recall": round(recall, 4),
        "false_omission_rate": round(false_omission_rate, 4),
    }
