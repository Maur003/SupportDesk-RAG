import math

import pytest

from supportdesk.eval import enforce_floors
from supportdesk.eval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def _report(recall: float, leak: float = 0.0, absent_hit: float = 0.0) -> dict:
    return {
        "tables": [
            {"group": "answerable", "vector": {"recall_at_5": recall}},
            {"group": "permission_trap", "vector": leak},
            {"group": "unanswerable", "vector": absent_hit},
        ]
    }


def test_floor_accepts_a_recall_above_the_guard() -> None:
    enforce_floors(_report(0.6310))


def test_floor_rejects_a_collapsed_recall_or_a_leak() -> None:
    with pytest.raises(SystemExit):
        enforce_floors(_report(0.49))
    with pytest.raises(SystemExit):
        enforce_floors(_report(0.6310, leak=1.0))
    with pytest.raises(SystemExit):
        enforce_floors(_report(0.6310, absent_hit=1.0))


def test_precision_divides_by_k_when_the_list_is_short() -> None:
    assert precision_at_k(["a"], {"a"}, 5) == 0.2


def test_recall_mrr_and_ndcg_use_the_labeled_chunk() -> None:
    retrieved = ["miss", "hit", "other"]
    relevant = {"hit"}
    assert recall_at_k(retrieved, relevant, 5) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, 5) == 1.0 / math.log2(3)
    assert ndcg_at_k(["hit"], relevant, 5) == 1.0
    assert reciprocal_rank(["miss"], relevant) == 0.0
