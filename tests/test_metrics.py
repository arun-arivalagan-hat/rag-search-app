import math

import pytest

from rag_lab.evaluation.metrics import (
    hit_at_k,
    hits_in_top_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unique_in_order,
)

RANKED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d5", "d9"}          # d9 is relevant but never returned


# ------------------------------------------------ unique_in_order
def test_unique_in_order_keeps_first_occurrence():
    assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_unique_in_order_of_nothing():
    assert unique_in_order([]) == []


# ------------------------------------------------ hand-computed values
def test_hits_in_top_k():
    assert hits_in_top_k(RANKED, RELEVANT, 1) == 0
    assert hits_in_top_k(RANKED, RELEVANT, 2) == 1       # d2
    assert hits_in_top_k(RANKED, RELEVANT, 5) == 2       # d2, d5


def test_precision_at_k():
    assert precision_at_k(RANKED, RELEVANT, 2) == 1 / 2
    assert precision_at_k(RANKED, RELEVANT, 5) == 2 / 5


def test_recall_at_k():
    assert math.isclose(recall_at_k(RANKED, RELEVANT, 2), 1 / 3)
    assert math.isclose(recall_at_k(RANKED, RELEVANT, 5), 2 / 3)     # d9 was never found


def test_hit_at_k():
    assert hit_at_k(RANKED, RELEVANT, 1) == 0.0
    assert hit_at_k(RANKED, RELEVANT, 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == 1 / 2            # first relevant is 2nd
    assert reciprocal_rank(["d2"], RELEVANT) == 1.0
    assert reciprocal_rank(["x", "y", "d5"], RELEVANT) == 1 / 3


def test_nothing_relevant_found_scores_zero_everywhere():
    ranked = ["x", "y"]
    assert precision_at_k(ranked, RELEVANT, 2) == 0
    assert recall_at_k(ranked, RELEVANT, 2) == 0
    assert hit_at_k(ranked, RELEVANT, 2) == 0
    assert reciprocal_rank(ranked, RELEVANT) == 0


def test_perfect_ranking():
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1
    assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1
    assert reciprocal_rank(["a", "b"], {"a", "b"}) == 1


# ------------------------------------------------ behaviour worth pinning down
def test_precision_divides_by_k_even_when_fewer_results_come_back():
    assert precision_at_k(["a"], {"a"}, 5) == 1 / 5


def test_repeated_documents_count_once():
    # three chunks of d1 then d2: d1 is ONE document, so d2 is 2nd, not 4th
    ranked = ["d1", "d1", "d1", "d2"]
    assert reciprocal_rank(ranked, {"d2"}) == 1 / 2
    assert hits_in_top_k(["d2", "d2", "d2"], {"d2"}, 3) == 1


def test_k_larger_than_the_list_is_fine():
    assert recall_at_k(["a"], {"a"}, 100) == 1


def test_empty_ranking_scores_zero():
    assert recall_at_k([], {"a"}, 3) == 0
    assert reciprocal_rank([], {"a"}) == 0


def test_metrics_never_leave_zero_to_one():
    for k in range(1, 8):
        for fn in (precision_at_k, recall_at_k, hit_at_k):
            assert 0 <= fn(RANKED, RELEVANT, k) <= 1


def test_recall_never_decreases_as_k_grows():
    values = [recall_at_k(RANKED, RELEVANT, k) for k in range(1, 8)]
    assert values == sorted(values)


# ------------------------------------------------ mean
def test_mean():
    assert mean([1, 0.5, 0]) == 0.5


def test_mean_of_nothing_rejected():
    with pytest.raises(ValueError):
        mean([])


# ------------------------------------------------ validation
def test_empty_relevant_set_rejected():
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(RANKED, set(), 3)
    with pytest.raises(ValueError, match="relevant"):
        reciprocal_rank(RANKED, set())


@pytest.mark.parametrize("k", [0, -1])
def test_bad_k_rejected(k):
    with pytest.raises(ValueError, match="k must"):
        precision_at_k(RANKED, RELEVANT, k)
