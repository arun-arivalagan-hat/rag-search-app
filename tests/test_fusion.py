import math

import pytest

from rag_lab.retrieval.fusion import reciprocal_rank_fusion


def test_hand_computed_scores():
    # k=60: a is 1st in one list; b is 2nd in one and 1st in the other; c is 2nd in one list.
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]]))
    assert math.isclose(fused["a"], 1 / 61)
    assert math.isclose(fused["b"], 1 / 62 + 1 / 61)
    assert math.isclose(fused["c"], 1 / 62)


def test_item_high_in_both_lists_beats_item_first_in_only_one():
    assert [i for i, _ in reciprocal_rank_fusion([["a", "b"], ["b", "c"]])] == ["b", "a", "c"]


def test_single_ranking_keeps_its_order():
    assert [i for i, _ in reciprocal_rank_fusion([["x", "y", "z"]])] == ["x", "y", "z"]


def test_item_missing_from_a_ranking_just_earns_nothing():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]]))
    assert math.isclose(fused["a"], 1 / 61) and math.isclose(fused["b"], 1 / 61)


def test_ties_keep_first_seen_order():
    assert [i for i, _ in reciprocal_rank_fusion([["a", "b"], ["b", "a"]])] == ["a", "b"]


def test_order_of_the_rankings_does_not_change_the_scores():
    one = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]]))
    two = dict(reciprocal_rank_fusion([["c", "a"], ["a", "b", "c"]]))
    assert one.keys() == two.keys()
    assert all(math.isclose(one[i], two[i]) for i in one)


def test_smaller_k_rewards_the_top_rank_more():
    ratio_k1 = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    ratio_k60 = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    assert ratio_k1["a"] / ratio_k1["b"] > ratio_k60["a"] / ratio_k60["b"]


def test_limit_keeps_only_the_top_results():
    assert [i for i, _ in reciprocal_rank_fusion([["a", "b", "c"]], limit=2)] == ["a", "b"]


def test_no_rankings_or_empty_rankings_give_empty_result():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_works_with_integer_ids_too():
    assert reciprocal_rank_fusion([[3, 1], [1, 3]])[0][0] == 3


def test_duplicate_id_inside_one_ranking_rejected():
    with pytest.raises(ValueError, match="twice"):
        reciprocal_rank_fusion([["a", "a"]])


def test_bad_k_and_limit_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], limit=0)
