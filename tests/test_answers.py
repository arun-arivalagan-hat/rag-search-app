import json

import pytest

from rag_lab.config import Settings
from rag_lab.evaluation.answers import (
    contains_answer,
    evaluate_answers,
    load_answers,
    normalise,
    summarise_answers,
)
from rag_lab.evaluation.qrels import EvalQuery, load_qrels
from rag_lab.ingestion import clean_corpus, load_corpus

SETTINGS = Settings.from_env({})
QUERIES = [EvalQuery("a", "keyword", "qa", frozenset({"d"})), EvalQuery("b", "keyword", "qb", frozenset({"d"}))]
ANSWERS = {"a": ["the answer"], "b": ["other answer", "second phrase"]}


# ------------------------------------------------ text matching
def test_normalise_ignores_case_and_line_breaks():
    assert normalise("The  Answer\n  IS\there") == "the answer is here"


def test_contains_answer_ignores_case_and_line_breaks():
    assert contains_answer("... THE\nanswer is ...", ["the answer"])


def test_contains_answer_any_phrase_is_enough():
    assert contains_answer("only the second phrase here", ["nope", "second phrase"])


def test_contains_answer_false_when_missing_or_split():
    assert not contains_answer("the ans", ["the answer"])      # a chunk boundary cut the phrase
    assert not contains_answer("", ["the answer"])


# ------------------------------------------------ evaluate_answers
def fake_search(table):
    return lambda query, depth: table[query][:depth]


def test_answer_in_first_chunk_scores_one():
    search = fake_search({"qa": ["the answer", "x"], "qb": ["second phrase"]})
    results = evaluate_answers(search, QUERIES, ANSWERS)
    assert [(r.hit, r.rr) for r in results] == [(1, 1), (1, 1)]


def test_answer_in_third_chunk_is_a_hit_at_3_but_rr_one_third():
    search = fake_search({"qa": ["x", "y", "the answer"], "qb": []})
    a, b = evaluate_answers(search, QUERIES, ANSWERS, k=3)
    assert (a.hit, a.rr) == (1, 1 / 3)
    assert (b.hit, b.rr) == (0, 0)


def test_answer_beyond_k_is_not_a_hit_but_still_gets_reciprocal_rank():
    search = fake_search({"qa": ["x", "y", "z", "the answer"], "qb": []})
    a, _ = evaluate_answers(search, QUERIES, ANSWERS, k=3)
    assert (a.hit, a.rr) == (0, 1 / 4)


def test_summary_averages():
    search = fake_search({"qa": ["the answer"], "qb": ["x"]})
    summary = summarise_answers(evaluate_answers(search, QUERIES, ANSWERS))
    assert summary == {"hit": 0.5, "mrr": 0.5}


def test_missing_phrases_for_a_query_rejected():
    with pytest.raises(KeyError):
        evaluate_answers(fake_search({"qa": []}), QUERIES, {"a": ["x"]})


def test_bad_k_and_depth_rejected():
    with pytest.raises(ValueError):
        evaluate_answers(fake_search({}), QUERIES, ANSWERS, k=5, depth=2)


def test_load_answers_rejects_empty_phrase_list(tmp_path):
    bad = tmp_path / "a.json"
    bad.write_text(json.dumps({"answers": {"q": []}}))
    with pytest.raises(ValueError, match="phrase"):
        load_answers(bad)


# ------------------------------------------------ the real answers.json
@pytest.fixture(scope="module")
def doc_text():
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(SETTINGS.sample_docs_dir))]
    return {d.doc_id: d.text for d in docs}


def test_every_query_has_answer_phrases():
    answers = load_answers(SETTINGS.eval_dir / "answers.json")
    queries = load_qrels(SETTINGS.eval_dir / "qrels.json") + load_qrels(SETTINGS.eval_dir / "qrels_hard.json")
    assert {q.query_id for q in queries} == set(answers)


def test_every_answer_phrase_really_appears_in_a_relevant_document(doc_text):
    """If a phrase is not in the labelled document, the label is wrong."""
    answers = load_answers(SETTINGS.eval_dir / "answers.json")
    queries = load_qrels(SETTINGS.eval_dir / "qrels.json") + load_qrels(SETTINGS.eval_dir / "qrels_hard.json")
    for q in queries:
        for phrase in answers[q.query_id]:
            assert any(contains_answer(doc_text[d], [phrase]) for d in q.relevant_doc_ids), (q.query_id, phrase)
