import math

import pytest

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, Settings
from rag_lab.evaluation.qrels import EvalQuery, load_qrels
from rag_lab.evaluation.runner import evaluate, summarise
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval import KeywordIndex

SETTINGS = Settings.from_env({})

QUERIES = [
    EvalQuery("q1", "keyword", "alpha", frozenset({"d1"})),
    EvalQuery("q2", "keyword", "beta", frozenset({"d2"})),
    EvalQuery("q3", "semantic", "gamma", frozenset({"d3", "d4"})),
]


def table_retriever(table):
    """A fake search that answers from a fixed table, ignoring ``depth``."""
    return lambda query, depth: table[query]


def test_perfect_retriever_scores_one():
    retriever = table_retriever({"alpha": ["d1"], "beta": ["d2"], "gamma": ["d3", "d4"]})
    results = evaluate(retriever, QUERIES, k=2)
    assert [r.rr for r in results] == [1, 1, 1]
    assert [r.hit for r in results] == [1, 1, 1]
    assert summarise(results)["all"]["recall"] == 1


def test_retriever_that_finds_nothing_scores_zero():
    results = evaluate(lambda q, d: [], QUERIES, k=3)
    summary = summarise(results)["all"]
    assert summary["hit"] == summary["precision"] == summary["recall"] == summary["mrr"] == 0


def test_per_query_scores_hand_computed():
    # alpha: right doc is 2nd; beta: not returned; gamma: finds d4 only
    retriever = table_retriever({"alpha": ["x", "d1"], "beta": ["x", "y"], "gamma": ["d4", "z"]})
    a, b, g = evaluate(retriever, QUERIES, k=2)
    assert (a.hit, a.precision, a.recall, a.rr) == (1, 0.5, 1, 0.5)
    assert (b.hit, b.precision, b.recall, b.rr) == (0, 0, 0, 0)
    assert (g.hit, g.precision, g.recall, g.rr) == (1, 0.5, 0.5, 1)


def test_summary_averages_over_all_and_per_kind():
    retriever = table_retriever({"alpha": ["x", "d1"], "beta": ["x", "y"], "gamma": ["d4", "z"]})
    summary = summarise(evaluate(retriever, QUERIES, k=2))
    assert set(summary) == {"all", "keyword", "semantic"}
    assert math.isclose(summary["all"]["mrr"], (0.5 + 0 + 1) / 3)
    assert math.isclose(summary["keyword"]["mrr"], (0.5 + 0) / 2)       # alpha, beta
    assert summary["semantic"]["mrr"] == 1
    assert summary["all"]["queries"] == 3 and summary["keyword"]["queries"] == 2


def test_repeated_chunks_of_one_document_collapse_to_one_document():
    # three chunks of "x" then d1: d1 is the 2nd DOCUMENT, so rr is 1/2, not 1/4
    retriever = table_retriever({"alpha": ["x", "x", "x", "d1"], "beta": [], "gamma": []})
    assert evaluate(retriever, QUERIES, k=2)[0].rr == 0.5


def test_ranked_docs_are_distinct_and_in_order():
    retriever = table_retriever({"alpha": ["b", "a", "b", "c"], "beta": [], "gamma": []})
    assert evaluate(retriever, QUERIES, k=2)[0].ranked_docs == ["b", "a", "c"]


def test_retriever_is_asked_for_depth_chunks():
    seen = []
    evaluate(lambda q, d: seen.append(d) or [], QUERIES, k=3, depth=7)
    assert seen == [7, 7, 7]


def test_bad_k_depth_and_empty_results_rejected():
    with pytest.raises(ValueError):
        evaluate(lambda q, d: [], QUERIES, k=0)
    with pytest.raises(ValueError):
        evaluate(lambda q, d: [], QUERIES, k=5, depth=2)
    with pytest.raises(ValueError):
        summarise([])


# ------------------------------------------------ real corpus, keyword search only (offline)
def test_keyword_search_finds_the_answer_for_every_keyword_query():
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(SETTINGS.sample_docs_dir))]
    chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, DEFAULT_CHUNK_SIZE), DEFAULT_CHUNK_OVERLAP)]
    index = KeywordIndex(chunks)
    queries = [q for q in load_qrels(SETTINGS.eval_dir / "qrels.json") if q.kind == "keyword"]

    def keyword(query, depth):
        return [h.chunk.doc_id for h in index.search(query, k=depth)]

    summary = summarise(evaluate(keyword, queries, k=3))["all"]
    assert len(queries) == 5
    assert summary["hit"] == 1.0
