"""The harder labelled query set is a fixture too, so its consistency is tested."""

import json

import pytest

from rag_lab.config import Settings
from rag_lab.evaluation.qrels import load_qrels
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval.tokenize import tokenize

SETTINGS = Settings.from_env({})
HARD = SETTINGS.eval_dir / "qrels_hard.json"


@pytest.fixture(scope="module")
def doc_texts():
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(SETTINGS.sample_docs_dir))]
    return {d.doc_id: set(tokenize(d.text)) for d in docs}


def test_hard_set_loads_and_covers_all_three_kinds():
    queries = load_qrels(HARD)
    assert len(queries) >= 15
    assert {q.kind for q in queries} == {"keyword", "semantic", "multi"}


def test_hard_labels_point_at_existing_documents(doc_texts):
    for q in load_qrels(HARD):
        assert q.relevant_doc_ids <= doc_texts.keys(), q.query_id


def test_hard_ids_and_queries_do_not_repeat_the_original_set():
    original = load_qrels(SETTINGS.eval_dir / "qrels.json")
    hard = load_qrels(HARD)
    assert not {q.query_id for q in original} & {q.query_id for q in hard}
    assert not {q.query.casefold() for q in original} & {q.query.casefold() for q in hard}


def test_every_keyword_query_shares_a_word_with_its_labelled_document(doc_texts):
    """A labelling sanity check: a 'keyword' query must actually contain a word from its answer."""
    for q in load_qrels(HARD):
        if q.kind != "keyword":
            continue
        words = set(tokenize(q.query))
        assert any(words & doc_texts[d] for d in q.relevant_doc_ids), q.query_id


def test_hard_file_is_valid_json_with_a_note():
    raw = json.loads(HARD.read_text(encoding="utf-8"))
    assert "_note" in raw and raw["queries"]
