import math

import pytest
from rank_bm25 import BM25Okapi

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, Settings
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval.bm25 import BM25
from rag_lab.retrieval.stats import idf
from rag_lab.retrieval.tokenize import tokenize

SETTINGS = Settings.from_env({})

# A tiny corpus small enough to check by hand.
TINY = [["moon", "landing", "moon"], ["sun", "and", "moon"], ["vitamin", "c"]]


def test_hand_computed_score():
    # query "landing": N=3, n=1 -> idf = ln(1 + 2.5/1.5) = 0.98083
    # chunk 0: tf=1, length 3, average length 8/3 -> length factor 1 - 0.75 + 0.75*3/(8/3) = 1.09375
    # contribution = 0.98083 * 1 * 2.5 / (1 + 1.5 * 1.09375) = 0.92860
    assert math.isclose(BM25(TINY).score(["landing"], 0), 0.92860, abs_tol=1e-4)


def test_score_matches_the_formula_written_out_independently():
    bm = BM25(TINY, k1=1.2, b=0.5)
    avg = 8 / 3
    expected = idf(3, 2) * 2 * 2.2 / (2 + 1.2 * (1 - 0.5 + 0.5 * 3 / avg))      # "moon", chunk 0
    assert math.isclose(bm.score(["moon"], 0), expected)


def test_chunk_without_the_word_scores_zero():
    assert BM25(TINY).score(["landing"], 2) == 0


def test_word_missing_from_the_whole_corpus_scores_zero_everywhere():
    assert BM25(TINY).scores(["zebra"]) == [0, 0, 0]


def test_empty_query_scores_zero():
    assert BM25(TINY).scores([]) == [0, 0, 0]


def test_scores_come_back_in_chunk_order_and_best_chunk_wins():
    scores = BM25(TINY).scores(["landing"])
    assert len(scores) == 3
    assert scores.index(max(scores)) == 0


def test_repeating_a_query_word_adds_its_contribution_again():
    bm = BM25(TINY)
    assert math.isclose(bm.score(["moon", "moon"], 0), 2 * bm.score(["moon"], 0))


def test_rare_word_beats_common_word():
    bm = BM25(TINY)
    assert bm.score(["landing"], 0) > bm.score(["moon"], 0) / 2             # 1 rare vs half of 2 common


# ------------------------------------------------ k1: saturation
def padded(n_moon: int, total: int = 20) -> list[str]:
    return ["moon"] * n_moon + ["x"] * (total - n_moon)


def test_each_extra_repeat_adds_less_than_the_one_before():
    bm = BM25([padded(n) for n in range(0, 11)] + [["y"] * 20])             # equal-length chunks
    s = [bm.score(["moon"], n) for n in range(1, 11)]
    gains = [b - a for a, b in zip(s, s[1:])]
    assert all(g > 0 for g in gains)
    assert gains == sorted(gains, reverse=True)


def test_score_can_never_pass_idf_times_k1_plus_one():
    bm = BM25([padded(n) for n in range(0, 11)] + [["y"] * 20], k1=1.5)
    ceiling = bm.idf_of("moon") * 2.5
    assert all(bm.score(["moon"], n) < ceiling for n in range(0, 11))


def test_k1_zero_ignores_how_many_times_the_word_repeats():
    bm = BM25([padded(1), padded(9), ["y"] * 20], k1=0)
    assert math.isclose(bm.score(["moon"], 0), bm.score(["moon"], 1))


# ------------------------------------------------ b: length fairness
def test_longer_chunk_scores_lower_for_the_same_count():
    bm = BM25([["moon"] + ["x"] * 4, ["moon"] + ["x"] * 40, ["y"] * 10])
    assert bm.score(["moon"], 0) > bm.score(["moon"], 1)


def test_b_zero_ignores_chunk_length():
    bm = BM25([["moon"] + ["x"] * 4, ["moon"] + ["x"] * 40, ["y"] * 10], b=0)
    assert math.isclose(bm.score(["moon"], 0), bm.score(["moon"], 1))


def test_all_empty_chunks_do_not_divide_by_zero():
    assert BM25([[], []]).scores(["moon"]) == [0, 0]


# ------------------------------------------------ validation
def test_empty_corpus_rejected():
    with pytest.raises(ValueError, match="at least one chunk"):
        BM25([])


@pytest.mark.parametrize("k1,b", [(-0.1, 0.5), (1.5, -0.1), (1.5, 1.1)])
def test_bad_parameters_rejected(k1, b):
    with pytest.raises(ValueError):
        BM25(TINY, k1=k1, b=b)


@pytest.mark.parametrize("index", [-1, 3])
def test_chunk_index_out_of_range_rejected(index):
    with pytest.raises(IndexError):
        BM25(TINY).score(["moon"], index)


# ------------------------------------------------ real corpus
@pytest.fixture(scope="module")
def real():
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(SETTINGS.sample_docs_dir))]
    chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, DEFAULT_CHUNK_SIZE), DEFAULT_CHUNK_OVERLAP)]
    tokens = [tokenize(c.text) for c in chunks]
    return chunks, tokens, BM25(tokens)


def test_agrees_with_rank_bm25_on_the_saturation_and_length_parts(real):
    """rank_bm25 uses a slightly different idf, so compare score / idf, which isolates
    the tf-saturation and length-normalisation parts that we wrote ourselves."""
    _, tokens, ours = real
    lib = BM25Okapi(tokens, k1=1.5, b=0.75)
    checked = 0
    for word, lib_idf in lib.idf.items():
        if lib_idf <= 0:                          # rank_bm25 replaces negative idf; skip those
            continue
        our_scores = ours.scores([word])
        lib_scores = lib.get_scores([word])
        for ours_i, lib_i in zip(our_scores, lib_scores):
            assert math.isclose(ours_i / ours.idf_of(word), lib_i / lib_idf, rel_tol=1e-9, abs_tol=1e-12)
        checked += 1
    assert checked > 500


def test_keyword_queries_find_the_right_document(real):
    chunks, _, bm = real
    cases = {
        "suprachiasmatic nucleus melatonin": "health_sleep_science",
        "three-way handshake SYN ACK": "computing_tcp_ip_networking",
        "Perseverance Jezero Crater river delta": "space_mars_exploration",
        "overfitting regularisation": "computing_machine_learning_basics",
    }
    for query, doc_id in cases.items():
        scores = bm.scores(tokenize(query))
        assert chunks[scores.index(max(scores))].doc_id == doc_id, query
