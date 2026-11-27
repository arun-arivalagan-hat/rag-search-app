"""Module 7 tuning demo: try many hybrid-search settings and see which score best.

    python scripts/tune_hybrid.py

To avoid fooling ourselves, settings are chosen on the ORIGINAL 14 queries (the "tuning set")
and the winners are then re-scored on the 19 HARD queries (the "check set"), which the choice
never saw. A setting that only wins on the tuning set has probably just memorised it.
"""

from __future__ import annotations

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, get_settings
from rag_lab.embeddings.embedder import Embedder
from rag_lab.embeddings.index import embed_chunks
from rag_lab.evaluation.qrels import load_qrels
from rag_lab.evaluation.runner import evaluate, summarise
from rag_lab.evaluation.tuning import CachingEmbedder, grid, marginal_means
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval import HybridIndex

MAX_CHARS = DEFAULT_CHUNK_SIZE
OVERLAP = DEFAULT_CHUNK_OVERLAP
K = 3
DEPTH = 10
DEFAULTS = {"k1": 1.5, "b": 0.75, "rrf_k": 60, "candidates": 20}


def main() -> None:
    settings = get_settings()
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(settings.sample_docs_dir))]
    chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, MAX_CHARS), OVERLAP)]
    tuning_set = load_qrels(settings.eval_dir / "qrels.json")
    check_set = load_qrels(settings.eval_dir / "qrels_hard.json")

    embedder = CachingEmbedder(Embedder(settings.embedding_model))
    matrix = embed_chunks(embedder, chunks)                     # embedded ONCE, reused by every combination

    def score(combo: dict, queries) -> dict[str, float]:
        index = HybridIndex(chunks, embedder, k1=combo["k1"], b=combo["b"], rrf_k=combo["rrf_k"], matrix=matrix)

        def retriever(query: str, depth: int) -> list[str]:
            return [h.chunk.doc_id for h in index.search(query, k=depth, candidates=combo["candidates"])]

        return summarise(evaluate(retriever, queries, k=K, depth=DEPTH))["all"]

    combos = grid(
        k1=[0.5, 1.0, 1.5, 2.0],
        b=[0.0, 0.25, 0.5, 0.75, 1.0],
        rrf_k=[1, 10, 30, 60, 100],
        candidates=[10, 20, 43],
    )
    rows = []
    for combo in combos:
        t, c = score(combo, tuning_set), score(combo, check_set)
        rows.append({**combo, "t_mrr": t["mrr"], "t_recall": t["recall"], "c_mrr": c["mrr"], "c_recall": c["recall"]})

    print(f"{len(combos)} combinations, {len(chunks)} chunks; tuning set = {len(tuning_set)} queries, "
          f"check set = {len(check_set)} queries\n")

    default = next(r for r in rows if all(r[k] == v for k, v in DEFAULTS.items()))
    print("1) The current defaults")
    print(f"   {DEFAULTS}")
    print(f"   tuning set  MRR {default['t_mrr']:.3f}  Recall@{K} {default['t_recall']:.3f}")
    print(f"   check set   MRR {default['c_mrr']:.3f}  Recall@{K} {default['c_recall']:.3f}\n")

    best_t = max(r["t_mrr"] for r in rows)
    tied = [r for r in rows if r["t_mrr"] == best_t]
    print(f"2) Best tuning-set MRR is {best_t:.3f}; {len(tied)} of {len(rows)} combinations tie for it")
    check_scores = sorted({round(r["c_mrr"], 3) for r in tied})
    print(f"   those tied winners score {check_scores[0]:.3f} to {check_scores[-1]:.3f} on the check set\n")

    print("3) How much does the score move at all? (min .. max over all combinations)")
    for label, key in (("tuning set MRR", "t_mrr"), ("check set  MRR", "c_mrr")):
        values = [r[key] for r in rows]
        print(f"   {label}: {min(values):.3f} .. {max(values):.3f}")
    print()

    print("4) One setting at a time: average MRR for each value (tuning set | check set)")
    for setting in ("k1", "b", "rrf_k", "candidates"):
        t_means = marginal_means(rows, setting, "t_mrr")
        c_means = marginal_means(rows, setting, "c_mrr")
        print(f"   {setting}")
        for value in t_means:
            print(f"      {value!s:>5}   {t_means[value]:.3f} | {c_means[value]:.3f}")

    print("\n5) The 5 lowest-scoring combinations on the tuning set")
    for r in sorted(rows, key=lambda r: (r["t_mrr"], r["t_recall"]))[:5]:
        print(f"   k1={r['k1']} b={r['b']} rrf_k={r['rrf_k']} cand={r['candidates']}   "
              f"tuning {r['t_mrr']:.3f}  check {r['c_mrr']:.3f}")


if __name__ == "__main__":
    main()
