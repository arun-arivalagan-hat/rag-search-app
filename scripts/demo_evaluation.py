"""Module 7 demo: grade keyword, semantic and hybrid search on the 14 labelled queries.

    python scripts/demo_evaluation.py          # the original 14 queries
    python scripts/demo_evaluation.py hard     # the harder 19-query set
"""

from __future__ import annotations

import sys

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, get_settings
from rag_lab.embeddings.embedder import Embedder
from rag_lab.embeddings.index import embed_chunks, nearest
from rag_lab.evaluation.qrels import load_qrels
from rag_lab.evaluation.runner import evaluate, summarise
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval import HybridIndex, KeywordIndex

MAX_CHARS = DEFAULT_CHUNK_SIZE
OVERLAP = DEFAULT_CHUNK_OVERLAP
K = 3


def main() -> None:
    settings = get_settings()
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(settings.sample_docs_dir))]
    chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, MAX_CHARS), OVERLAP)]
    qrels_file = "qrels_hard.json" if "hard" in sys.argv[1:] else "qrels.json"
    queries = load_qrels(settings.eval_dir / qrels_file)
    print(f"query set: {qrels_file}")

    embedder = Embedder(settings.embedding_model)
    keyword_index = KeywordIndex(chunks)
    matrix = embed_chunks(embedder, chunks)
    hybrid_index = HybridIndex(chunks, embedder)

    def keyword(query: str, depth: int) -> list[str]:
        return [h.chunk.doc_id for h in keyword_index.search(query, k=depth)]

    def semantic(query: str, depth: int) -> list[str]:
        return [chunks[i].doc_id for i, _ in nearest(embedder.embed_text(query), matrix, k=depth)]

    def hybrid(query: str, depth: int) -> list[str]:
        return [h.chunk.doc_id for h in hybrid_index.search(query, k=depth)]

    retrievers = {"keyword": keyword, "semantic": semantic, "hybrid": hybrid}
    results = {name: evaluate(fn, queries, k=K) for name, fn in retrievers.items()}
    summaries = {name: summarise(r) for name, r in results.items()}

    print(f"{len(queries)} labelled queries, {len(chunks)} chunks, scoring the first {K} documents\n")
    for group in ("all", "keyword", "semantic", "multi"):
        n = int(summaries["keyword"][group]["queries"])
        print(f"{group} queries ({n})")
        print(f"   {'search':10} {'Hit@' + str(K):>7} {'Recall@' + str(K):>9} {'Prec@' + str(K):>8} {'MRR':>7}")
        for name in retrievers:
            s = summaries[name][group]
            print(f"   {name:10} {s['hit']:7.3f} {s['recall']:9.3f} {s['precision']:8.3f} {s['mrr']:7.3f}")
        print()

    print("Queries where the first relevant document was NOT ranked 1st")
    by_id = {q.query_id: q for q in queries}
    for name in retrievers:
        misses = [r for r in results[name] if r.rr < 1]
        print(f"   {name}: {len(misses)} of {len(queries)}")
        for r in misses:
            q = by_id[r.query_id]
            print(f"      {r.query_id} ({r.kind}) rr={r.rr:.2f}  {q.query[:55]!r}")
            print(f"         wanted {sorted(q.relevant_doc_ids)}  got {r.ranked_docs[:3]}")


if __name__ == "__main__":
    main()
