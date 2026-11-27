"""Module 4 demo: embed every chunk of the sample corpus and search it by meaning.

    python scripts/demo_embeddings.py
"""

from __future__ import annotations

import time

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, get_settings
from rag_lab.embeddings.embedder import Embedder
from rag_lab.embeddings.index import embed_chunks, nearest
from rag_lab.evaluation.qrels import load_qrels
from rag_lab.ingestion import clean_corpus, load_corpus

MAX_CHARS = DEFAULT_CHUNK_SIZE
OVERLAP = DEFAULT_CHUNK_OVERLAP
TOP_K = 3


def main() -> None:
    settings = get_settings()
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(settings.sample_docs_dir))]
    chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, MAX_CHARS), OVERLAP)]

    embedder = Embedder(settings.embedding_model)
    started = time.time()
    matrix = embed_chunks(embedder, chunks)
    print(f"1) Embedded {len(docs)} documents -> {len(chunks)} chunks -> matrix {matrix.shape} "
          f"({matrix.dtype}) in {time.time() - started:.1f}s")
    print(f"   model: {embedder.model_name}   dimension: {embedder.dimension}")

    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    print("\n2) Chunks per document")
    for doc_id, n in per_doc.items():
        print(f"   {doc_id:36} {n}")

    print(f"\n3) Top {TOP_K} chunks for each labelled query (OK = from a labelled relevant document)")
    queries = load_qrels(settings.eval_dir / "qrels.json")
    rank1_hits = 0
    for q in queries:
        results = nearest(embedder.embed_text(q.query), matrix, k=TOP_K)
        rank1_hits += chunks[results[0][0]].doc_id in q.relevant_doc_ids
        print(f"\n   {q.query_id} [{q.kind}] {q.query!r}")
        for rank, (i, score) in enumerate(results, 1):
            mark = "OK" if chunks[i].doc_id in q.relevant_doc_ids else "  "
            print(f"     {rank}. {score:.3f} {mark} {chunks[i].doc_id}  {chunks[i].text[:55]!r}")

    print(f"\n4) Rank-1 chunk came from a relevant document: {rank1_hits}/{len(queries)} queries")
    print("   (document-level quick check; real metrics such as Precision@K and MRR come in Module 6)")


if __name__ == "__main__":
    main()
