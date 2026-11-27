"""Module 7 demo: fixed vs sentence vs recursive chunking, at several sizes, on search quality.

    python scripts/compare_chunking.py

Hybrid search is run on all 33 labelled queries. Two scores per setting:

* doc MRR      : was the right DOCUMENT found, and how high? (1.000 = always first)
* answer hit@3 : does one of the top 3 CHUNKS actually contain the answer phrase?
* answer MRR   : how high is the first chunk that contains the answer?

The answer scores are the ones that can see a chunk being too small or cutting the answer in half.
"""

from __future__ import annotations

from rag_lab.chunking import add_overlap, chunk_fixed_size, chunk_recursive, chunk_sentences
from rag_lab.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, get_settings
from rag_lab.embeddings.embedder import Embedder
from rag_lab.embeddings.index import embed_chunks
from rag_lab.evaluation.answers import evaluate_answers, load_answers, summarise_answers
from rag_lab.evaluation.qrels import load_qrels
from rag_lab.evaluation.runner import evaluate, summarise
from rag_lab.evaluation.tuning import CachingEmbedder, grid
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval import HybridIndex

K = 3
DEPTH = 30
OVERLAP = DEFAULT_CHUNK_OVERLAP
SIZES = [200, 300, 500, 800]
STRATEGIES = {"fixed": chunk_fixed_size, "sentence": chunk_sentences, "recursive": chunk_recursive}
CURRENT = ("recursive", DEFAULT_CHUNK_SIZE)


def main() -> None:
    settings = get_settings()
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(settings.sample_docs_dir))]
    queries = load_qrels(settings.eval_dir / "qrels.json") + load_qrels(settings.eval_dir / "qrels_hard.json")
    answers = load_answers(settings.eval_dir / "answers.json")
    embedder = CachingEmbedder(Embedder(settings.embedding_model))

    print(f"{len(queries)} queries, hybrid search, overlap {OVERLAP}, scoring the first {K}\n")
    print(f"{'strategy':>9} {'size':>5} {'chunks':>6} {'avg len':>7} | {'doc MRR':>7} | {'answer hit@3':>12} {'answer MRR':>10}")

    for combo in grid(strategy=list(STRATEGIES), size=SIZES):
        name, size = combo["strategy"], combo["size"]
        chunker = STRATEGIES[name]
        chunks = [c for d in docs for c in add_overlap(d, chunker(d, size), OVERLAP)]
        index = HybridIndex(chunks, embedder, matrix=embed_chunks(embedder, chunks))

        def doc_ids(q: str, depth: int) -> list[str]:
            return [h.chunk.doc_id for h in index.search(q, k=depth)]

        def texts(q: str, depth: int) -> list[str]:
            return [h.chunk.text for h in index.search(q, k=depth)]

        doc = summarise(evaluate(doc_ids, queries, k=K, depth=DEPTH))["all"]
        ans = summarise_answers(evaluate_answers(texts, queries, answers, k=K, depth=DEPTH))
        avg = sum(len(c.text) for c in chunks) / len(chunks)
        mark = "  <- current" if (name, size) == CURRENT else ""
        print(f"{name:>9} {size:5d} {len(chunks):6d} {avg:7.0f} | {doc['mrr']:7.3f} | {ans['hit']:12.3f} {ans['mrr']:10.3f}{mark}")


if __name__ == "__main__":
    main()
