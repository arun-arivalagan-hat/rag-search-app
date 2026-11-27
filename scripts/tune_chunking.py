"""Module 7 demo: does chunk size or overlap change search quality?

    python scripts/tune_chunking.py

For each (chunk size, overlap) the documents are re-cut, the new chunks are re-embedded, and
keyword, semantic and hybrid search are scored on all 33 labelled queries (both sets).
"truncated" counts chunks longer than the embedding model can read (it silently ignores the
rest of the text, so those chunks are only partly searchable by meaning).
"""

from __future__ import annotations

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, get_settings
from rag_lab.embeddings.embedder import Embedder
from rag_lab.embeddings.index import embed_chunks, nearest
from rag_lab.evaluation.qrels import load_qrels
from rag_lab.evaluation.runner import evaluate, summarise
from rag_lab.evaluation.tuning import CachingEmbedder, grid
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval import HybridIndex, KeywordIndex

K = 3
DEPTH = 30                      # small chunks mean many chunks per document, so look deeper
SIZES = [200, 300, 500, 800, 1200]
OVERLAPS = [0, 50, 100]
DEFAULT = (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)


def main() -> None:
    settings = get_settings()
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(settings.sample_docs_dir))]
    easy = load_qrels(settings.eval_dir / "qrels.json")
    hard = load_qrels(settings.eval_dir / "qrels_hard.json")
    both = easy + hard

    embedder = Embedder(settings.embedding_model)
    cached = CachingEmbedder(embedder)
    model = embedder.model
    limit = model.max_seq_length

    print(f"{len(both)} queries ({len(easy)} easy + {len(hard)} hard), scoring the first {K} documents")
    print(f"embedding model reads at most {limit} tokens per chunk\n")
    print(f"{'size':>5} {'ovlp':>4} {'chunks':>6} {'trunc':>5} | {'keyword':>8} {'semantic':>8} {'hybrid':>7} | "
          f"{'hybrid recall':>13} | hybrid MRR easy / hard")

    rows = []
    for combo in grid(size=SIZES, overlap=OVERLAPS):
        size, overlap = combo["size"], combo["overlap"]
        chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, size), overlap)]
        truncated = sum(len(model.tokenizer(c.text)["input_ids"]) > limit for c in chunks)
        matrix = embed_chunks(cached, chunks)
        keyword_index = KeywordIndex(chunks)
        hybrid_index = HybridIndex(chunks, cached, matrix=matrix)

        def keyword(q: str, depth: int) -> list[str]:
            return [h.chunk.doc_id for h in keyword_index.search(q, k=depth)]

        def semantic(q: str, depth: int) -> list[str]:
            return [chunks[i].doc_id for i, _ in nearest(cached.embed_text(q), matrix, k=depth)]

        def hybrid(q: str, depth: int) -> list[str]:
            return [h.chunk.doc_id for h in hybrid_index.search(q, k=depth)]

        s = {name: summarise(evaluate(fn, both, k=K, depth=DEPTH))["all"]
             for name, fn in (("keyword", keyword), ("semantic", semantic), ("hybrid", hybrid))}
        h_easy = summarise(evaluate(hybrid, easy, k=K, depth=DEPTH))["all"]["mrr"]
        h_hard = summarise(evaluate(hybrid, hard, k=K, depth=DEPTH))["all"]["mrr"]
        mark = "  <- current" if (size, overlap) == DEFAULT else ""
        print(f"{size:5d} {overlap:4d} {len(chunks):6d} {truncated:5d} | {s['keyword']['mrr']:8.3f} "
              f"{s['semantic']['mrr']:8.3f} {s['hybrid']['mrr']:7.3f} | {s['hybrid']['recall']:13.3f} | "
              f"{h_easy:.3f} / {h_hard:.3f}{mark}")
        rows.append((size, overlap, s["hybrid"]["mrr"]))

    best = max(r[2] for r in rows)
    tied = [r for r in rows if abs(r[2] - best) < 1e-9]
    print(f"\nbest hybrid MRR over all queries: {best:.3f}  ({len(tied)} of {len(rows)} settings tie)")
    print("columns keyword / semantic / hybrid are MRR (1.000 = right document always ranked first)")


if __name__ == "__main__":
    main()
