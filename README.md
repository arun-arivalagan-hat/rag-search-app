# RAG Search Lab

A learning-oriented build of a production-style Retrieval-Augmented Generation system.
Core algorithms (cosine similarity, BM25, RRF, MMR, metrics) are implemented by hand first,
then compared with the standard libraries.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add ANTHROPIC_API_KEY when you reach Module 14
python scripts/generate_sample_pdf.py   # only if the PDF is missing
pytest
```

## Progress

| Module | Status |
|---|---|
| 0  Scaffold, config, types, sample corpus, qrels | done |
| 1  Ingestion (TXT / Markdown / PDF loaders, manifest metadata, PDF page ranges) | done, approved |
| 2  Cleaning (offset map, Unicode, whitespace, PDF reflow, hyphenation, Markdown markup, pipeline) | done, approved |
| 3  Chunking (fixed, sentence, recursive, overlap) | done, approved |
| 4  Embeddings (vector maths by hand, cosine similarity, all-MiniLM-L6-v2 embedder, nearest-chunk search) | done, approved |
| 5  Keyword search (tokenizer, TF / IDF, BM25 by hand, top-k search) | done, approved |
| 6+ | pending approval |

Demos: `python scripts/demo_ingestion.py`, `python scripts/demo_cleaning.py`, `python scripts/demo_chunking.py`, `python scripts/demo_embeddings.py`

See `data/sample_docs/` (10 documents, 3 topics) and `data/eval/qrels.json` (14 labelled queries).
