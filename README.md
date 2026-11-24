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
| 2  Cleaning (offset map, Unicode, whitespace, PDF reflow, hyphenation, Markdown markup, pipeline) | done, awaiting review |
| 3  Chunking (fixed, sentence, recursive, overlap) | done, awaiting review |
| 4+ | pending approval |

Demos: `python scripts/demo_ingestion.py`, `python scripts/demo_cleaning.py`, `python scripts/demo_chunking.py`

See `data/sample_docs/` (10 documents, 3 topics) and `data/eval/qrels.json` (14 labelled queries).
