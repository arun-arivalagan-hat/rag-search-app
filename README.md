# RAG Search Lab

A learning-oriented build of a production-style Retrieval-Augmented Generation system.
Core algorithms (cosine similarity, BM25, RRF, MMR, metrics) are implemented by hand first,
then compared with the standard libraries.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add OPENROUTER_API_KEY to get written answers
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

## Project structure

```
Rag-Research-lab/
├── README.md
├── pyproject.toml          # dependencies and project settings
├── .env / .env.example     # API keys and settings (.env is not pushed to GitHub)
├── .gitignore
├── data/                   # sample_docs/ (the corpus), eval/ (labelled questions), indexes/
├── frontend/src/           # the search page: index.html, scss/, css/, js/app.js
├── scripts/                # demos and tuning runs
├── tests/                  # unit and integration tests
├── logs/                   # rag_lab.log (server errors and warnings, not pushed)
└── src/rag_lab/
    ├── config.py           # settings and shared constants
    ├── ingestion/          # load PDF / TXT / Markdown and clean the text
    ├── chunking/           # split text into small chunks
    ├── embeddings/         # chunks to vectors, cosine similarity, nearest-chunk search
    ├── retrieval/          # BM25 keyword search, hybrid search, RRF
    ├── prompts/            # build the prompt: system rules, numbered sources, escaping
    ├── llm/                # call the model through OpenRouter, parse citations
    ├── evaluation/         # metrics, runner, tuning
    ├── api/                # FastAPI backend (service, app, main)
    └── utils/              # logging setup
```

There is no `vectordb/` folder yet: vectors are searched in memory, which is fast enough for this corpus.

Demos: `python scripts/demo_ingestion.py`, `python scripts/demo_cleaning.py`, `python scripts/demo_chunking.py`, `python scripts/demo_embeddings.py`

See `data/sample_docs/` (10 documents, 3 topics) and `data/eval/qrels.json` (14 labelled queries).
