"""RAG Search Lab: a learning-oriented RAG system with the core algorithms written by hand.

How one question travels through the packages (a map for anyone new to the code):

    ingestion  -> load and clean the documents          (runs once, at startup)
    chunking   -> cut the text into small chunks        (runs once)
    embeddings -> one meaning-vector per chunk          (runs once)
    retrieval  -> BM25 + vector + RRF find the best chunks for a question
    prompts    -> pack the chunks and the rules into a prompt
    llm        -> send it to OpenRouter, parse the cited answer
    api        -> FastAPI exposes search and answer to the frontend
    evaluation -> offline: measures every choice above with real numbers

``config.py`` holds the settings, ``types.py`` the shared ``Chunk`` and ``Document`` types.
"""

__version__ = "0.1.0"
