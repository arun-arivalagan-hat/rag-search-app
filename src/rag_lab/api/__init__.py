"""The web backend.

    service.py  SearchService: vector, BM25 and hybrid search behind one function
    app.py      FastAPI routes (/api/health, /api/search, /api/answer) and validation
    main.py     entry point: uvicorn rag_lab.api.main:app --host 127.0.0.1
"""
