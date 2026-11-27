"""Module 5 and hybrid search: find the best chunks for a question.

    tokenize.py, stats.py   words, term frequency, document frequency, IDF
    bm25.py                 BM25 scoring written by hand
    search.py               KeywordIndex: BM25 top-k over chunks
    fusion.py               reciprocal rank fusion: merge several ranked lists
    hybrid.py               HybridIndex: keyword + vector search merged with RRF
"""

from rag_lab.retrieval.bm25 import BM25
from rag_lab.retrieval.fusion import reciprocal_rank_fusion
from rag_lab.retrieval.hybrid import HybridHit, HybridIndex
from rag_lab.retrieval.search import KeywordHit, KeywordIndex, top_k
from rag_lab.retrieval.stats import document_frequencies, idf, idf_table, term_frequencies
from rag_lab.retrieval.tokenize import tokenize

__all__ = [
    "BM25",
    "HybridHit",
    "HybridIndex",
    "KeywordHit",
    "KeywordIndex",
    "document_frequencies",
    "idf",
    "idf_table",
    "reciprocal_rank_fusion",
    "term_frequencies",
    "tokenize",
    "top_k",
]
