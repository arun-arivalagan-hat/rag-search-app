"""Module 3: chunking strategies. Every chunker returns ``Chunk`` objects whose offsets
point into the text they were given (checked by ``verify_chunk_integrity``)."""

from rag_lab.chunking.base import build_chunks, make_chunk
from rag_lab.chunking.fixed import chunk_fixed_size, fixed_size_spans
from rag_lab.chunking.overlap import add_overlap
from rag_lab.chunking.recursive import chunk_recursive, recursive_spans
from rag_lab.chunking.sentences import chunk_sentences, sentence_spans

__all__ = [
    "add_overlap",
    "build_chunks",
    "chunk_fixed_size",
    "chunk_recursive",
    "chunk_sentences",
    "fixed_size_spans",
    "make_chunk",
    "recursive_spans",
    "sentence_spans",
]
