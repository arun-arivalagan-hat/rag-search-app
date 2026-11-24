"""Module 3 demo: compare the chunking strategies on the real, cleaned sample documents.

    python scripts/demo_chunking.py
"""

from __future__ import annotations

from statistics import mean

from rag_lab.chunking import add_overlap, chunk_fixed_size, chunk_recursive, chunk_sentences
from rag_lab.config import get_settings
from rag_lab.ingestion import clean_corpus, load_corpus

MAX_CHARS = 300
OVERLAP = 60

STRATEGIES = {
    "fixed": lambda d: chunk_fixed_size(d, MAX_CHARS),
    "sentence": lambda d: chunk_sentences(d, MAX_CHARS),
    "recursive": lambda d: chunk_recursive(d, MAX_CHARS),
    "recursive+overlap": lambda d: add_overlap(d, chunk_recursive(d, MAX_CHARS), OVERLAP),
}


def mid_word_cuts(document, chunks) -> int:
    """Chunk starts that fall in the middle of a word (a sign of a bad cut)."""
    text = document.text
    return sum(1 for c in chunks[1:] if 0 < c.start_char < len(text)
               and not text[c.start_char - 1].isspace() and not text[c.start_char].isspace())


def main() -> None:
    docs = {c.doc_id: c.retrieval_document for c in clean_corpus(load_corpus(get_settings().sample_docs_dir))}

    print(f"1) All 10 documents, max_chars={MAX_CHARS}, overlap={OVERLAP}")
    print(f"   {'strategy':18} {'chunks':>7} {'avg len':>8} {'longest':>8} {'mid-word cuts':>14}")
    for name, chunker in STRATEGIES.items():
        chunks_by_doc = {i: chunker(d) for i, d in docs.items()}
        all_chunks = [c for cs in chunks_by_doc.values() for c in cs]
        cuts = sum(mid_word_cuts(docs[i], cs) for i, cs in chunks_by_doc.items())
        print(f"   {name:18} {len(all_chunks):>7} {mean(len(c.text) for c in all_chunks):>8.0f} "
              f"{max(len(c.text) for c in all_chunks):>8} {cuts:>14}")

    doc = docs["space_apollo_program"]
    print("\n2) The same boundary in each strategy (space_apollo_program, chunks 1 and 2)")
    for name, chunker in STRATEGIES.items():
        a, b = chunker(doc)[1:3]
        print(f"   {name}")
        print(f"     chunk 1 ends   : ...{a.text[-45:]!r}")
        print(f"     chunk 2 starts : {b.text[:45]!r}...")

    print("\n3) Offsets still point at the source text (recursive+overlap, first 3 chunks)")
    for c in STRATEGIES["recursive+overlap"](doc)[:3]:
        print(f"   {c.chunk_id}  [{c.start_char}:{c.end_char}]  overlap={c.metadata['overlap_chars']}  "
              f"text matches source: {doc.text[c.start_char:c.end_char] == c.text}")


if __name__ == "__main__":
    main()
