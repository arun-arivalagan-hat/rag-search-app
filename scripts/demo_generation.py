"""Module 8 demo: question -> hybrid search -> prompt -> (optionally) the model's answer.

    python scripts/demo_generation.py                       # dry run: no API call, no cost
    python scripts/demo_generation.py --live                # really calls the chosen model
    python scripts/demo_generation.py "your question" --live

The model comes from OPENROUTER_MODEL (and the key from OPENROUTER_API_KEY) in .env.
The dry run stops after building the prompt, so you can see exactly what would be sent.
"""

from __future__ import annotations

import sys

from rag_lab.chunking import add_overlap, chunk_recursive
from rag_lab.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, get_settings
from rag_lab.embeddings.embedder import Embedder
from rag_lab.llm import make_generator
from rag_lab.prompts import build_prompt
from rag_lab.ingestion import clean_corpus, load_corpus
from rag_lab.retrieval import HybridIndex

DEFAULT_QUESTION = "Why did Perseverance land in Jezero Crater?"
TOP_K = 3
MAX_TOKENS = 4096        # also pays for the model's thinking, so keep it generous


def main() -> None:
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    settings = get_settings()

    question = args[0] if args else DEFAULT_QUESTION
    docs = [c.retrieval_document for c in clean_corpus(load_corpus(settings.sample_docs_dir))]
    chunks = [c for d in docs for c in add_overlap(d, chunk_recursive(d, DEFAULT_CHUNK_SIZE), DEFAULT_CHUNK_OVERLAP)]
    index = HybridIndex(chunks, Embedder(settings.embedding_model))

    print(f"Question: {question}\n")
    hits = index.search(question, k=TOP_K)
    print(f"1) Hybrid search returned {len(hits)} chunks")
    for h in hits:
        print(f"   #{h.rank} {h.chunk.chunk_id:36} {h.chunk.text[:60]!r}")

    prompt = build_prompt(question, [h.chunk for h in hits])
    print(f"\n2) Prompt built: {len(prompt.system) + len(prompt.user)} characters, {len(prompt.sources)} sources")
    print(f"   answer model: openrouter:{settings.openrouter_model}")

    if "--live" not in flags:
        print("\n3) DRY RUN: nothing was sent. Add --live to call the model (needs the provider's API key).")
        return

    label, generate = make_generator(settings, max_tokens=MAX_TOKENS)
    print(f"\n3) Calling {label} ...")
    answer = generate(question, [h.chunk for h in hits])
    print(f"\nANSWER:\n{answer.text}\n")
    print("Cited sources:")
    for s in answer.cited:
        print(f"   [{s.number}] {s.chunk_id}")
    flags_out = {"unknown": answer.unknown, "refused": answer.refused, "truncated": answer.truncated,
                 "invalid citations": bool(answer.invalid_citations)}
    print(f"\nflags: {flags_out}")
    print(f"tokens: {answer.input_tokens} in, {answer.output_tokens} out   model: {answer.model}")


if __name__ == "__main__":
    main()
