"""Module 1 demo: load the real corpus and show exactly what came out.

    python scripts/demo_ingestion.py
"""

from __future__ import annotations

from rag_lab.config import get_settings
from rag_lab.ingestion import load_corpus, page_for_offset


def main() -> None:
    settings = get_settings()
    docs = load_corpus(settings.sample_docs_dir)

    print(f"Loaded {len(docs)} documents from {settings.sample_docs_dir}\n")
    print(f"{'doc_id':36} {'type':9} {'topic':10} {'year':5} {'chars':>6} {'words':>6}  sha256")
    print("-" * 100)
    for d in docs:
        m = d.metadata
        print(f"{d.doc_id:36} {m['file_type']:9} {m['topic']:10} {m['year']:<5} "
              f"{len(d.text):>6} {len(d.text.split()):>6}  {m['sha256'][:12]}")

    # ---- Markdown: structure lifted into metadata --------------------------
    md = next(d for d in docs if d.doc_id == "space_mars_exploration")
    print(f"\n=== Markdown metadata: {md.doc_id} ===")
    print("derived title :", md.metadata["title"])
    print("heading outline (level, text, char offset):")
    for level, title, off in md.metadata["headings"]:
        print(f"  {'  ' * (level - 1)}h{level} {title!r:32} @ {off}")
    print("markup preserved ->", repr(md.text[:60]))

    # ---- TXT ---------------------------------------------------------------
    txt = next(d for d in docs if d.doc_id == "computing_tcp_ip_networking")
    print(f"\n=== TXT: {txt.doc_id} ===")
    print(repr(txt.text[:160]) + " ...")

    # ---- PDF: page ranges and what extraction did to the layout ------------
    pdf = next(d for d in docs if d.doc_id == "space_black_holes")
    print(f"\n=== PDF: {pdf.doc_id} ===")
    print("page_count :", pdf.metadata["page_count"])
    print("page_ranges:", pdf.metadata["page_ranges"])
    probe = pdf.text.index("LIGO")
    print(f"offset of 'LIGO' = {probe}  ->  page {page_for_offset(pdf, probe)}")
    print("\nFirst 420 chars with newlines made visible (\\n):")
    print(repr(pdf.text[:420]))
    lines = pdf.text.split("\n")
    print(f"\nPage 1 of the source PDF is a title plus 3 paragraphs. Extraction returned "
          f"{len(lines)} lines for the whole document, hard-wrapped by the PDF layout.")
    print("Line lengths of the first 8 lines:", [len(l) for l in lines[:8]])


if __name__ == "__main__":
    main()
