"""Module 2 demo: before/after of the full cleaning pipeline on the real sample documents.

    python scripts/demo_cleaning.py
"""

from __future__ import annotations

from rag_lab.config import get_settings
from rag_lab.ingestion import clean_corpus, load_corpus


def lines(text: str) -> int:
    return len(text.split("\n"))


def main() -> None:
    docs = load_corpus(get_settings().sample_docs_dir)
    cleaned = {c.doc_id: c for c in clean_corpus(docs)}

    print("1) All 10 documents: original -> cleaned")
    print(f"   {'doc_id':36} {'type':9} {'chars':>13} {'lines':>9}  steps")
    for c in cleaned.values():
        o = c.original
        print(f"   {c.doc_id:36} {o.metadata['file_type']:9} {len(o.text):>5} -> {len(c.text):<5} "
              f"{lines(o.text):>3} -> {lines(c.text):<3}  {'+'.join(dict.fromkeys(c.steps))}")

    print("\n2) PDF (space_black_holes): the whole first page, before and after")
    pdf = cleaned["space_black_holes"]
    first_page_end = pdf.original.metadata["page_ranges"][0][1]
    print("   BEFORE:")
    for ln in pdf.original.text[:first_page_end].split("\n"):
        print(f"     |{ln}")
    cut = pdf.text.index("In 2019")
    print("   AFTER:")
    for ln in pdf.text[:cut].rstrip().split("\n"):
        print(f"     |{ln}")
    print(f"   phrase 'in direct proportion'  before: {'in direct proportion' in pdf.original.text}"
          f"  after: {'in direct proportion' in pdf.text}")

    print("\n3) Markdown (space_mars_exploration): start of the file")
    mars = cleaned["space_mars_exploration"]
    print("   BEFORE:", repr(mars.original.text[:62]))
    print("   AFTER :", repr(mars.text[:60]))
    print("   headings in original:", [h[1] for h in mars.original.metadata["headings"]])
    i = mars.text.index("Rovers")
    print(f"   cleaned heading 'Rovers' at {i}: quote with its marker -> {mars.quote(i - 3, i + 6)!r}")

    print("\n4) Markdown with bold text and bullets (health_vitamins_nutrition)")
    vit = cleaned["health_vitamins_nutrition"]
    i0 = vit.original.text.index("- **Vitamin C**")
    j0 = vit.text.index("- Vitamin C")
    print("   BEFORE:", repr(vit.original.text[i0:i0 + 55]))
    print("   AFTER :", repr(vit.text[j0:j0 + 51]))

    print("\n5) Markdown with inline code (computing_database_indexing)")
    db = cleaned["computing_database_indexing"]
    k0 = db.original.text.index("`WHERE")
    k1 = db.text.index("WHERE")
    print("   BEFORE:", repr(db.original.text[k0:k0 + 34]))
    print("   AFTER :", repr(db.text[k1:k1 + 32]))

    print("\n6) Citations: cleaned span -> original wording -> PDF page")
    for needle in ["Event Horizon Telescope", "in direct proportion", "LIGO detectors"]:
        s = pdf.text.index(needle)
        e = s + len(needle)
        a, b = pdf.source_span(s, e)
        print(f"   cleaned ({s:4},{e:4}) {needle!r:28} -> original ({a:4},{b:4}) {pdf.quote(s, e)!r:28} page {pdf.page_for_span(s, e)}")

    print("\n7) retrieval_document (what Module 3 will chunk)")
    r = pdf.retrieval_document
    print("   text length:", len(r.text), "| original length:", r.metadata["original_char_count"])
    print("   steps      :", r.metadata["cleaning_steps"])
    print("   offset-bearing metadata dropped:", [k for k in ("page_ranges", "headings") if k not in r.metadata])


if __name__ == "__main__":
    main()
