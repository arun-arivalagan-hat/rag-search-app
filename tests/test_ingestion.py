import hashlib
import json

import pytest

from rag_lab.config import Settings
from rag_lab.ingestion import (
    IngestionError,
    load_corpus,
    load_document,
    load_manifest,
    page_for_offset,
)
from rag_lab.ingestion.loaders import decode_text, extract_headings

DOCS_DIR = Settings.from_env({}).sample_docs_dir


# ------------------------------------------------------------------ decoding
def test_crlf_and_cr_normalised_to_lf():
    assert decode_text(b"a\r\nb\rc\n") == "a\nb\nc\n"


def test_bom_is_stripped():
    assert decode_text("\ufeffhello".encode("utf-8")) == "hello"


def test_invalid_utf8_raises_not_replaced():
    with pytest.raises(IngestionError, match="UTF-8"):
        decode_text(b"caf\xe9")  # latin-1 'é', invalid as UTF-8


def test_unicode_preserved():
    assert decode_text("naïve \u2013 café ✓".encode("utf-8")) == "naïve \u2013 café ✓"


# ------------------------------------------------------------------ txt
def test_txt_loader_metadata_and_hash(tmp_path):
    p = tmp_path / "note.txt"
    raw = "line one\r\nline two\r\n".encode()
    p.write_bytes(raw)
    doc = load_document(p)
    assert doc.doc_id == "note"
    assert doc.text == "line one\nline two\n"
    assert doc.metadata["file_type"] == "txt"
    assert doc.metadata["sha256"] == hashlib.sha256(raw).hexdigest()  # hash of RAW bytes
    assert doc.metadata["char_count"] == len(doc.text)


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("  \n\n")
    with pytest.raises(IngestionError, match="empty"):
        load_document(p)


def test_unsupported_extension_rejected(tmp_path):
    p = tmp_path / "data.docx"
    p.write_bytes(b"x")
    with pytest.raises(IngestionError, match="unsupported"):
        load_document(p)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(IngestionError, match="cannot read"):
        load_document(tmp_path / "nope.txt")


# ------------------------------------------------------------------ markdown
def test_markdown_title_and_heading_offsets(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Title\n\nbody\n\n## Section\n\nmore\n")
    doc = load_document(p)
    assert doc.metadata["title"] == "Title"
    headings = doc.metadata["headings"]
    assert [(lvl, t) for lvl, t, _ in headings] == [(1, "Title"), (2, "Section")]
    for _, title, off in headings:  # offsets point at the heading line
        assert doc.text[off:].lstrip("#").lstrip().startswith(title)


def test_markdown_markup_is_preserved():
    pass_through = "# H\n\n**bold** and `code` and [link](http://x)\n"
    assert decode_text(pass_through.encode()) == pass_through


def test_markdown_headings_inside_code_fences_ignored():
    text = "# Real\n\n```bash\n# just a comment\n```\n\n## Also real\n"
    assert [h[1] for h in extract_headings(text)] == ["Real", "Also real"]


def test_markdown_closing_hashes_stripped():
    assert extract_headings("## Title ##\n")[0][1] == "Title"


def test_markdown_hash_without_space_is_not_heading():
    assert extract_headings("#hashtag\n") == ()


def test_front_matter_extracted_and_removed_from_text(tmp_path):
    p = tmp_path / "fm.md"
    p.write_text('---\nauthor: Ada\ntags: "a, b"\n---\n# Heading\n\nbody\n')
    doc = load_document(p)
    assert doc.metadata["fm_author"] == "Ada"
    assert doc.metadata["fm_tags"] == "a, b"
    assert doc.text.startswith("# Heading")  # front matter is not content
    # heading offsets are relative to the body that remains
    assert doc.metadata["headings"][0][2] == 0


def test_malformed_front_matter_rejected(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("---\nthis is not key value\n---\nbody\n")
    with pytest.raises(IngestionError, match="front-matter"):
        load_document(p)


def test_manifest_overrides_derived_metadata(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Derived Title\n\nbody\n")
    doc = load_document(p, extra_metadata={"title": "Curated Title", "topic": "t"})
    assert doc.metadata["title"] == "Curated Title"
    assert doc.metadata["topic"] == "t"


# ------------------------------------------------------------------ pdf
def test_corrupt_pdf_rejected(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(IngestionError, match="PDF"):
        load_document(p)


def test_pdf_without_text_layer_rejected(tmp_path):
    from reportlab.pdfgen import canvas  # dev dependency

    p = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(p))
    c.showPage()  # a page with nothing on it
    c.save()
    with pytest.raises(IngestionError, match="no extractable text"):
        load_document(p)


def test_sample_pdf_pages_and_ranges():
    doc = load_document(DOCS_DIR / "space_black_holes.pdf")
    assert doc.metadata["file_type"] == "pdf"
    assert doc.metadata["page_count"] == 2
    ranges = doc.metadata["page_ranges"]
    assert len(ranges) == 2
    # Each recorded range must reproduce exactly the text of that page.
    assert doc.text[ranges[0][0]:ranges[0][1]].startswith("Black Holes")
    assert "Event Horizon Telescope" in doc.text[ranges[1][0]:ranges[1][1]]
    assert "Event Horizon Telescope" not in doc.text[ranges[0][0]:ranges[0][1]]


def test_page_for_offset_maps_every_character():
    doc = load_document(DOCS_DIR / "space_black_holes.pdf")
    (s1, e1), (s2, e2) = doc.metadata["page_ranges"]
    assert page_for_offset(doc, 0) == 1
    assert page_for_offset(doc, e1 - 1) == 1
    assert page_for_offset(doc, e1) == 1          # separator gap -> previous page
    assert page_for_offset(doc, s2) == 2
    assert page_for_offset(doc, len(doc.text) - 1) == 2
    pages = {page_for_offset(doc, i) for i in range(len(doc.text))}
    assert pages == {1, 2}


def test_page_for_offset_validation():
    pdf = load_document(DOCS_DIR / "space_black_holes.pdf")
    with pytest.raises(ValueError, match="outside"):
        page_for_offset(pdf, len(pdf.text))
    txt = load_document(DOCS_DIR / "computing_tcp_ip_networking.txt")
    with pytest.raises(ValueError, match="not a paginated"):
        page_for_offset(txt, 0)


# ------------------------------------------------------------------ corpus
@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DOCS_DIR)


def test_corpus_loads_all_ten(corpus):
    assert len(corpus) == 10
    assert len({d.doc_id for d in corpus}) == 10


def test_corpus_doc_ids_match_qrels_vocabulary(corpus):
    qrels = json.loads((Settings.from_env({}).eval_dir / "qrels.json").read_text())
    wanted = {d for q in qrels["queries"] for d in q["relevant"]}
    assert wanted <= {d.doc_id for d in corpus}


def test_corpus_metadata_merged_from_manifest(corpus):
    by_id = {d.doc_id: d for d in corpus}
    bh = by_id["space_black_holes"]
    assert (bh.metadata["topic"], bh.metadata["year"]) == ("space", 2023)
    assert bh.metadata["file_type"] == "pdf"
    assert by_id["computing_database_indexing"].metadata["title"] == "Database Indexing"


def test_every_document_has_base_metadata(corpus):
    for d in corpus:
        assert {"source", "file_type", "sha256", "char_count", "topic", "year", "level"} <= d.metadata.keys()
        assert d.metadata["char_count"] == len(d.text)
        assert len(d.metadata["sha256"]) == 64


def test_loading_is_deterministic():
    a = load_document(DOCS_DIR / "space_black_holes.pdf")
    b = load_document(DOCS_DIR / "space_black_holes.pdf")
    assert (a.text, dict(a.metadata)) == (b.text, dict(b.metadata))


def test_no_carriage_returns_or_bom_anywhere(corpus):
    for d in corpus:
        assert "\r" not in d.text and "\ufeff" not in d.text


def test_markdown_titles_derived_from_h1(corpus):
    md = [d for d in corpus if d.metadata["file_type"] == "markdown"]
    assert len(md) == 6
    for d in md:
        assert d.metadata["title"]  # every sample .md starts with an H1
        assert len(d.metadata["headings"]) >= 2


def test_manifest_stale_entry_detected(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "manifest.json").write_text(json.dumps({"documents": [
        {"file": "a.txt", "topic": "x"}, {"file": "ghost.txt", "topic": "y"}]}))
    with pytest.raises(IngestionError, match="ghost.txt"):
        load_corpus(tmp_path)


def test_unlisted_file_still_loads_without_manifest_fields(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    docs = load_corpus(tmp_path)
    assert docs[0].metadata["source"] == "a.txt" and "topic" not in docs[0].metadata


def test_duplicate_doc_id_rejected(tmp_path):
    (tmp_path / "same.txt").write_text("one")
    (tmp_path / "same.md").write_text("two")
    with pytest.raises(IngestionError, match="duplicate doc_id"):
        load_corpus(tmp_path)


def test_manifest_duplicate_entry_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"documents": [
        {"file": "a.txt"}, {"file": "a.txt"}]}))
    with pytest.raises(IngestionError, match="twice"):
        load_manifest(tmp_path)
