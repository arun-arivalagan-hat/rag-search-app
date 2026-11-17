"""The sample corpus and qrels are test fixtures for every later module, so
their consistency is itself tested."""

import json
from collections import Counter

import pytest

from rag_lab.config import Settings
from rag_lab.evaluation.qrels import load_qrels

SETTINGS = Settings.from_env({})
DOCS_DIR = SETTINGS.sample_docs_dir


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DOCS_DIR / "manifest.json").read_text(encoding="utf-8"))["documents"]


def test_ten_documents_three_topics(manifest):
    assert len(manifest) == 10
    assert {d["topic"] for d in manifest} == {"space", "health", "computing"}


def test_every_manifest_file_exists_and_every_file_is_listed(manifest):
    listed = {d["file"] for d in manifest}
    on_disk = {p.name for p in DOCS_DIR.iterdir() if p.suffix in {".txt", ".md", ".pdf"}}
    assert listed == on_disk


def test_all_three_formats_present_including_one_pdf(manifest):
    exts = Counter(d["file"].rsplit(".", 1)[1] for d in manifest)
    assert exts["pdf"] == 1
    assert exts["txt"] >= 1 and exts["md"] >= 1


def test_pdf_is_a_real_pdf():
    header = (DOCS_DIR / "space_black_holes.pdf").read_bytes()[:5]
    assert header == b"%PDF-"


def test_text_docs_are_non_trivial_utf8(manifest):
    for d in manifest:
        if d["file"].endswith(".pdf"):
            continue
        text = (DOCS_DIR / d["file"]).read_text(encoding="utf-8")
        assert len(text.split()) > 120, d["file"]


def test_manifest_metadata_complete(manifest):
    for d in manifest:
        assert {"file", "topic", "title", "year", "level"} <= d.keys()
        assert isinstance(d["year"], int)


def test_qrels_reference_existing_documents(manifest):
    doc_ids = {d["file"].rsplit(".", 1)[0] for d in manifest}
    queries = load_qrels(SETTINGS.eval_dir / "qrels.json")
    assert len(queries) >= 10
    for q in queries:
        assert q.relevant_doc_ids <= doc_ids, q.query_id


def test_qrels_cover_all_three_query_kinds():
    kinds = {q.kind for q in load_qrels(SETTINGS.eval_dir / "qrels.json")}
    assert kinds == {"keyword", "semantic", "multi"}


def test_load_qrels_rejects_bad_kind(tmp_path):
    bad = tmp_path / "q.json"
    bad.write_text(json.dumps({"queries": [
        {"id": "a", "kind": "nonsense", "query": "x", "relevant": ["d"]}]}))
    with pytest.raises(ValueError, match="kind"):
        load_qrels(bad)
