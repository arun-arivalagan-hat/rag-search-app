"""Loader for the hand-labelled relevance judgements (``data/eval/qrels.json``).

Metrics (Precision@K, Recall@K, MRR) arrive as soon as there is a first
retriever to measure (Module 6). The *labels* exist from day one so that every
later retriever is judged on the same fixed questions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_KINDS = frozenset({"keyword", "semantic", "multi"})


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    kind: str                    # keyword | semantic | multi
    query: str
    relevant_doc_ids: frozenset[str]


def load_qrels(path: Path) -> list[EvalQuery]:
    """Parse and validate a qrels file. Raises ``ValueError`` on malformed rows."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries: list[EvalQuery] = []
    seen: set[str] = set()
    for row in raw["queries"]:
        qid = row["id"]
        if qid in seen:
            raise ValueError(f"duplicate query id {qid!r}")
        seen.add(qid)
        if row["kind"] not in VALID_KINDS:
            raise ValueError(f"{qid}: kind must be one of {sorted(VALID_KINDS)}")
        if not row["relevant"]:
            raise ValueError(f"{qid}: needs at least one relevant doc")
        queries.append(
            EvalQuery(qid, row["kind"], row["query"], frozenset(row["relevant"]))
        )
    return queries
