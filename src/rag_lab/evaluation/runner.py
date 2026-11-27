"""Step 7b: run labelled queries through a search and grade the results.

A "retriever" here is any function ``retriever(query_text, depth) -> list of doc ids`` that
returns the document id of each result chunk, best first (so a document id can repeat).
Keyword, semantic and hybrid search can all be wrapped in that shape, which is what makes
them comparable on exactly the same questions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag_lab.evaluation.metrics import (
    hit_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unique_in_order,
)
from rag_lab.evaluation.qrels import EvalQuery

Retriever = Callable[[str, int], list[str]]
METRICS = ("hit", "precision", "recall", "mrr")


@dataclass(frozen=True)
class QueryResult:
    """How one query went: what came back, and its scores."""

    query_id: str
    kind: str
    ranked_docs: list[str]            # distinct documents, best first
    hit: float
    precision: float
    recall: float
    rr: float                         # reciprocal rank of the first relevant document


def evaluate(retriever: Retriever, queries: Sequence[EvalQuery], k: int = 3, depth: int = 20) -> list[QueryResult]:
    """Run every query and score the first ``k`` DOCUMENTS of each result list.

    ``depth`` is how many chunks to ask the retriever for. It should comfortably exceed
    ``k``, because several chunks can belong to one document and shrink the document list.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if depth < k:
        raise ValueError("depth must be at least k")
    results = []
    for q in queries:
        ranked = unique_in_order(retriever(q.query, depth))
        results.append(
            QueryResult(
                query_id=q.query_id,
                kind=q.kind,
                ranked_docs=ranked,
                hit=hit_at_k(ranked, q.relevant_doc_ids, k),
                precision=precision_at_k(ranked, q.relevant_doc_ids, k),
                recall=recall_at_k(ranked, q.relevant_doc_ids, k),
                rr=reciprocal_rank(ranked, q.relevant_doc_ids),
            )
        )
    return results


def summarise(results: Sequence[QueryResult]) -> dict[str, dict[str, float]]:
    """Average each metric over all queries (group ``"all"``) and over each query kind.

    The ``"mrr"`` entry is the mean of the reciprocal ranks (Mean Reciprocal Rank).
    """
    if len(results) == 0:
        raise ValueError("no results to summarise")
    groups: dict[str, list[QueryResult]] = {"all": list(results)}
    for r in results:
        groups.setdefault(r.kind, []).append(r)
    return {
        name: {
            "hit": mean([r.hit for r in rows]),
            "precision": mean([r.precision for r in rows]),
            "recall": mean([r.recall for r in rows]),
            "mrr": mean([r.rr for r in rows]),
            "queries": float(len(rows)),
        }
        for name, rows in groups.items()
    }
