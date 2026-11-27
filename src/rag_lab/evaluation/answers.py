"""Step 7f: chunk-level check, "does the returned chunk actually contain the answer?"

The document-level metrics (step 7a) only ask whether the right DOCUMENT was found. A tiny
chunk can sit in the right document and still miss the answer sentence. Here every query has
a few answer phrases (copied from the document, ``data/eval/answers.json``) and a result
chunk counts as correct only if its text contains one of them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_lab.evaluation.metrics import mean
from rag_lab.evaluation.qrels import EvalQuery

ChunkSearch = Callable[[str, int], list[str]]       # (query, how many) -> chunk TEXTS, best first


def normalise(text: str) -> str:
    """Lower-case and collapse every run of whitespace to one space, so line breaks
    inside a chunk cannot hide a phrase."""
    return re.sub(r"\s+", " ", text).casefold().strip()


def contains_answer(chunk_text: str, phrases: Sequence[str]) -> bool:
    """True if ``chunk_text`` contains at least one answer phrase."""
    haystack = normalise(chunk_text)
    return any(normalise(p) in haystack for p in phrases)


def load_answers(path: Path) -> dict[str, list[str]]:
    """Read ``answers.json``: ``{query_id: [phrase, ...]}``. Every query needs a phrase."""
    raw = json.loads(path.read_text(encoding="utf-8"))["answers"]
    for query_id, phrases in raw.items():
        if not phrases or not all(isinstance(p, str) and p.strip() for p in phrases):
            raise ValueError(f"{query_id}: needs at least one non-empty answer phrase")
    return raw


@dataclass(frozen=True)
class AnswerResult:
    """Answer-level result for one question: did a top-k chunk contain the expected phrase?"""
    query_id: str
    hit: float          # 1.0 if one of the top ``k`` chunks contains the answer
    rr: float           # 1 / position of the first chunk that contains the answer (0 if none)


def evaluate_answers(
    search: ChunkSearch, queries: Sequence[EvalQuery], answers: Mapping[str, Sequence[str]], k: int = 3, depth: int = 10
) -> list[AnswerResult]:
    """For each query, look at the top ``depth`` chunks and check where the answer shows up.

    ``hit`` looks only at the first ``k`` chunks. Unlike the document metrics, nothing is
    collapsed: it is the CHUNKS themselves that must carry the answer.
    """
    if k < 1 or depth < k:
        raise ValueError("need 1 <= k <= depth")
    results = []
    for q in queries:
        if q.query_id not in answers:
            raise KeyError(f"no answer phrases for query {q.query_id}")
        texts = search(q.query, depth)
        positions = [i for i, t in enumerate(texts, start=1) if contains_answer(t, answers[q.query_id])]
        first = positions[0] if positions else None
        results.append(
            AnswerResult(
                query_id=q.query_id,
                hit=1.0 if first is not None and first <= k else 0.0,
                rr=1 / first if first is not None else 0.0,
            )
        )
    return results


def summarise_answers(results: Sequence[AnswerResult]) -> dict[str, float]:
    """Average ``hit`` and ``rr`` over the queries."""
    return {"hit": mean([r.hit for r in results]), "mrr": mean([r.rr for r in results])}
