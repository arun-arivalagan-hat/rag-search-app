"""Call the language model and turn its reply into an ``Answer``.

    openrouter.py  send the prompt through OpenRouter, retry temporary errors
    answer.py      the Answer type, [n] citation parsing, unknown / refused / truncated flags
    factory.py     make_generator: the one generate(question, chunks) function the app uses
"""

from rag_lab.llm.answer import Answer, build_answer, extract_citations, no_sources_answer
from rag_lab.llm.factory import make_generator
from rag_lab.llm.openrouter import OpenRouterError, generate_answer_openrouter

__all__ = [
    "Answer",
    "OpenRouterError",
    "build_answer",
    "extract_citations",
    "generate_answer_openrouter",
    "make_generator",
    "no_sources_answer",
]
