"""Build the prompt that is sent to the model (no API call happens here).

    builder.py  system rules, numbered and escaped sources, size budget
"""

from rag_lab.prompts.builder import SYSTEM_PROMPT, Prompt, SourceRef, build_prompt

__all__ = ["SYSTEM_PROMPT", "Prompt", "SourceRef", "build_prompt"]
