"""Graph state: everything that flows between the nodes is defined here.

A node returns only the keys it changes, not the whole state; LangGraph merges them.
Only `trace` has a reducer (operator.add), so every node can *append* its line instead of overwriting.
"""
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # ---- Input ----
    question: str                      # the user's original question
    chat_history: list[dict]           # [{"role": "user"|"assistant", "content": "..."}]

    # ---- Query analysis ----
    standalone_question: str           # the follow-up rewritten to be understandable without the history
    query_type: str                    # how_to | conceptual | reference | troubleshooting
    search_query: str                  # the query retrieval is currently using (changes on rewrite)

    # ---- Retrieval + grading ----
    documents: list[Document]          # what retrieve returned
    relevant_documents: list[Document]  # what is left after grading (or the web results)

    # ---- Retry control ----
    retry_count: int                   # how many times the query has been rewritten
    max_retries: int                   # the limit (from config), so the loop can never be infinite
    used_web_search: bool              # whether the web fallback ran

    # ---- Generation + hallucination check ----
    answer: str
    sources: list[dict[str, Any]]
    generation_attempts: int
    grounded: bool | None              # None = the check did not run
    grounding_feedback: str            # unsupported claims, used for regeneration

    # ---- Debug ----
    trace: Annotated[list[str], operator.add]   # one-line decision of every node