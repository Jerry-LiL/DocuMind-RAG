"""State that moves along the graph."""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    documents: list[dict]
    sql: str
    sql_rows: list[dict]
    sql_error: str
    draft: str
    answer: str
    citations: list[dict]
    grounded: bool
    confidence: float
    low_confidence: bool
    clarification: str | None
    grade_notes: str
    needs_rewrite: bool
    rewrite_count: int
    prompt_tokens: int
    completion_tokens: int
    token_source: str


def initial_state(question: str) -> GraphState:
    return {
        "question": question,
        "route": "",
        "route_reason": "",
        "documents": [],
        "sql": "",
        "sql_rows": [],
        "sql_error": "",
        "draft": "",
        "answer": "",
        "citations": [],
        "grounded": False,
        "confidence": 0.0,
        "low_confidence": True,
        "clarification": None,
        "grade_notes": "",
        "needs_rewrite": False,
        "rewrite_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "token_source": "",
    }
