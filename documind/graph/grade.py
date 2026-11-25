"""Grounding checks that do not trust the model to grade itself."""

from __future__ import annotations

import json
import re

from documind.textutil import content_tokens, strip_citations

# A cited answer has to share a real slice of the passage, not just a stopword.
MIN_OVERLAP = 0.35
_NUMBER = re.compile(r"\d+")


def grade_retrieval(
    answer: str,
    citations: list[tuple[str, int]],
    documents: list[dict],
) -> tuple[bool, float, str]:
    if not documents:
        return False, 0.0, "nothing was retrieved"
    if not citations:
        return False, 0.15, "no inline citations"
    retrieved = {(doc["doc"], int(doc["page"])) for doc in documents}
    valid = [pair for pair in citations if pair in retrieved]
    precision = len(valid) / len(citations)
    cited = [doc for doc in documents if (doc["doc"], int(doc["page"])) in set(valid)]
    source_tokens = content_tokens(" ".join(doc["text"] for doc in cited))
    answer_tokens = content_tokens(strip_citations(answer))
    overlap = (len(answer_tokens & source_tokens) / len(answer_tokens)) if answer_tokens else 0.0
    confidence = round(min(1.0, 0.6 * precision + 0.4 * overlap), 4)
    if precision < 1:
        return False, confidence, "a citation is not in the retrieved passages"
    if overlap < MIN_OVERLAP:
        return False, confidence, "the answer does not overlap enough with the cited passages"
    return True, confidence, "citations match the retrieved passages"


def grade_sql(answer: str, rows: list[dict], error: str) -> tuple[bool, float, str]:
    if error:
        return False, 0.1, error
    if not rows:
        return True, 0.35, "the query returned no rows"
    answer_numbers = set(_NUMBER.findall(answer))
    row_numbers = set(_NUMBER.findall(json.dumps(rows)))
    if answer_numbers and not answer_numbers <= row_numbers:
        return False, 0.2, "the answer contains numbers that are not in the query result"
    return True, 0.85, "the answer stays inside the query result"


def should_rewrite(
    route: str,
    grounded: bool,
    rewrite_count: int,
    has_documents: bool,
    sql_error: str,
) -> bool:
    if grounded or rewrite_count >= 1 or sql_error:
        return False
    if route == "retrieve":
        return has_documents
    return route == "sql"
