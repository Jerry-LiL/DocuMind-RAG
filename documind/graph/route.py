"""Local routing rules used on their own, and as a fallback if a model returns junk."""

from __future__ import annotations

import re

from documind.schemas import RouteDecision
from documind.sql.tool import match_template
from documind.textutil import content_tokens

_VAGUE = re.compile(
    r"^(hi|hello|hey|help|explain|more|continue|go on|tell me more|"
    r"what about (it|that|this)|and that|why\??|how\??|what\??|how many\??)$",
    re.IGNORECASE,
)


def heuristic_route(question: str) -> RouteDecision:
    cleaned = " ".join(question.strip().lower().split())
    if not content_tokens(cleaned) or _VAGUE.match(cleaned):
        return RouteDecision(
            route="clarify",
            reason="The question does not name a topic to look up.",
        )
    if match_template(question):
        return RouteDecision(
            route="sql",
            reason="The question asks for a catalog statistic.",
        )
    return RouteDecision(route="retrieve", reason="The question looks like document content.")
