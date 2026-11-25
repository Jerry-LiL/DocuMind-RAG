"""Small text helpers shared by ingestion, retrieval, and citation checks."""

from __future__ import annotations

import re
from pathlib import Path

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "for",
        "to",
        "and",
        "or",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "be",
        "with",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "as",
        "if",
        "not",
        "we",
        "you",
        "your",
    }
)

CITATION_RE = re.compile(r"\[([A-Za-z0-9_-]+):(\d+)\]")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_token_list(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1]


def content_tokens(text: str) -> set[str]:
    return set(content_token_list(text))


def citation_label(path: Path) -> str:
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")
    return label or "doc"


def citation_pairs(text: str) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []
    for doc, page in CITATION_RE.findall(text):
        pair = (doc, int(page))
        if pair not in seen:
            seen.append(pair)
    return seen


def strip_citations(text: str) -> str:
    return CITATION_RE.sub(" ", text)


def estimate_tokens(text: str) -> int:
    """Rough token count used when a provider does not report usage."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def sentences(text: str) -> list[str]:
    found: list[str] = []
    for line in re.split(r"[\n\r]+", text.strip()):
        parts = re.split(r"(?<=[.!?])\s+", line.strip())
        found.extend(part.strip() for part in parts if part.strip())
    return found
