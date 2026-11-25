"""Extractive answers for local mode. The text is copied from a retrieved passage."""

from __future__ import annotations

from documind.textutil import content_tokens, sentences


def extractive_answer(question: str, documents: list[dict]) -> str:
    if not documents:
        return "I could not find this in the knowledge base."
    wanted = content_tokens(question)
    ranked: list[tuple[int, set[str], str, dict]] = []
    for doc in documents:
        for sentence in sentences(doc["text"]):
            cleaned = sentence.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            matched = _matched(wanted, cleaned)
            if matched:
                ranked.append((len(matched), matched, cleaned, doc))
    if not ranked:
        return "I could not find this in the knowledge base."
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[str, dict]] = []
    covered: set[str] = set()
    seen: set[str] = set()
    for _score, matched, sentence, doc in ranked:
        if sentence in seen:
            continue
        # Keep a second sentence only when it covers a question term the first missed.
        if covered and not (matched - covered):
            continue
        seen.add(sentence)
        chosen.append((sentence, doc))
        covered |= matched
        if len(chosen) == 2:
            break
    return " ".join(f"{sentence} [{doc['doc']}:{doc['page']}]" for sentence, doc in chosen)


def _matched(question: set[str], sentence: str) -> set[str]:
    words = content_tokens(sentence)
    found: set[str] = set()
    for token in question:
        if token in words or _prefix_hit(token, words):
            found.add(token)
    return found


def _prefix_hit(token: str, words: set[str]) -> bool:
    if len(token) < 4:
        return False
    return any(
        len(other) >= 4 and (other.startswith(token) or token.startswith(other)) for other in words
    )


def format_sql_answer(rows: list[dict]) -> str:
    if not rows:
        return "The analytics query returned no rows."
    if len(rows) == 1 and len(rows[0]) == 1:
        key, value = next(iter(rows[0].items()))
        return f"The {key.replace('_', ' ')} is {value}."
    pieces = []
    for row in rows:
        bits = [f"{key.replace('_', ' ')} {value}" for key, value in row.items()]
        pieces.append(", ".join(bits))
    return "Analytics result: " + "; ".join(pieces) + "."


def clarification_text() -> str:
    return (
        "Which document or topic should I look up? "
        "For catalog numbers, you can ask how many documents are ingested."
    )
