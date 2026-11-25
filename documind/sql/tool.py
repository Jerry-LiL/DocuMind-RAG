"""SELECT-only access to the catalog. Writes never reach SQLite."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|replace|pragma|"
    r"vacuum|reindex|grant|truncate|begin|commit|rollback|load_extension)\b",
    re.IGNORECASE,
)

_TEMPLATES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(how many|number of|count of) documents\b", re.I),
        "SELECT COUNT(*) AS document_count FROM documents",
    ),
    (
        re.compile(r"\b(list|show) (all )?(the )?(documents|files)\b", re.I),
        "SELECT title, filename, file_type, page_count, chunk_count FROM documents ORDER BY title",
    ),
    (
        re.compile(r"\b(average|avg).*(chunks|pages)\b|\b(chunks|pages) per document\b", re.I),
        "SELECT ROUND(AVG(page_count), 2) AS avg_pages, "
        "ROUND(AVG(chunk_count), 2) AS avg_chunks FROM documents",
    ),
    (
        re.compile(r"\bfile types?\b|\bbreakdown by type\b", re.I),
        "SELECT file_type, COUNT(*) AS documents FROM documents "
        "GROUP BY file_type ORDER BY file_type",
    ),
    (
        re.compile(
            r"\b(total|how many) (pages|chunks)\b.*\b(corpus|index|database)\b",
            re.I,
        ),
        "SELECT COALESCE(SUM(page_count), 0) AS pages, "
        "COALESCE(SUM(chunk_count), 0) AS chunks FROM documents",
    ),
    (
        re.compile(r"\b(corpus|ingestion) (stats|statistics|summary|size)\b", re.I),
        "SELECT COUNT(*) AS document_count, "
        "COALESCE(SUM(page_count), 0) AS pages, "
        "COALESCE(SUM(chunk_count), 0) AS chunks FROM documents",
    ),
]


class SqlRejected(ValueError):
    """The statement was not a single read-only SELECT."""


def match_template(question: str) -> str | None:
    for pattern, statement in _TEMPLATES:
        if pattern.search(question):
            return statement
    return None


def validate_select(sql: str) -> str:
    cleaned = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    cleaned = re.sub(r"--[^\n]*", " ", cleaned).strip()
    cleaned = cleaned.rstrip(";").strip()
    if not cleaned:
        raise SqlRejected("empty statement")
    if ";" in cleaned:
        raise SqlRejected("only a single statement is allowed")
    if not re.match(r"(?is)(select|with)\b", cleaned):
        raise SqlRejected("only SELECT statements are allowed")
    if _FORBIDDEN.search(cleaned):
        raise SqlRejected("statement contains a forbidden keyword")
    return cleaned


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def execute_readonly(path: Path, sql: str, limit: int = 50) -> list[dict]:
    statement = validate_select(sql)
    conn = connect_readonly(path)
    try:
        cursor = conn.execute(statement)
        rows = cursor.fetchmany(limit)
        return [{key: _jsonable(row[key]) for key in row.keys()} for row in rows]
    finally:
        conn.close()


def _jsonable(value):
    if isinstance(value, float):
        return round(value, 4)
    return value
