"""SQLite catalog rebuilt from the chunk index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
)
"""


class AnalyticsDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")

    def rebuild(self, documents: list[dict]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents")
            conn.executemany(
                """
                INSERT INTO documents
                    (doc, title, filename, file_type, page_count, chunk_count, ingested_at)
                VALUES
                    (:doc, :title, :filename, :file_type, :page_count, :chunk_count, :ingested_at)
                """,
                documents,
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn
