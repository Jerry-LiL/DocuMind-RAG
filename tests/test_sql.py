import sqlite3

import pytest

from documind.ingestion.pipeline import ingest_directory
from documind.sql.tool import SqlRejected, connect_readonly, execute_readonly, match_template
from tests.conftest import SAMPLE


def test_templates_do_not_steal_content_questions():
    assert "COUNT" in match_template("How many documents are in the corpus?")
    assert match_template("How many days are chat logs kept?") is None
    assert "SUM" in match_template("How many pages are in the corpus?")


def test_write_statements_are_rejected(settings, store):
    ingest_directory(SAMPLE, store, settings)
    with pytest.raises(SqlRejected):
        execute_readonly(settings.sqlite_path, "DELETE FROM documents")
    with pytest.raises(SqlRejected):
        execute_readonly(
            settings.sqlite_path,
            "SELECT COUNT(*) FROM documents; DROP TABLE documents",
        )
    with pytest.raises(SqlRejected):
        execute_readonly(settings.sqlite_path, "ATTACH DATABASE 'other.db' AS other")
    rows = execute_readonly(
        settings.sqlite_path,
        "SELECT COUNT(*) AS document_count FROM documents",
    )
    assert rows == [{"document_count": 3}]


def test_readonly_connection_cannot_create_tables(settings, store):
    ingest_directory(SAMPLE, store, settings)
    conn = connect_readonly(settings.sqlite_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE hack (id INTEGER)")
    finally:
        conn.close()
