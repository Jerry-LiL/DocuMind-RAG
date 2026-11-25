"""Prompts for the router, the SQL tool, and the answer draft."""

from __future__ import annotations

ROUTER_SYSTEM = """You route a question to one tool.
retrieve: the answer is inside a PDF or Markdown document.
sql: the question asks for catalog statistics in the documents table
(counts of documents, pages, or chunks, file types, or a list of ingested files).
A question about a fact inside a document is retrieve, even if it says "how many".
clarify: the question has no topic, or it refers to earlier context you do not have.
Return JSON with keys route and reason. route is retrieve, sql, or clarify."""

SQL_SYSTEM = """Write one SQLite SELECT for this catalog.
Table documents(
  doc TEXT,
  title TEXT,
  filename TEXT,
  file_type TEXT,
  page_count INTEGER,
  chunk_count INTEGER,
  ingested_at TEXT
)
Return JSON with a single key sql. One SELECT or WITH statement. No writes."""

RETRIEVE_SYSTEM = """Answer the question using only the passages.
End every factual sentence with a citation like [handbook:2], using the doc and page
printed on the passage. Do not cite a document that is not listed.
If the passages do not contain the answer, say you cannot find it in the knowledge base
and do not invent a citation.
Keep the answer under 80 words."""

SQL_ANSWER_SYSTEM = """Answer the catalog question in one or two sentences using only the rows.
Do not invent numbers. Do not add [Doc:Page] citations."""


def passages_block(documents: list[dict]) -> str:
    blocks = []
    for doc in documents:
        heading = f" ({doc['heading']})" if doc.get("heading") else ""
        blocks.append(f"[{doc['doc']}:{doc['page']}]{heading}\n{doc['text']}")
    return "\n\n".join(blocks)


def retrieve_user(question: str, documents: list[dict], draft: str, notes: str) -> str:
    parts = [f"Question: {question}", "", "Passages:", passages_block(documents) or "(none)"]
    if notes:
        parts.extend(
            [
                "",
                f"The previous answer was rejected: {notes}",
                f"Previous answer: {draft}",
            ]
        )
    return "\n".join(parts)


def sql_answer_user(question: str, sql: str, rows: list[dict], notes: str) -> str:
    parts = [
        f"Question: {question}",
        f"SQL: {sql}",
        "Rows:",
        json_rows(rows),
    ]
    if notes:
        parts.append(f"The previous answer was rejected: {notes}")
    return "\n".join(parts)


def json_rows(rows: list[dict]) -> str:
    import json

    return json.dumps(rows, ensure_ascii=False)
