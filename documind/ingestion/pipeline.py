"""Ingest a file or a directory into the chunk index and the analytics catalog."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from documind.config import Settings
from documind.ingestion.chunker import chunk_pages
from documind.ingestion.loaders import load_file
from documind.schemas import IngestedDocument, IngestResponse

SUPPORTED = {".pdf", ".md", ".markdown"}


def iter_corpus_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED
    )


def ingest_path(path: Path, store, settings: Settings) -> IngestedDocument:
    loaded = load_file(path)
    chunks = chunk_pages(loaded.pages, settings)
    record = {
        "doc": loaded.doc,
        "title": loaded.title,
        "filename": loaded.filename,
        "file_type": loaded.file_type,
        "page_count": loaded.page_count,
        "chunk_count": len(chunks),
        "ingested_at": datetime.now(UTC).isoformat(),
    }
    store.replace_document(record, chunks)
    return IngestedDocument(
        doc=loaded.doc,
        filename=loaded.filename,
        file_type=loaded.file_type,
        page_count=loaded.page_count,
        chunk_count=len(chunks),
    )


def ingest_directory(folder: Path, store, settings: Settings) -> IngestResponse:
    documents = [ingest_path(path, store, settings) for path in iter_corpus_files(folder)]
    return IngestResponse(documents=documents, chunk_count=store.chunk_count())
