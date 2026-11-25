"""Recursive character chunking that keeps the page metadata on every piece."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from documind.config import Settings
from documind.documents import Chunk, Page


def chunk_pages(pages: list[Page], settings: Settings) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page in pages:
        pieces = splitter.split_text(page.text)
        for index, piece in enumerate(pieces):
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{page.doc}-p{page.page}-c{index}",
                    text=text,
                    doc=page.doc,
                    filename=page.filename,
                    file_type=page.file_type,
                    page=page.page,
                    heading=page.heading,
                    chunk_index=index,
                )
            )
    return chunks
