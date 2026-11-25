"""In-memory shapes for a loaded file and the chunks we index."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Page:
    doc: str
    filename: str
    file_type: str
    page: int
    heading: str
    text: str


@dataclass
class LoadedDocument:
    doc: str
    title: str
    filename: str
    file_type: str
    page_count: int
    pages: list[Page]


@dataclass
class Chunk:
    id: str
    text: str
    doc: str
    filename: str
    file_type: str
    page: int
    heading: str
    chunk_index: int

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> Chunk:
        return cls(
            id=payload["id"],
            text=payload["text"],
            doc=payload["doc"],
            filename=payload["filename"],
            file_type=payload["file_type"],
            page=int(payload["page"]),
            heading=payload.get("heading") or "",
            chunk_index=int(payload["chunk_index"]),
        )

    def metadata(self) -> dict:
        return {
            "doc": self.doc,
            "filename": self.filename,
            "file_type": self.file_type,
            "page": self.page,
            "heading": self.heading[:200],
            "chunk_index": self.chunk_index,
        }
