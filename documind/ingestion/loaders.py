"""PDF pages and Markdown heading sections become citation pages."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from documind.documents import LoadedDocument, Page
from documind.textutil import citation_label

_HEADING = re.compile(r"^(#{1,2})\s+(.+?)\s*$", re.MULTILINE)


def load_file(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".md", ".markdown"}:
        return _load_markdown(path)
    raise ValueError(f"unsupported file type: {path.name}")


def _load_pdf(path: Path) -> LoadedDocument:
    reader = PdfReader(str(path))
    doc = citation_label(path)
    pages: list[Page] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if not text:
            continue
        pages.append(
            Page(
                doc=doc,
                filename=path.name,
                file_type="pdf",
                page=number,
                heading="",
                text=text,
            )
        )
    title = _first_line(pages[0].text) if pages else path.stem
    return LoadedDocument(
        doc=doc,
        title=title[:120],
        filename=path.name,
        file_type="pdf",
        page_count=len(reader.pages),
        pages=pages,
    )


def _load_markdown(path: Path) -> LoadedDocument:
    text = path.read_text(encoding="utf-8")
    doc = citation_label(path)
    matches = list(_HEADING.finditer(text))
    sections: list[tuple[str, str]] = []
    if not matches:
        if text.strip():
            sections.append(("", text.strip()))
    else:
        if matches[0].start() > 0 and text[: matches[0].start()].strip():
            sections.append(("", text[: matches[0].start()].strip()))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[match.start() : end].strip()
            if body:
                sections.append((match.group(2).strip(), body))

    pages = [
        Page(
            doc=doc,
            filename=path.name,
            file_type="md",
            page=number,
            heading=heading,
            text=body,
        )
        for number, (heading, body) in enumerate(sections, start=1)
    ]
    title = next((heading for heading, _ in sections if heading), path.stem)
    return LoadedDocument(
        doc=doc,
        title=title[:120],
        filename=path.name,
        file_type="md",
        page_count=len(pages),
        pages=pages,
    )


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text[:120]
