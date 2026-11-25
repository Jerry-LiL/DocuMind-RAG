from pathlib import Path

from documind.config import Settings
from documind.documents import Page
from documind.ingestion.chunker import chunk_pages
from documind.ingestion.loaders import load_file
from documind.textutil import citation_label
from tests.conftest import SAMPLE


def test_markdown_pages_follow_headings():
    loaded = load_file(SAMPLE / "handbook.md")
    assert loaded.doc == "handbook"
    assert loaded.file_type == "md"
    headings = [page.heading for page in loaded.pages]
    assert "Retention" in headings
    retention = next(page for page in loaded.pages if page.heading == "Retention")
    assert "90 days" in retention.text
    assert retention.page == headings.index("Retention") + 1


def test_pdf_keeps_real_page_numbers():
    loaded = load_file(SAMPLE / "policy.pdf")
    assert loaded.file_type == "pdf"
    assert loaded.page_count == 2
    assert loaded.pages[0].page == 1
    assert "BLUE-17" in loaded.pages[0].text
    assert "communications lead" in loaded.pages[1].text


def test_recursive_chunks_overlap_and_keep_metadata(tmp_path: Path):
    settings = Settings(
        chunk_size=80,
        chunk_overlap=20,
        chroma_dir=tmp_path / "chroma",
        index_path=tmp_path / "chunks.json",
        sqlite_path=tmp_path / "analytics.sqlite",
        log_path=tmp_path / "usage.jsonl",
    )
    words = " ".join(f"token{i}" for i in range(80))
    page = Page(
        doc="notes",
        filename="notes.md",
        file_type="md",
        page=2,
        heading="Notes",
        text=words,
    )
    chunks = chunk_pages([page], settings)
    assert len(chunks) > 1
    assert all(
        chunk.page == 2 and chunk.doc == "notes" and chunk.heading == "Notes" for chunk in chunks
    )
    assert chunks[0].id == "notes-p2-c0"
    shared = set(chunks[0].text.split()) & set(chunks[1].text.split())
    assert shared


def test_citation_label_strips_punctuation():
    assert citation_label(Path("Q3 Policy (final).md")) == "Q3_Policy_final"
