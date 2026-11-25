from __future__ import annotations

from pathlib import Path

import pytest

from documind.config import Settings
from documind.retrieval.store import CorpusStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="local",
        chroma_dir=tmp_path / "chroma",
        index_path=tmp_path / "chunks.json",
        sqlite_path=tmp_path / "analytics.sqlite",
        corpus_dir=SAMPLE,
        upload_dir=tmp_path / "uploads",
        log_path=tmp_path / "usage.jsonl",
        auto_ingest=False,
    )


@pytest.fixture
def store(settings: Settings) -> CorpusStore:
    return CorpusStore(settings)
