"""Runtime settings. Environment variables override the defaults; see .env.example."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "local"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    chroma_dir: Path = Path("data/chroma")
    index_path: Path = Path("data/chunks.json")
    sqlite_path: Path = Path("data/analytics.sqlite")
    corpus_dir: Path = Path("sample_data")
    upload_dir: Path = Path("data/uploads")
    log_path: Path = Path("logs/usage.jsonl")

    low_confidence_threshold: float = 0.45
    retrieval_k: int = 4
    chunk_size: int = 700
    chunk_overlap: int = 100
    auto_ingest: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    hash_dim: int = Field(default=256, ge=32, le=2048)

    @field_validator("llm_provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"local", "openai", "ollama"}:
            raise ValueError("llm_provider must be local, openai, or ollama")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 700)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if value < 0:
            raise ValueError("chunk_overlap cannot be negative")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
