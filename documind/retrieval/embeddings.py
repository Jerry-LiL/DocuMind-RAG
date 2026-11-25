"""Embedding functions for local hashing, OpenAI, and Ollama."""

from __future__ import annotations

import hashlib
import math

from documind.config import Settings
from documind.textutil import tokenize


def hash_embed(text: str, dim: int = 256) -> list[float]:
    """Feature-hash tokens into a unit vector.

    This is only for offline runs. It picks up repeated tokens, not paraphrases.
    """
    vec = [0.0] * dim
    tokens = tokenize(text) or ["empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign
    norm = math.sqrt(sum(value * value for value in vec)) or 1.0
    return [value / norm for value in vec]


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.llm_provider
        self.dimension = settings.hash_dim if settings.llm_provider == "local" else 0
        self._openai = None
        self._ollama = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "local":
            return [hash_embed(text, self.settings.hash_dim) for text in texts]
        if self.provider == "openai":
            vectors = self._openai_client().embed_documents(texts)
        else:
            vectors = self._ollama_client().embed_documents(texts)
        self._remember_dim(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if self.provider == "local":
            return hash_embed(text, self.settings.hash_dim)
        if self.provider == "openai":
            vector = self._openai_client().embed_query(text)
        else:
            vector = self._ollama_client().embed_query(text)
        self._remember_dim([vector])
        return vector

    def _remember_dim(self, vectors: list[list[float]]) -> None:
        if vectors and self.dimension == 0:
            self.dimension = len(vectors[0])

    def _openai_client(self):
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        if self._openai is None:
            from langchain_openai import OpenAIEmbeddings

            self._openai = OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                api_key=self.settings.openai_api_key,
            )
        return self._openai

    def _ollama_client(self):
        if self._ollama is None:
            from langchain_ollama import OllamaEmbeddings

            self._ollama = OllamaEmbeddings(
                model=self.settings.ollama_embedding_model,
                base_url=self.settings.ollama_base_url,
            )
        return self._ollama
