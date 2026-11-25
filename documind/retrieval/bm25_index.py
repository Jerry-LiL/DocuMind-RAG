"""Keyword index over the chunk corpus."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from documind.documents import Chunk
from documind.textutil import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(chunk.text) or ["empty"] for chunk in chunks]
        self._index = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._index is None or not self.chunks or k <= 0:
            return []
        query_tokens = tokenize(query) or ["empty"]
        scores = [float(score) for score in self._index.get_scores(query_tokens)]
        # Okapi IDF drops to 0 on a tiny corpus when a term sits in about half
        # the chunks. Token overlap still ranks that keyword hit.
        if not scores or max(scores) <= 0:
            wanted = set(query_tokens)
            scores = [float(len(wanted & set(tokenize(chunk.text)))) for chunk in self.chunks]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        hits: list[tuple[Chunk, float]] = []
        for index, score in ranked[:k]:
            if score <= 0:
                continue
            hits.append((self.chunks[index], float(score)))
        return hits
