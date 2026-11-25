"""Chunk corpus, Chroma vectors, and the BM25 index kept in step."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from documind.config import Settings
from documind.documents import Chunk
from documind.retrieval.bm25_index import BM25Index
from documind.retrieval.embeddings import Embedder
from documind.retrieval.hybrid import reciprocal_rank_fusion
from documind.sql.analytics import AnalyticsDB


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class CorpusStore:
    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        self._lock = threading.Lock()
        self.chunks: list[Chunk] = []
        self.documents: list[dict] = []
        self._by_id: dict[str, Chunk] = {}
        self._bm25 = BM25Index([])
        self.analytics = AnalyticsDB(settings.sqlite_path)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._open_collection()
        self._load()

    def chunk_count(self) -> int:
        return len(self.chunks)

    def document_count(self) -> int:
        return len(self.documents)

    def replace_document(self, record: dict, chunks: list[Chunk]) -> None:
        with self._lock:
            self.documents = [doc for doc in self.documents if doc["doc"] != record["doc"]]
            self.documents.append(record)
            self.documents.sort(key=lambda doc: doc["doc"])
            self.chunks = [chunk for chunk in self.chunks if chunk.doc != record["doc"]]
            self.chunks.extend(chunks)
            self._reindex_memory()
            self._persist()
            self._replace_vectors(record["doc"], chunks)
            self.analytics.rebuild(self.documents)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        with self._lock:
            if not self.chunks or k <= 0:
                return []
            dense_ids = self._dense_ids(query, k)
            lexical_ids = [chunk.id for chunk, _score in self._bm25.search(query, k=max(k * 3, k))]
            fused = reciprocal_rank_fusion([dense_ids, lexical_ids])
            hits: list[ScoredChunk] = []
            for chunk_id, score in fused[:k]:
                chunk = self._by_id.get(chunk_id)
                if chunk is not None:
                    hits.append(ScoredChunk(chunk=chunk, score=round(score, 6)))
            return hits

    def _dense_ids(self, query: str, k: int) -> list[str]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(max(k * 3, k), count),
        )
        return list(result["ids"][0]) if result["ids"] else []

    def _open_collection(self):
        name = f"documind-{self.settings.llm_provider}"
        dim = self.embedder.dimension
        existing = {collection.name for collection in self._client.list_collections()}
        if name in existing:
            collection = self._client.get_collection(name, embedding_function=None)
            stored = int((collection.metadata or {}).get("dim") or 0)
            if stored and dim and stored != dim:
                self._client.delete_collection(name)
            else:
                return collection
        return self._client.create_collection(
            name,
            metadata={"dim": dim},
            embedding_function=None,
        )

    def _load(self) -> None:
        path = self.settings.index_path
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.documents = list(payload.get("documents") or [])
            self.chunks = [Chunk.from_dict(item) for item in payload.get("chunks") or []]
        self._reindex_memory()
        self._sync_vectors()
        self.analytics.rebuild(self.documents)

    def _reindex_memory(self) -> None:
        self._by_id = {chunk.id: chunk for chunk in self.chunks}
        self._bm25 = BM25Index(self.chunks)

    def _persist(self) -> None:
        path = self.settings.index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self.documents,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _sync_vectors(self) -> None:
        stored = self._collection.count()
        if stored == len(self.chunks) and stored > 0:
            return
        ids = self._collection.get().get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        if self.chunks:
            self._add_vectors(self.chunks)

    def _replace_vectors(self, doc: str, chunks: list[Chunk]) -> None:
        found = self._collection.get(where={"doc": doc})
        old_ids = list(found.get("ids") or [])
        if old_ids:
            self._collection.delete(ids=old_ids)
        if chunks:
            self._add_vectors(chunks)

    def _add_vectors(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        dim = len(vectors[0]) if vectors else 0
        if dim:
            self.embedder.dimension = dim
            stored = int((self._collection.metadata or {}).get("dim") or 0)
            if stored not in (0, dim):
                name = self._collection.name
                self._client.delete_collection(name)
                self._collection = self._client.create_collection(
                    name,
                    metadata={"dim": dim},
                    embedding_function=None,
                )
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=vectors,
        )
