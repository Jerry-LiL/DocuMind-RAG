import math

from documind.documents import Chunk
from documind.ingestion.pipeline import ingest_directory, ingest_path
from documind.retrieval.bm25_index import BM25Index
from documind.retrieval.embeddings import hash_embed
from documind.retrieval.hybrid import reciprocal_rank_fusion
from documind.retrieval.store import CorpusStore
from tests.conftest import SAMPLE


def test_hash_embeddings_are_stable_unit_vectors():
    first = hash_embed("BLUE-17 escalation")
    second = hash_embed("BLUE-17 escalation")
    assert first == second
    assert abs(math.sqrt(sum(value * value for value in first)) - 1) < 1e-6
    assert hash_embed("chat logs") != hash_embed("BLUE-17 escalation")


def test_reciprocal_rank_fusion_prefers_agreement():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]]))
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def test_bm25_ranks_the_rare_token_first():
    chunks = [
        Chunk("a", "Chat logs are kept for 90 days.", "handbook", "handbook.md", "md", 3, "", 0),
        Chunk(
            "b",
            "The on-call escalation code is BLUE-17.",
            "policy",
            "policy.pdf",
            "pdf",
            1,
            "",
            0,
        ),
    ]
    hits = BM25Index(chunks).search("BLUE-17", k=2)
    assert hits[0][0].id == "b"


def test_hybrid_search_returns_the_keyword_chunk(settings, store: CorpusStore):
    ingest_directory(SAMPLE, store, settings)
    hits = store.search("BLUE-17", k=3)
    assert hits
    assert hits[0].chunk.doc == "policy"
    assert hits[0].chunk.page == 1
    assert "BLUE-17" in hits[0].chunk.text


def test_reingest_replaces_a_document(settings, store: CorpusStore):
    ingest_path(SAMPLE / "handbook.md", store, settings)
    first = store.chunk_count()
    ingest_path(SAMPLE / "handbook.md", store, settings)
    assert store.document_count() == 1
    assert store.chunk_count() == first
    reopened = CorpusStore(settings)
    assert reopened.chunk_count() == first
    assert reopened.document_count() == 1
