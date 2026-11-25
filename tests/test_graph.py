from documind.graph.builder import build_graph, run_question
from documind.ingestion.pipeline import ingest_directory
from documind.llm.providers import LLMResult, build_model
from documind.textutil import citation_pairs
from tests.conftest import SAMPLE


def _graph(settings, store):
    ingest_directory(SAMPLE, store, settings)
    return build_graph(settings, store, build_model(settings))


def test_document_question_is_cited_and_grounded(settings, store):
    state = run_question(_graph(settings, store), "How long are chat logs kept?")
    assert state["route"] == "retrieve"
    assert "90 days" in state["answer"]
    assert citation_pairs(state["answer"])
    assert state["grounded"] is True
    assert state["low_confidence"] is False
    assert state["prompt_tokens"] > 0
    assert state["completion_tokens"] > 0
    assert state["token_source"] == "estimate"


def test_content_count_stays_on_the_retriever(settings, store):
    state = run_question(_graph(settings, store), "How many days are chat logs kept?")
    assert state["route"] == "retrieve"
    assert "90 days" in state["answer"]


def test_keyword_lookup_cites_the_pdf(settings, store):
    state = run_question(_graph(settings, store), "What is the on-call escalation code?")
    assert "BLUE-17" in state["answer"]
    assert ("policy", 1) in citation_pairs(state["answer"])
    assert state["grounded"] is True


def test_unknown_fact_is_flagged(settings, store):
    state = run_question(_graph(settings, store), "What is the office wifi password?")
    assert state["route"] == "retrieve"
    assert state["grounded"] is False
    assert state["low_confidence"] is True
    assert state["citations"] == []


def test_vague_question_asks_for_clarification(settings, store):
    state = run_question(_graph(settings, store), "what about that")
    assert state["route"] == "clarify"
    assert state["clarification"]
    assert state["low_confidence"] is True
    assert state["citations"] == []


def test_catalog_question_uses_read_only_sql(settings, store):
    state = run_question(_graph(settings, store), "How many documents are ingested?")
    assert state["route"] == "sql"
    assert "SELECT" in state["sql"].upper()
    assert state["sql_rows"] == [{"document_count": 3}]
    assert "3" in state["answer"]
    assert state["grounded"] is True
    assert state["low_confidence"] is False


class _Scripted:
    """Stand-in chat model with a fixed list of completions and streams."""

    provider = "openai"
    model_name = "scripted"

    def __init__(self, completes: list[str], streams: list[str]):
        self._completes = list(completes)
        self._streams = list(streams)

    def complete(self, system: str, user: str) -> LLMResult:
        return LLMResult(self._completes.pop(0), 11, 3, "provider")

    def complete_json(self, system: str, user: str, schema):
        result = self.complete(system, user)
        from documind.llm.providers import extract_json

        return schema.model_validate(extract_json(result.text)), result

    def stream(self, system: str, user: str, usage):
        usage.prompt_tokens = 15
        usage.completion_tokens = 6
        usage.token_source = "provider"
        yield self._streams.pop(0)


def test_ungrounded_draft_is_rewritten_once(settings, store):
    settings.retrieval_k = 20
    ingest_directory(SAMPLE, store, settings)
    page = next(chunk.page for chunk in store.chunks if "90 days" in chunk.text)
    model = _Scripted(
        completes=['{"route": "retrieve", "reason": "the retention policy is in the handbook"}'],
        streams=[
            "The wifi password is swordfish.",
            f"Chat logs are kept for 90 days. [handbook:{page}]",
        ],
    )
    state = run_question(build_graph(settings, store, model), "How long are chat logs kept?")
    assert state["rewrite_count"] == 1
    assert "90 days" in state["answer"]
    assert state["grounded"] is True
    assert state["token_source"] == "provider"
    assert model._streams == []


def test_generated_sql_is_executed_read_only(settings, store):
    ingest_directory(SAMPLE, store, settings)
    model = _Scripted(
        completes=[
            '{"route": "sql", "reason": "catalog size"}',
            '{"sql": "SELECT COUNT(*) AS document_count FROM documents"}',
        ],
        streams=["The document count is 3."],
    )
    state = run_question(build_graph(settings, store, model), "What is the size of the catalog?")
    assert state["route"] == "sql"
    assert state["sql_rows"] == [{"document_count": 3}]
    assert state["grounded"] is True
    assert state["token_source"] == "provider"
