import json
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from documind.api.app import create_app
from documind.llm.providers import LLMResult
from tests.conftest import SAMPLE


def _events(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


def test_health_query_and_usage_log(settings):
    settings.auto_ingest = True
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["documents"] == 3
        assert health.json()["provider"] == "local"

        response = client.post("/query", json={"question": "How long are chat logs kept?"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["route"] == "retrieve"
        assert "90 days" in payload["answer"]
        assert payload["grounded"] is True
        assert payload["citations"][0]["doc"] == "handbook"
        assert payload["usage"]["latency_ms"] >= 0
        assert payload["usage"]["token_source"] == "estimate"
        assert payload["usage"]["prompt_tokens"] > 0

        catalog = client.post("/query", json={"question": "How many documents are ingested?"})
        assert catalog.json()["route"] == "sql"
        assert "3" in catalog.json()["answer"]

        vague = client.post("/query", json={"question": "what about that"})
        assert vague.json()["route"] == "clarify"
        assert vague.json()["low_confidence"] is True

    lines = settings.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert record["route"] == "retrieve"
    assert "latency_ms" in record
    assert "prompt_tokens" in record


def test_stream_emits_route_tokens_and_result(settings):
    settings.auto_ingest = True
    with TestClient(create_app(settings)) as client:
        with client.stream(
            "POST",
            "/query/stream",
            json={"question": "What is the on-call escalation code?"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
    events = _events(body)
    kinds = [event["type"] for event in events]
    assert kinds[0] == "route"
    assert "token" in kinds
    assert kinds[-1] == "result"
    result = events[-1]["answer"]
    assert "BLUE-17" in result["answer"]
    assert result["grounded"] is True
    assert any(item["doc"] == "policy" for item in result["citations"])


def test_upload_indexes_markdown(settings):
    with TestClient(create_app(settings)) as client:
        contents = b"# Notes\n\nThe lunch room code is OAK-4.\n"
        response = client.post(
            "/ingest",
            files={"file": ("notes.md", contents, "text/markdown")},
        )
        assert response.status_code == 200
        assert response.json()["documents"][0]["doc"] == "notes"
        answer = client.post("/query", json={"question": "What is the lunch room code?"})
        assert "OAK-4" in answer.json()["answer"]
        rejected = client.post(
            "/ingest",
            files={"file": ("notes.txt", b"nope", "text/plain")},
        )
        assert rejected.status_code == 400


def test_sample_ingest_endpoint(settings):
    settings.corpus_dir = SAMPLE
    with TestClient(create_app(settings)) as client:
        response = client.post("/ingest/sample")
        assert response.status_code == 200
        assert response.json()["chunk_count"] > 0
        assert {item["filename"] for item in response.json()["documents"]} == {
            "handbook.md",
            "release_notes.md",
            "policy.pdf",
        }


class _EchoChat(BaseChatModel):
    reply: str

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(
        self,
        messages,
        stop=None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def _stream(
        self,
        messages,
        stop=None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ):
        head, tail = self.reply.split("]", 1)
        # Keep the citation marker intact so the streamed pieces are easy to spot.
        parts = [head + "]", tail]
        for part in parts:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=part))
            if run_manager:
                run_manager.on_llm_new_token(part, chunk=chunk)
            yield chunk


class _StreamingModel:
    provider = "openai"
    model_name = "echo"

    def __init__(self, reply: str):
        self.reply = reply

    def complete(self, system: str, user: str) -> LLMResult:
        return LLMResult('{"route": "retrieve", "reason": "document fact"}', 6, 4, "provider")

    def complete_json(self, system: str, user: str, schema):
        result = self.complete(system, user)
        from documind.llm.providers import extract_json

        return schema.model_validate(extract_json(result.text)), result

    def stream(self, system: str, user: str, usage):
        usage.prompt_tokens = 9
        usage.completion_tokens = 7
        usage.token_source = "provider"
        chat = _EchoChat(reply=self.reply)
        for chunk in chat.stream("question"):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                yield text


def test_stream_forwards_chat_model_tokens(settings):
    from documind.ingestion.loaders import load_file

    settings.auto_ingest = True
    settings.retrieval_k = 20
    handbook = load_file(SAMPLE / "handbook.md")
    page = next(item.page for item in handbook.pages if "90 days" in item.text)
    reply = f"Chat logs are kept for 90 days. [handbook:{page}]"
    with TestClient(create_app(settings, model=_StreamingModel(reply))) as client:
        with client.stream(
            "POST",
            "/query/stream",
            json={"question": "How long are chat logs kept?"},
        ) as response:
            body = "".join(response.iter_text())
    events = _events(body)
    token_text = "".join(event["text"] for event in events if event["type"] == "token")
    assert token_text == reply
    result = events[-1]["answer"]
    assert result["grounded"] is True
    assert result["usage"]["token_source"] == "provider"
    assert result["usage"]["model"] == "echo"


def test_cli_ask(settings, tmp_path: Path):
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.update(
        {
            "LLM_PROVIDER": "local",
            "CHROMA_DIR": str(tmp_path / "cli-chroma"),
            "INDEX_PATH": str(tmp_path / "chunks.json"),
            "SQLITE_PATH": str(tmp_path / "analytics.sqlite"),
            "CORPUS_DIR": str(SAMPLE),
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "LOG_PATH": str(tmp_path / "usage.jsonl"),
            "AUTO_INGEST": "false",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "documind", "ingest", str(SAMPLE)],
        cwd=SAMPLE.parents[0],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    asked = subprocess.run(
        [sys.executable, "-m", "documind", "ask", "How long are chat logs kept?"],
        cwd=SAMPLE.parents[0],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert asked.returncode == 0, asked.stderr
    assert "90 days" in asked.stdout
