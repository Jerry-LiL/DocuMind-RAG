"""HTTP API. /query returns the graded answer; /query/stream emits it as server-sent events."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from documind.config import Settings, get_settings
from documind.graph.builder import build_graph
from documind.graph.state import initial_state
from documind.ingestion.pipeline import ingest_directory, ingest_path
from documind.llm.providers import ModelProvider, build_model, message_text
from documind.logging_utils import configure_logging, log_usage
from documind.retrieval.store import CorpusStore
from documind.schemas import IngestResponse, QueryRequest, QueryResponse, Usage

logger = logging.getLogger("documind")
_UPLOAD_SUFFIXES = {".pdf", ".md", ".markdown"}


def create_app(settings: Settings | None = None, model: ModelProvider | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = CorpusStore(app.state.settings)
        if app.state.settings.auto_ingest and store.chunk_count() == 0:
            folder = app.state.settings.corpus_dir
            if folder.exists():
                summary = ingest_directory(folder, store, app.state.settings)
                logger.info("auto-ingested %s chunks", summary.chunk_count)
        app.state.store = store
        app.state.model = app.state.model_override or build_model(app.state.settings)
        app.state.graph = build_graph(app.state.settings, store, app.state.model)
        yield

    configure_logging()
    app = FastAPI(title="DocuMind", lifespan=lifespan)
    app.state.settings = settings
    app.state.model_override = model
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/")
    def root() -> dict:
        return {
            "name": "DocuMind",
            "endpoints": ["/health", "/query", "/query/stream", "/ingest", "/ingest/sample"],
        }

    @app.get("/health")
    def health(request: Request) -> dict:
        store: CorpusStore = request.app.state.store
        model: ModelProvider = request.app.state.model
        return {
            "status": "ok",
            "provider": model.provider,
            "model": model.model_name,
            "documents": store.document_count(),
            "chunks": store.chunk_count(),
        }

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, request: Request) -> QueryResponse:
        started = time.perf_counter()
        state = request.app.state.graph.invoke(initial_state(body.question))
        return _finish(request, state, started, streamed_tokens=False)

    @app.post("/query/stream")
    def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
        return StreamingResponse(
            _event_stream(request, body.question),
            media_type="text/event-stream",
        )

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: Request, file: UploadFile = File(...)) -> IngestResponse:
        name = Path(file.filename or "upload").name
        if Path(name).suffix.lower() not in _UPLOAD_SUFFIXES:
            raise HTTPException(status_code=400, detail="only PDF and Markdown files are supported")
        settings: Settings = request.app.state.settings
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        destination = settings.upload_dir / name
        destination.write_bytes(await file.read())
        document = ingest_path(destination, request.app.state.store, settings)
        return IngestResponse(
            documents=[document],
            chunk_count=request.app.state.store.chunk_count(),
        )

    @app.post("/ingest/sample", response_model=IngestResponse)
    def ingest_sample(request: Request) -> IngestResponse:
        settings: Settings = request.app.state.settings
        if not settings.corpus_dir.exists():
            raise HTTPException(status_code=404, detail="corpus directory is missing")
        return ingest_directory(settings.corpus_dir, request.app.state.store, settings)


async def _event_stream(request: Request, question: str) -> AsyncIterator[str]:
    started = time.perf_counter()
    graph = request.app.state.graph
    tokens_seen = False
    final: dict | None = None
    try:
        async for event in graph.astream_events(initial_state(question), version="v2"):
            name = event.get("name")
            node = (event.get("metadata") or {}).get("langgraph_node")
            if event["event"] == "on_chat_model_stream":
                text = message_text(event["data"].get("chunk"))
                if text:
                    tokens_seen = True
                    yield _sse({"type": "token", "text": text})
            elif event["event"] == "on_chain_end" and node == "router":
                output = event["data"].get("output") or {}
                if not isinstance(output, dict):
                    continue
                yield _sse(
                    {
                        "type": "route",
                        "route": output.get("route"),
                        "reason": output.get("route_reason") or "",
                    }
                )
            elif event["event"] == "on_chain_end" and node == "retrieve":
                output = event["data"].get("output") or {}
                if not isinstance(output, dict):
                    continue
                documents = output.get("documents") or []
                yield _sse(
                    {
                        "type": "retrieval",
                        "passages": [
                            {
                                "doc": item["doc"],
                                "page": item["page"],
                                "score": item.get("score", 0),
                            }
                            for item in documents
                        ],
                    }
                )
            elif event["event"] == "on_chain_end" and node == "sql":
                output = event["data"].get("output") or {}
                if not isinstance(output, dict):
                    continue
                yield _sse(
                    {
                        "type": "sql",
                        "sql": output.get("sql") or "",
                        "error": output.get("sql_error") or None,
                    }
                )
            elif event["event"] == "on_chain_end" and node == "grade":
                output = event["data"].get("output") or {}
                if not isinstance(output, dict):
                    continue
                if output.get("needs_rewrite"):
                    yield _sse({"type": "status", "message": "revising answer"})
            elif event["event"] == "on_chain_end" and name == "LangGraph" and node is None:
                final = event["data"].get("output")
    except Exception:
        logger.exception("stream failed")
        yield _sse({"type": "error", "message": "query failed"})
        return

    if final is None:
        yield _sse({"type": "error", "message": "query produced no result"})
        return
    if not tokens_seen:
        for piece in _pieces(final.get("answer") or "", size=24):
            yield _sse({"type": "token", "text": piece})
    response = _finish(request, final, started, streamed_tokens=tokens_seen)
    yield _sse({"type": "result", "answer": response.model_dump()})


def _finish(request: Request, state: dict, started: float, streamed_tokens: bool) -> QueryResponse:
    settings: Settings = request.app.state.settings
    model: ModelProvider = request.app.state.model
    route = state.get("route") or "clarify"
    usage = Usage(
        request_id=uuid.uuid4().hex[:12],
        model=model.model_name,
        prompt_tokens=int(state.get("prompt_tokens") or 0),
        completion_tokens=int(state.get("completion_tokens") or 0),
        token_source=state.get("token_source") or "estimate",
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        route=route,
    )
    response = QueryResponse(
        answer=state.get("answer") or "",
        citations=state.get("citations") or [],
        route=route,
        grounded=bool(state.get("grounded")),
        confidence=float(state.get("confidence") or 0),
        low_confidence=bool(state.get("low_confidence", True)),
        clarification=state.get("clarification"),
        route_reason=state.get("route_reason") or "",
        usage=usage,
    )
    log_usage(
        settings,
        {
            "request_id": usage.request_id,
            "route": usage.route,
            "model": usage.model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "token_source": usage.token_source,
            "latency_ms": usage.latency_ms,
            "grounded": response.grounded,
            "low_confidence": response.low_confidence,
            "streamed_tokens": streamed_tokens,
        },
    )
    return response


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _pieces(text: str, size: int) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


app = create_app()
