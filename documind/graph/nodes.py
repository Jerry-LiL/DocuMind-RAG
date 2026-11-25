"""Graph nodes. Each one returns a partial state update."""

from __future__ import annotations

import sqlite3

from pydantic import ValidationError

from documind.config import Settings
from documind.graph.grade import grade_retrieval, grade_sql, should_rewrite
from documind.graph.local_answer import clarification_text, extractive_answer, format_sql_answer
from documind.graph.prompts import (
    RETRIEVE_SYSTEM,
    ROUTER_SYSTEM,
    SQL_ANSWER_SYSTEM,
    SQL_SYSTEM,
    retrieve_user,
    sql_answer_user,
)
from documind.graph.state import GraphState
from documind.llm.providers import ModelProvider, StreamUsage
from documind.retrieval.store import CorpusStore
from documind.schemas import Answer, Citation, RouteDecision, SqlDecision
from documind.sql.tool import SqlRejected, execute_readonly, match_template
from documind.textutil import citation_pairs, estimate_tokens, strip_citations


def make_nodes(settings: Settings, store: CorpusStore, model: ModelProvider):
    def router(state: GraphState) -> dict:
        if model.provider == "local":
            decision = _heuristic(state["question"])
            return {
                "route": decision.route,
                "route_reason": decision.reason,
                "token_source": "estimate",
            }
        try:
            decision, result = model.complete_json(ROUTER_SYSTEM, state["question"], RouteDecision)
        except (ValidationError, ValueError):
            decision = _heuristic(state["question"])
            result = None
        update = {"route": decision.route, "route_reason": decision.reason}
        if result is not None:
            update.update(
                _add_usage(
                    state,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.token_source,
                )
            )
        return update

    def retrieve(state: GraphState) -> dict:
        hits = store.search(state["question"], k=settings.retrieval_k)
        documents = []
        for hit in hits:
            payload = hit.chunk.as_dict()
            payload["score"] = hit.score
            documents.append(payload)
        return {"documents": documents}

    def run_sql(state: GraphState) -> dict:
        statement = ""
        extra: dict = {}
        try:
            statement, extra = _choose_sql(state["question"], model, state)
            rows = execute_readonly(settings.sqlite_path, statement)
        except (SqlRejected, ValidationError, ValueError, sqlite3.Error) as exc:
            return {"sql": statement, "sql_rows": [], "sql_error": str(exc), **extra}
        return {"sql": statement, "sql_rows": rows, "sql_error": "", **extra}

    def clarify(state: GraphState) -> dict:
        text = clarification_text()
        answer = Answer(
            answer=text,
            citations=[],
            route="clarify",
            grounded=True,
            confidence=0.0,
            low_confidence=True,
            clarification=text,
            route_reason=state.get("route_reason") or "",
        )
        prompt_tokens = state.get("prompt_tokens", 0) + estimate_tokens(state["question"])
        completion_tokens = state.get("completion_tokens", 0) + estimate_tokens(text)
        return {
            "draft": text,
            "answer": answer.answer,
            "citations": [],
            "grounded": True,
            "confidence": 0.0,
            "low_confidence": True,
            "clarification": text,
            "needs_rewrite": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "token_source": state.get("token_source") or "estimate",
        }

    def generate(state: GraphState) -> dict:
        rewriting = state.get("rewrite_count", 0) > 0
        notes = state.get("grade_notes") or "" if rewriting else ""
        usage = StreamUsage()
        if state.get("route") == "sql":
            draft, usage = _draft_sql(state, model, notes)
        else:
            draft, usage = _draft_retrieval(state, model, notes)
        return {
            "draft": draft,
            "prompt_tokens": state.get("prompt_tokens", 0) + usage.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + usage.completion_tokens,
            "token_source": _merge_source(state.get("token_source"), usage.token_source),
        }

    def grade(state: GraphState) -> dict:
        route = state.get("route") or "retrieve"
        draft = state.get("draft") or ""
        if route == "sql":
            grounded, confidence, notes = grade_sql(
                draft,
                state.get("sql_rows") or [],
                state.get("sql_error") or "",
            )
            citations: list[Citation] = []
        else:
            pairs = citation_pairs(draft)
            grounded, confidence, notes = grade_retrieval(
                draft,
                pairs,
                state.get("documents") or [],
            )
            citations = _citations_for(pairs, state.get("documents") or [])
        low_confidence = (not grounded) or confidence < settings.low_confidence_threshold
        rewrite = should_rewrite(
            route,
            grounded,
            state.get("rewrite_count", 0),
            bool(state.get("documents")),
            state.get("sql_error") or "",
        )
        answer = _validated_answer(
            draft=draft,
            citations=citations,
            route=route,
            grounded=grounded,
            confidence=confidence,
            low_confidence=low_confidence,
            route_reason=state.get("route_reason") or "",
        )
        return {
            "answer": answer.answer,
            "citations": [item.model_dump() for item in answer.citations],
            "grounded": answer.grounded,
            "confidence": answer.confidence,
            "low_confidence": answer.low_confidence,
            "clarification": None,
            "grade_notes": notes,
            "needs_rewrite": rewrite,
            "rewrite_count": state.get("rewrite_count", 0) + (1 if rewrite else 0),
        }

    return {
        "router": router,
        "retrieve": retrieve,
        "sql": run_sql,
        "clarify": clarify,
        "generate": generate,
        "grade": grade,
    }


def _heuristic(question: str) -> RouteDecision:
    from documind.graph.route import heuristic_route

    return heuristic_route(question)


def _choose_sql(question: str, model: ModelProvider, state: GraphState) -> tuple[str, dict]:
    template = match_template(question)
    if template is not None or model.provider == "local":
        if template is None:
            raise SqlRejected("no catalog query matches that question")
        return template, {}
    decision, result = model.complete_json(SQL_SYSTEM, question, SqlDecision)
    return decision.sql, _add_usage(
        state,
        result.prompt_tokens,
        result.completion_tokens,
        result.token_source,
    )


def _draft_retrieval(
    state: GraphState,
    model: ModelProvider,
    notes: str,
) -> tuple[str, StreamUsage]:
    documents = state.get("documents") or []
    if model.provider == "local":
        draft = extractive_answer(state["question"], documents)
        usage = StreamUsage(
            prompt_tokens=estimate_tokens(
                state["question"] + " " + " ".join(doc["text"] for doc in documents)
            ),
            completion_tokens=estimate_tokens(draft),
            token_source="estimate",
        )
        return draft, usage
    user = retrieve_user(state["question"], documents, state.get("draft") or "", notes)
    return _stream_draft(model, RETRIEVE_SYSTEM, user)


def _draft_sql(state: GraphState, model: ModelProvider, notes: str) -> tuple[str, StreamUsage]:
    if state.get("sql_error"):
        draft = f"I could not run that analytics query: {state['sql_error']}"
        return draft, StreamUsage(
            prompt_tokens=estimate_tokens(state["question"]),
            completion_tokens=estimate_tokens(draft),
            token_source="estimate",
        )
    rows = state.get("sql_rows") or []
    if model.provider == "local":
        draft = format_sql_answer(rows)
        return draft, StreamUsage(
            prompt_tokens=estimate_tokens(state["question"] + str(rows)),
            completion_tokens=estimate_tokens(draft),
            token_source="estimate",
        )
    user = sql_answer_user(state["question"], state.get("sql") or "", rows, notes)
    return _stream_draft(model, SQL_ANSWER_SYSTEM, user)


def _stream_draft(model: ModelProvider, system: str, user: str) -> tuple[str, StreamUsage]:
    usage = StreamUsage()
    parts = list(model.stream(system, user, usage))
    return "".join(parts).strip(), usage


def _citations_for(pairs: list[tuple[str, int]], documents: list[dict]) -> list[Citation]:
    citations: list[Citation] = []
    for doc_name, page in pairs:
        match = next(
            (doc for doc in documents if doc["doc"] == doc_name and int(doc["page"]) == page),
            None,
        )
        if page < 1:
            continue
        citations.append(
            Citation(
                doc=doc_name,
                page=page,
                chunk_id=match["id"] if match else "",
            )
        )
    return citations


def _validated_answer(
    draft: str,
    citations: list[Citation],
    route: str,
    grounded: bool,
    confidence: float,
    low_confidence: bool,
    route_reason: str,
) -> Answer:
    try:
        return Answer(
            answer=draft,
            citations=citations,
            route=route,  # type: ignore[arg-type]
            grounded=grounded,
            confidence=confidence,
            low_confidence=low_confidence,
            route_reason=route_reason,
        )
    except ValidationError:
        # Drop markers the schema will not accept and surface the draft as ungrounded.
        return Answer(
            answer=strip_citations(draft).strip() or "I could not produce a valid answer.",
            citations=[],
            route=route,  # type: ignore[arg-type]
            grounded=False,
            confidence=0.0,
            low_confidence=True,
            route_reason=route_reason,
        )


def _add_usage(state: GraphState, prompt_tokens: int, completion_tokens: int, source: str) -> dict:
    return {
        "prompt_tokens": state.get("prompt_tokens", 0) + prompt_tokens,
        "completion_tokens": state.get("completion_tokens", 0) + completion_tokens,
        "token_source": _merge_source(state.get("token_source"), source),
    }


def _merge_source(current: str | None, new: str) -> str:
    current = current or ""
    if "estimate" in (current, new):
        return "estimate"
    return new or current or "estimate"
