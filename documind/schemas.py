"""Pydantic models for citations, graded answers, and the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from documind.textutil import citation_pairs

RouteName = Literal["retrieve", "sql", "clarify"]
TokenSource = Literal["provider", "estimate"]


class Citation(BaseModel):
    doc: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    page: int = Field(ge=1)
    chunk_id: str = ""

    @property
    def label(self) -> str:
        return f"[{self.doc}:{self.page}]"


class Answer(BaseModel):
    """A graded answer. Inline [Doc:Page] markers and the citation list must agree."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: RouteName
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool
    clarification: str | None = None
    route_reason: str = ""

    @model_validator(mode="after")
    def _citations_agree(self) -> Answer:
        inline = citation_pairs(self.answer)
        listed = [(item.doc, item.page) for item in self.citations]
        if set(inline) != set(listed):
            raise ValueError("inline [Doc:Page] citations must match the citation list")
        if self.route == "retrieve" and self.grounded and not self.citations:
            raise ValueError("a grounded document answer needs at least one citation")
        if self.route != "retrieve" and self.citations:
            raise ValueError("only document answers carry citations")
        if not self.grounded and not self.low_confidence:
            raise ValueError("ungrounded answers must be flagged low confidence")
        return self


class Usage(BaseModel):
    request_id: str
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    token_source: TokenSource
    latency_ms: float = Field(ge=0)
    route: RouteName


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    route: RouteName
    grounded: bool
    confidence: float
    low_confidence: bool
    clarification: str | None = None
    route_reason: str = ""
    usage: Usage


class IngestedDocument(BaseModel):
    doc: str
    filename: str
    file_type: str
    page_count: int
    chunk_count: int


class IngestResponse(BaseModel):
    documents: list[IngestedDocument]
    chunk_count: int


class RouteDecision(BaseModel):
    route: RouteName
    reason: str = ""


class SqlDecision(BaseModel):
    sql: str = Field(min_length=1)
