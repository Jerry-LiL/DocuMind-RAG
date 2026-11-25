"""OpenAI and Ollama chat models, plus JSON parsing around them."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from documind.config import Settings
from documind.textutil import estimate_tokens

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    token_source: str


@dataclass
class StreamUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_source: str = "estimate"


class ModelProvider:
    """Chat completion for the graph. Embeddings live on the corpus store."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.llm_provider
        self._chat = None

    @property
    def model_name(self) -> str:
        if self.provider == "openai":
            return self.settings.openai_model
        if self.provider == "ollama":
            return self.settings.ollama_model
        return "local-hash"

    def complete(self, system: str, user: str) -> LLMResult:
        if self.provider == "local":
            raise RuntimeError("local mode does not call a chat model")
        message = self._chat_model().invoke(
            [_system(system), _human(user)],
        )
        text = message_text(message)
        prompt_tokens, completion_tokens = usage_counts(message)
        source = "provider"
        if prompt_tokens == 0 and completion_tokens == 0:
            prompt_tokens = estimate_tokens(system + user)
            completion_tokens = estimate_tokens(text)
            source = "estimate"
        return LLMResult(text, prompt_tokens, completion_tokens, source)

    def stream(self, system: str, user: str, usage: StreamUsage) -> Iterator[str]:
        if self.provider == "local":
            raise RuntimeError("local mode does not call a chat model")
        parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for chunk in self._chat_model().stream([_system(system), _human(user)]):
            text = message_text(chunk)
            if text:
                parts.append(text)
                yield text
            seen_prompt, seen_completion = usage_counts(chunk)
            if seen_prompt or seen_completion:
                prompt_tokens = seen_prompt
                completion_tokens = seen_completion
        full = "".join(parts)
        if prompt_tokens == 0 and completion_tokens == 0:
            usage.prompt_tokens = estimate_tokens(system + user)
            usage.completion_tokens = estimate_tokens(full)
            usage.token_source = "estimate"
        else:
            usage.prompt_tokens = prompt_tokens
            usage.completion_tokens = completion_tokens
            usage.token_source = "provider"

    def complete_json(self, system: str, user: str, schema: type[T]) -> tuple[T, LLMResult]:
        result = self.complete(
            system + "\nRespond with one JSON object and no markdown.",
            user,
        )
        parsed = schema.model_validate(extract_json(result.text))
        return parsed, result

    def _chat_model(self):
        if self._chat is not None:
            return self._chat
        if self.provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            from langchain_openai import ChatOpenAI

            self._chat = ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                temperature=0,
                streaming=True,
                stream_usage=True,
            )
            return self._chat
        from langchain_ollama import ChatOllama

        self._chat = ChatOllama(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
            temperature=0,
        )
        return self._chat


def build_model(settings: Settings) -> ModelProvider:
    return ModelProvider(settings)


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model did not return JSON")
    return json.loads(cleaned[start : end + 1])


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


def usage_counts(message) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return (0, 0)
    if isinstance(usage, dict):
        return (int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _system(content: str):
    from langchain_core.messages import SystemMessage

    return SystemMessage(content=content)


def _human(content: str):
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=content)
