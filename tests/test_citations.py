import pytest
from pydantic import ValidationError

from documind.schemas import Answer, Citation


def test_grounded_answer_requires_matching_inline_citations():
    answer = Answer(
        answer="Chat logs are kept for 90 days. [handbook:3]",
        citations=[Citation(doc="handbook", page=3, chunk_id="handbook-p3-c0")],
        route="retrieve",
        grounded=True,
        confidence=0.9,
        low_confidence=False,
    )
    assert answer.citations[0].label == "[handbook:3]"


def test_inline_citation_missing_from_the_list_is_rejected():
    with pytest.raises(ValidationError):
        Answer(
            answer="Chat logs are kept for 90 days. [handbook:3]",
            citations=[],
            route="retrieve",
            grounded=False,
            confidence=0.2,
            low_confidence=True,
        )


def test_listed_citation_must_appear_inline():
    with pytest.raises(ValidationError):
        Answer(
            answer="Chat logs are kept for 90 days.",
            citations=[Citation(doc="handbook", page=3)],
            route="retrieve",
            grounded=False,
            confidence=0.2,
            low_confidence=True,
        )


def test_ungrounded_answer_cannot_look_confident():
    with pytest.raises(ValidationError):
        Answer(
            answer="I could not find this in the knowledge base.",
            route="retrieve",
            grounded=False,
            confidence=0.0,
            low_confidence=False,
        )


def test_sql_answer_has_no_citations():
    answer = Answer(
        answer="The document count is 3.",
        route="sql",
        grounded=True,
        confidence=0.85,
        low_confidence=False,
    )
    assert answer.citations == []


def test_grounded_document_answer_needs_a_citation():
    with pytest.raises(ValidationError):
        Answer(
            answer="Chat logs are kept for 90 days.",
            route="retrieve",
            grounded=True,
            confidence=0.8,
            low_confidence=False,
        )
