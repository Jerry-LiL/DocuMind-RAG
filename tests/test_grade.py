from documind.graph.grade import grade_retrieval, grade_sql, should_rewrite


def test_grade_accepts_a_cited_sentence():
    documents = [
        {"doc": "handbook", "page": 3, "text": "Chat logs are kept for 90 days."},
    ]
    grounded, confidence, _notes = grade_retrieval(
        "Chat logs are kept for 90 days. [handbook:3]",
        [("handbook", 3)],
        documents,
    )
    assert grounded is True
    assert confidence > 0.7


def test_grade_rejects_a_citation_that_was_not_retrieved():
    documents = [
        {"doc": "handbook", "page": 3, "text": "Chat logs are kept for 90 days."},
    ]
    grounded, _confidence, notes = grade_retrieval(
        "The wifi password is swordfish. [policy:9]",
        [("policy", 9)],
        documents,
    )
    assert grounded is False
    assert "citation" in notes


def test_grade_rejects_numbers_missing_from_sql_rows():
    grounded, _confidence, notes = grade_sql(
        "The document count is 9.",
        [{"document_count": 3}],
        "",
    )
    assert grounded is False
    assert "numbers" in notes


def test_rewrite_runs_once_when_a_passage_was_retrieved():
    assert should_rewrite("retrieve", False, 0, True, "") is True
    assert should_rewrite("retrieve", False, 1, True, "") is False
    assert should_rewrite("retrieve", False, 0, False, "") is False
    assert should_rewrite("sql", False, 0, False, "only SELECT statements are allowed") is False
