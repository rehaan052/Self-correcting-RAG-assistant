"""Graph behaviour: routing, retry limit, grading, fallback, grounding loop, fail-open."""
import pytest
from langgraph.graph import END

from app import nodes, prompts
from app.config import settings
from app.graph import (
    _status,
    route_after_generation,
    route_after_grounding,
    route_after_grading,
    route_after_web_search,
    run_query,
)


def roles(fake):
    return [r for r, _ in fake.llm.calls]


# ------------------------------------------------------------------ happy path


def test_happy_path(fake, doc):
    fake.retriever.on([doc("path param syntax", section="A"), doc("unrelated", section="B")])
    fake.llm.on("grade", fake.llm.grades(True, False))
    fake.llm.on("generate", "Use braces [1].")

    state = run_query("How do I declare a path parameter?")

    assert state["status"] == "answered"
    assert state["retry_count"] == 0
    assert state["grounded"] is True
    assert [d.page_content for d in state["relevant_documents"]] == ["path param syntax"]
    assert [s["id"] for s in state["sources"]] == [1]
    assert roles(fake) == ["analyze", "grade", "generate", "grounding"]
    assert len(fake.retriever.calls) == 1


def test_retrieval_uses_search_query_and_standalone_question(fake):
    fake.llm.on(
        "analyze",
        fake.llm.analysis(
            "What happens if I pass a string to an int path parameter?", "path parameter int string error", "conceptual"
        ),
    )
    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    run_query("What happens if I pass a string there?", history)

    assert fake.retriever.calls[0] == [
        "path parameter int string error",
        "What happens if I pass a string to an int path parameter?",
    ]
    # the grader gets the standalone question, not the one with "there"
    assert "Question: What happens if I pass a string to an int path parameter?" in fake.llm.human("grade")


def test_history_is_trimmed_to_max_turns(fake):
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"} for i in range(20)]
    run_query("follow up", history)
    human = fake.llm.human("analyze")
    assert "msg-19" in human and "msg-10" in human  # last 10 messages (= 5 turns)
    assert "msg-9" not in human and "msg-0" not in human


# ------------------------------------------------------------------ retry / rewrite


def test_rewrite_then_success(fake, doc):
    fake.retriever.on([doc("off topic")], [doc("the answer")])
    fake.llm.on("grade", fake.llm.grades(False), fake.llm.grades(True))
    fake.llm.on("rewrite", "better query")

    state = run_query("original question")

    assert state["status"] == "answered"
    assert state["retry_count"] == 1
    assert state["search_query"] == "better query"
    assert fake.retriever.calls[1][0] == "better query"
    assert roles(fake) == ["analyze", "grade", "rewrite", "grade", "generate", "grounding"]


def test_retries_exhausted_falls_back(fake, doc):
    fake.retriever.on([doc("irrelevant")])
    fake.llm.on("grade", fake.llm.grades(False))

    state = run_query("q")

    assert state["status"] == "no_answer"
    assert state["retry_count"] == 2
    assert state["answer"] == prompts.FALLBACK_ANSWER
    assert state["sources"] == []
    assert len(fake.retriever.calls) == 3
    assert (fake.llm.count("rewrite"), fake.llm.count("grade")) == (2, 3)
    assert fake.llm.count("generate") == 0 and fake.llm.count("grounding") == 0
    # trace reducer: every node's line is appended, not overwritten
    assert state["trace"][0].startswith("analyze_query")
    assert sum(t.startswith("retrieve[") for t in state["trace"]) == 3
    assert state["trace"][-1].startswith("fallback")


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_retry_limit_is_respected(fake, doc, monkeypatch, limit):
    monkeypatch.setattr(settings, "max_retries", limit)
    fake.retriever.on([doc("irrelevant")])
    fake.llm.on("grade", fake.llm.grades(False))

    state = run_query("q")

    assert state["retry_count"] == limit
    assert fake.llm.count("rewrite") == limit
    assert len(fake.retriever.calls) == limit + 1
    assert state["status"] == "no_answer"


def test_empty_retrieval_skips_grader_and_still_retries(fake):
    fake.retriever.on([])
    state = run_query("q")
    assert fake.llm.count("grade") == 0
    assert state["retry_count"] == 2
    assert state["status"] == "no_answer"
    assert any("nothing retrieved" in t for t in state["trace"])


# ------------------------------------------------------------------ web search fallback


def _fake_web(docs):
    calls = []

    def web(state):
        calls.append(state["standalone_question"])
        return {"relevant_documents": docs, "used_web_search": True, "trace": [f"web_search: {len(docs)} results"]}

    web.calls = calls
    return web


def test_web_search_used_after_retries_when_enabled(fake, doc, monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test")
    web = _fake_web([doc("web content", source="web")])
    monkeypatch.setattr(nodes, "web_search", web)  # the graph is not built yet (cache is cleared)
    fake.retriever.on([doc("irrelevant")])
    fake.llm.on("grade", fake.llm.grades(False))

    state = run_query("q")

    assert state["status"] == "answered_from_web"
    assert state["used_web_search"] is True
    assert web.calls == ["q"]
    assert fake.llm.count("grade") == 3  # web results are not graded
    assert fake.llm.count("generate") == 1


def test_web_search_with_no_results_falls_back(fake, doc, monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test")
    monkeypatch.setattr(nodes, "web_search", _fake_web([]))
    fake.retriever.on([doc("irrelevant")])
    fake.llm.on("grade", fake.llm.grades(False))

    state = run_query("q")
    assert state["status"] == "no_answer"
    assert state["answer"] == prompts.FALLBACK_ANSWER


def test_web_search_skipped_when_flag_off(fake, doc, monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test")
    monkeypatch.setattr(settings, "enable_web_search", False)
    web = _fake_web([doc("web content")])
    monkeypatch.setattr(nodes, "web_search", web)
    fake.retriever.on([doc("irrelevant")])
    fake.llm.on("grade", fake.llm.grades(False))

    state = run_query("q")
    assert web.calls == []
    assert state["status"] == "no_answer"


# ------------------------------------------------------------------ grounding loop


def test_ungrounded_answer_is_regenerated_with_feedback(fake):
    fake.llm.on("generate", "Draft with extra claim [1].", "Clean answer [1].")
    fake.llm.on("grounding", fake.llm.grounding(False, "retries three times"), fake.llm.grounding(True))

    state = run_query("q")

    assert fake.llm.count("generate") == 2
    assert state["answer"] == "Clean answer [1]."
    assert state["grounded"] is True
    assert state["status"] == "answered"
    assert state["generation_attempts"] == 2
    assert "retries three times" in fake.llm.human("generate", 1)
    assert "retries three times" not in fake.llm.human("generate", 0)


def test_grounding_never_passes_returns_unverified(fake):
    fake.llm.on("grounding", fake.llm.grounding(False, "bad claim"))
    state = run_query("q")
    assert fake.llm.count("generate") == 2  # max_generation_attempts
    assert fake.llm.count("grounding") == 2
    assert state["grounded"] is False
    assert state["status"] == "unverified"
    assert state["answer"]  # the answer is still returned, with the flag


@pytest.mark.parametrize("attempts", [1, 3])
def test_generation_attempt_limit(fake, monkeypatch, attempts):
    monkeypatch.setattr(settings, "max_generation_attempts", attempts)
    fake.llm.on("grounding", fake.llm.grounding(False, "bad claim"))
    run_query("q")
    assert fake.llm.count("generate") == attempts


def test_hallucination_check_can_be_disabled(fake, monkeypatch):
    monkeypatch.setattr(settings, "enable_hallucination_check", False)
    state = run_query("q")
    assert fake.llm.count("grounding") == 0
    assert state["grounded"] is None
    assert state["status"] == "answered"


# ------------------------------------------------------------------ fail-open + grader quirks


def test_grader_failure_keeps_all_chunks(fake):
    fake.llm.on("grade", RuntimeError("boom"))
    state = run_query("q")
    assert len(state["relevant_documents"]) == 2
    assert state["status"] == "answered"
    assert any("grader failed" in t for t in state["trace"])


def test_analysis_failure_uses_raw_question(fake):
    fake.llm.on("analyze", RuntimeError("down"))
    state = run_query("raw question")
    assert state["standalone_question"] == "raw question"
    assert fake.retriever.calls[0] == ["raw question"]
    assert state["status"] == "answered"
    assert any("analysis failed" in t for t in state["trace"])


def test_grounding_check_failure_is_skipped(fake):
    fake.llm.on("grounding", RuntimeError("down"))
    state = run_query("q")
    assert state["grounded"] is True
    assert state["status"] == "answered"
    assert any("skipped" in t for t in state["trace"])


def test_zero_based_grader_ids_are_corrected(fake, doc):
    from app.nodes import ChunkGrade, GradeResult

    fake.retriever.on([doc("first"), doc("second")])
    fake.llm.on(
        "grade",
        GradeResult(grades=[ChunkGrade(id=0, reason="no", relevant=False), ChunkGrade(id=1, reason="yes", relevant=True)]),
    )
    state = run_query("q")
    assert [d.page_content for d in state["relevant_documents"]] == ["second"]


def test_chunk_missing_from_grader_output_is_dropped(fake, doc):
    fake.retriever.on([doc("first"), doc("second")])
    fake.llm.on("grade", fake.llm.grades(True))  # only chunk 1 was graded
    state = run_query("q")
    assert [d.page_content for d in state["relevant_documents"]] == ["first"]
    assert any("did not grade this chunk" in t for t in state["trace"])


def test_dense_only_mode(fake, doc, monkeypatch):
    class FakeStore:
        def similarity_search_with_score(self, query, k):
            return [(doc("x"), 0.25)]

    monkeypatch.setattr(settings, "hybrid_search", False)
    monkeypatch.setattr(nodes, "get_vectorstore", lambda: FakeStore())
    state = run_query("q")
    assert any(t.startswith("retrieve[dense]") for t in state["trace"])
    assert state["documents"][0].metadata["score"] == 0.75


# ------------------------------------------------------------------ citations


def test_oss_style_citations_are_normalised(fake):
    fake.llm.on("generate", "Use int 【2†L1-L7】 and 【1】.")
    state = run_query("q")
    assert state["answer"] == "Use int [2] and [1]."
    assert [s["id"] for s in state["sources"]] == [1, 2]


def test_only_cited_sources_are_returned(fake, doc):
    fake.retriever.on([doc("a"), doc("b"), doc("c")])
    fake.llm.on("generate", "Only [2] matters.")
    state = run_query("q")
    assert [s["id"] for s in state["sources"]] == [2]


def test_no_citation_returns_all_sources(fake, doc):
    fake.retriever.on([doc("a"), doc("b"), doc("c")])
    fake.llm.on("generate", "No citations here.")
    state = run_query("q")
    assert [s["id"] for s in state["sources"]] == [1, 2, 3]


# ------------------------------------------------------------------ routing functions (pure)


def test_route_after_grading(monkeypatch, doc):
    assert route_after_grading({"relevant_documents": [doc("x")]}) == "generate"
    none_left = {"relevant_documents": [], "retry_count": 0, "max_retries": 2}
    assert route_after_grading(none_left) == "rewrite_query"
    exhausted = {"relevant_documents": [], "retry_count": 2, "max_retries": 2}
    assert route_after_grading(exhausted) == "fallback"
    monkeypatch.setattr(settings, "tavily_api_key", "key")
    assert route_after_grading(exhausted) == "web_search"
    monkeypatch.setattr(settings, "enable_web_search", False)
    assert route_after_grading(exhausted) == "fallback"


def test_route_after_web_search(doc):
    assert route_after_web_search({"relevant_documents": [doc("x")]}) == "generate"
    assert route_after_web_search({"relevant_documents": []}) == "fallback"


def test_route_after_generation(monkeypatch):
    assert route_after_generation({}) == "check_grounding"
    monkeypatch.setattr(settings, "enable_hallucination_check", False)
    assert route_after_generation({}) == END


def test_route_after_grounding():
    assert route_after_grounding({"grounded": True}) == END
    assert route_after_grounding({}) == END
    assert route_after_grounding({"grounded": False, "generation_attempts": 1}) == "generate"
    assert route_after_grounding({"grounded": False, "generation_attempts": 2}) == END


@pytest.mark.parametrize(
    "state, expected",
    [
        ({"relevant_documents": []}, "no_answer"),
        ({"relevant_documents": ["d"], "grounded": False}, "unverified"),
        ({"relevant_documents": ["d"], "grounded": True, "used_web_search": True}, "answered_from_web"),
        ({"relevant_documents": ["d"], "grounded": None}, "answered"),
    ],
)
def test_status(state, expected):
    assert _status(state) == expected