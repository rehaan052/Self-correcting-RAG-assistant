"""Shared fixtures: a fake LLM and a fake retriever, so tests run without Groq, internet or embeddings."""
import os

os.environ.setdefault("GROQ_API_KEY", "test-key")  # settings.groq_api_key is required

import re
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app import nodes, prompts
from app.config import settings
from app.graph import get_graph
from app.nodes import ChunkGrade, GradeResult, GroundingResult, QueryAnalysis

ROLE_PROMPTS = {
    "analyze": prompts.ANALYZE_SYSTEM,
    "grade": prompts.GRADE_SYSTEM,
    "rewrite": prompts.REWRITE_SYSTEM,
    "generate": prompts.GENERATE_SYSTEM,
    "grounding": prompts.GROUNDING_SYSTEM,
}


def make_doc(text="chunk text", source="doc.md", section="Sec", score=0.5) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "section": section, "url": f"https://example.com/{source}", "score": score},
    )


class FakeLLM:
    """Identifies the calling node from its system prompt and returns scripted answers.

    llm.on("grade", resp1, resp2): calls get resp1, resp2 in order; the last response repeats.
    Response: pydantic object / string / Exception (it will be raised).
    With no script, a sensible default is used (all chunks KEEP, grounded=True, ...).
    """

    def __init__(self):
        self.script: dict[str, list] = {}
        self.calls: list[tuple[str, list]] = []
        self.defaults = {
            "analyze": self._default_analyze,
            "grade": self._default_grade,
            "rewrite": lambda messages: f"rewritten query {self.count('rewrite')}",
            "generate": lambda messages: "Answer text [1].",
            "grounding": lambda messages: self.grounding(True),
        }

    # ---- scripting helpers ----
    def on(self, role, *responses):
        self.script[role] = list(responses)
        return self

    def count(self, role) -> int:
        return sum(1 for r, _ in self.calls if r == role)

    def human(self, role, n=-1) -> str:
        """Human message text of the n-th call of this role."""
        return [m for r, m in self.calls if r == role][n][-1].content

    @staticmethod
    def grades(*flags) -> GradeResult:
        return GradeResult(
            grades=[ChunkGrade(id=i, reason=f"reason {i}", relevant=f) for i, f in enumerate(flags, 1)]
        )

    @staticmethod
    def grounding(ok: bool, *claims) -> GroundingResult:
        return GroundingResult(grounded=ok, unsupported_claims=list(claims))

    @staticmethod
    def analysis(standalone, search, qtype="how_to") -> QueryAnalysis:
        return QueryAnalysis(standalone_question=standalone, search_query=search, query_type=qtype)

    # ---- defaults ----
    @staticmethod
    def _default_analyze(messages):
        q = messages[-1].content.split("Latest question:", 1)[1].strip()
        return QueryAnalysis(standalone_question=q, query_type="how_to", search_query=q)

    @staticmethod
    def _default_grade(messages):
        n = len(re.findall(r"^\[(\d+)\] \(", messages[-1].content, re.M))
        return GradeResult(grades=[ChunkGrade(id=i, reason="ok", relevant=True) for i in range(1, n + 1)])

    # ---- LangChain-like interface (what nodes.py uses) ----
    def _respond(self, messages):
        system = messages[0].content
        role = next((r for r, p in ROLE_PROMPTS.items() if p == system), None)
        assert role, "FakeLLM: unknown system prompt"
        self.calls.append((role, messages))
        queue = self.script.get(role)
        item = (queue.pop(0) if len(queue) > 1 else queue[0]) if queue else self.defaults[role]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        return item

    def invoke(self, messages):
        return AIMessage(content=self._respond(messages))

    def with_structured_output(self, schema, method=None, **kwargs):
        return SimpleNamespace(invoke=lambda messages: self._respond(messages))


class FakeRetriever:
    """Replaces nodes.hybrid_search. The queries of every call are recorded in `calls`."""

    def __init__(self):
        self.script: list[list[Document]] = []
        self.calls: list[list[str]] = []

    def on(self, *results):
        self.script = [list(r) for r in results]
        return self

    def __call__(self, queries, k=None):
        self.calls.append(list(queries))
        if not self.script:
            return [make_doc("default chunk one", section="A"), make_doc("default chunk two", section="B")]
        return list(self.script.pop(0) if len(self.script) > 1 else self.script[0])


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Keep your .env from affecting tests, and give every test a fresh graph cache."""
    defaults = dict(
        hybrid_search=True,
        top_k=8,
        max_retries=2,
        max_generation_attempts=2,
        enable_hallucination_check=True,
        enable_web_search=True,
        tavily_api_key="",
        max_history_turns=5,
    )
    for key, value in defaults.items():
        monkeypatch.setattr(settings, key, value)
    get_graph.cache_clear()
    yield
    get_graph.cache_clear()


@pytest.fixture
def fake(monkeypatch):
    llm, retriever = FakeLLM(), FakeRetriever()
    monkeypatch.setattr(nodes, "get_llm", lambda: llm)
    monkeypatch.setattr(nodes, "hybrid_search", retriever)
    return SimpleNamespace(llm=llm, retriever=retriever)


@pytest.fixture
def doc():
    return make_doc