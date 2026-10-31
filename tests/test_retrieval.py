"""Hybrid retrieval: BM25 + dense, RRF fusion."""
import pytest
from langchain_core.documents import Document

from app import retrieval
from app.config import settings


def _doc(text, i):
    return Document(page_content=text, metadata={"source": "d.md", "chunk_index": i, "section": "S"})


STUB = _doc("Predefined values enum choices", 0)
OTHER = _doc("Order matters users me", 1)
TARGET = _doc("You can declare path parameters with the same syntax as format strings", 2)


class FakeStore:
    """Dense order = list order (stub on top, target last): a fake version of the embedding model's mistake."""

    def __init__(self, docs):
        self.docs = docs

    def get(self, include=None):
        return {
            "ids": [f"id{i}" for i in range(len(self.docs))],
            "documents": [d.page_content for d in self.docs],
            "metadatas": [d.metadata for d in self.docs],
        }

    def similarity_search_with_score(self, query, k=4):
        return [(d, 0.2 + 0.1 * i) for i, d in enumerate(self.docs[:k])]


@pytest.fixture
def store(monkeypatch):
    s = FakeStore([STUB, OTHER, TARGET])
    monkeypatch.setattr(retrieval, "get_vectorstore", lambda: s)
    retrieval._cache.update(key=None, bm25=None, docs=[])
    monkeypatch.setattr(settings, "candidate_k", 20)
    monkeypatch.setattr(settings, "rrf_k", 60)
    yield s
    retrieval._cache.update(key=None, bm25=None, docs=[])


def test_tokenize_drops_stopwords_and_stems_plurals():
    assert retrieval.tokenize("How do I declare path parameters?") == ["declare", "path", "parameter"]
    assert retrieval.tokenize("class") == ["class"]  # words ending in 'ss' are left alone


def test_bm25_rescues_chunk_that_dense_ranks_last(store):
    docs = retrieval.hybrid_search("declare path parameter", k=2)
    assert len(docs) == 2
    assert docs[0].page_content.startswith("You can declare")
    assert docs[0].metadata["via"] == "bm25+dense"
    assert docs[0].metadata["score"] == 0.6  # cosine = 1 - distance(0.4)
    assert "rrf" in docs[0].metadata


def test_multiple_queries_are_fused(store):
    # the first query has no BM25 match; the second one finds the target
    docs = retrieval.hybrid_search(["zzzz qqqq", "declare path parameter"], k=1)
    assert docs[0].page_content.startswith("You can declare")


def test_blank_queries_return_nothing(store):
    assert retrieval.hybrid_search(["", "  "]) == []


def test_empty_store_returns_nothing(monkeypatch):
    monkeypatch.setattr(retrieval, "get_vectorstore", lambda: FakeStore([]))
    retrieval._cache.update(key=None, bm25=None, docs=[])
    assert retrieval.hybrid_search("anything") == []
    retrieval._cache.update(key=None, bm25=None, docs=[])