"""Hybrid retrieval: dense (embeddings) + BM25 (keywords), fused with Reciprocal Rank Fusion.

Why: the dense model ranks short or code-heavy chunks containing words like "path parameter" too high;
BM25 catches exact words ("declare", "path", "parameter"). Combining both rankings gives a better order.
"""
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings
from app.vectorstore import get_vectorstore

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on", "for", "and", "or",
    "how", "do", "does", "i", "you", "it", "this", "that", "with", "can", "what", "my", "me",
}
_cache: dict = {"key": None, "bm25": None, "docs": []}


def _stem(token: str) -> str:
    """A very light stemmer: 'parameters' == 'parameter'."""
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOP]


def _key(doc: Document) -> tuple:
    return (doc.metadata.get("source"), doc.metadata.get("chunk_index"))


def _get_index() -> tuple[BM25Okapi | None, list[Document]]:
    """The BM25 index is built from the store's chunks and rebuilds itself when the chunks change (after ingest)."""
    data = get_vectorstore().get(include=["documents", "metadatas"])
    key = tuple(sorted(data["ids"]))
    if _cache["key"] != key:
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        corpus = [tokenize(f"{d.metadata.get('section', '')}\n{d.page_content}") for d in docs]
        _cache.update(key=key, docs=docs, bm25=BM25Okapi(corpus) if corpus else None)
    return _cache["bm25"], _cache["docs"]


def hybrid_search(query: str | list[str], k: int | None = None) -> list[Document]:
    """Search with one or more queries; the dense and BM25 ranks of all of them are fused together with RRF."""
    k = k or settings.top_k
    queries = [query] if isinstance(query, str) else [q for q in query if q and q.strip()]
    bm25, docs = _get_index()
    if not docs or bm25 is None or not queries:
        return []

    vs = get_vectorstore()
    rrf: dict[tuple, float] = {}
    cosine: dict[tuple, float] = {}
    via: dict[tuple, set] = {}

    for q in queries:
        dense = vs.similarity_search_with_score(q, k=min(len(docs), 100))
        dense_order = []
        for d, dist in dense:
            key = _key(d)
            cosine[key] = max(cosine.get(key, -1.0), round(1 - float(dist), 3))
            dense_order.append(key)
        dense_order = dense_order[: settings.candidate_k]

        scores = bm25.get_scores(tokenize(q))
        ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
        bm25_order = [_key(docs[i]) for i in ranked if scores[i] > 0][: settings.candidate_k]

        for name, order in (("dense", dense_order), ("bm25", bm25_order)):
            for rank, key in enumerate(order, 1):
                rrf[key] = rrf.get(key, 0.0) + 1.0 / (settings.rrf_k + rank)
                via.setdefault(key, set()).add(name)

    by_key = {_key(d): d for d in docs}
    out = []
    for key in sorted(rrf, key=lambda x: -rrf[x])[:k]:
        d = by_key[key]
        out.append(
            Document(
                page_content=d.page_content,
                metadata={**d.metadata, "score": cosine.get(key, 0.0), "rrf": round(rrf[key], 4), "via": "+".join(sorted(via[key]))},
            )
        )
    return out