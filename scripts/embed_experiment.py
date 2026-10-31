"""Experiment: embedding text badalne se root/'with types' chunks ka rank kaise badalta hai."""
import re

import numpy as np

from app.llm import get_embeddings
from app.vectorstore import get_vectorstore

QUERIES = ["How do I declare a path parameter?", "declare path parameter"]
CODE = re.compile(r"```.*?```", re.S)


def unit(m):
    m = np.array(m, dtype=float)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def label(m):
    return f"{m['source'].replace('fastapi-', '')}#{m['chunk_index']} {m['section'].split(' > ')[-1]}"


def main():
    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]
    emb = get_embeddings()

    variants = {
        "A raw (current)": docs,
        "B breadcrumb + text": [f"{m['section']}\n{d}" for d, m in zip(docs, metas)],
        "C breadcrumb + text, code hata ke": [
            f"{m['section']}\n{CODE.sub('', d)}" for d, m in zip(docs, metas)
        ],
    }

    targets = {
        i: label(m)
        for i, m in enumerate(metas)
        if m["source"] == "fastapi-path-parameters.md" and m["chunk_index"] in (0, 1)
    }

    for q in QUERIES:
        print(f"\n######## QUERY: {q!r}")
        qv = unit(emb.embed_query(q))
        for name, texts in variants.items():
            D = unit(emb.embed_documents(texts))
            scores = D @ qv
            order = list(np.argsort(-scores))
            print(f"\n[{name}]")
            for i, lab in targets.items():
                print(f"   target {lab}: rank {order.index(i) + 1}/{len(docs)}  score {scores[i]:.3f}")
            print("   top-5:")
            for i in order[:5]:
                print(f"     {scores[i]:.3f} | {len(docs[i]):4d} chars | {label(metas[i])}")


if __name__ == "__main__":
    main()