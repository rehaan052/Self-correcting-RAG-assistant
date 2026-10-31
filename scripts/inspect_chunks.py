"""Debug: print all chunks of the path-parameters doc + the top-15 retrieval scores."""
import sys

from app.vectorstore import get_vectorstore

SOURCE_KEYWORD = "path-parameters"
QUERY = sys.argv[1] if len(sys.argv) > 1 else "How do I declare a path parameter?"


def main():
    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]
    print(f"Total chunks in DB: {len(docs)}")
    if metas:
        print(f"Metadata keys: {sorted(metas[0].keys())}")

    print(f"\n=== Chunks whose metadata contains '{SOURCE_KEYWORD}' ===")
    n = 0
    for doc, meta in zip(docs, metas):
        if SOURCE_KEYWORD in str(meta):
            n += 1
            print(f"\n--- chunk {n} | {len(doc)} chars | {meta}")
            print(doc[:300].replace("\n", " ") + (" ..." if len(doc) > 300 else ""))
    print(f"\n({n} chunks in total for path-parameters)")

    print(f"\n=== Top-15 retrieval for: {QUERY!r} ===")
    results = vs.similarity_search_with_relevance_scores(QUERY, k=15)
    for i, (d, score) in enumerate(results, 1):
        heading = d.metadata.get("heading") or d.metadata.get("header_path") or d.metadata
        print(f"{i:2d}. {score:.3f} | {len(d.page_content):4d} chars | {heading}")


if __name__ == "__main__":
    main()