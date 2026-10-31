"""Ingestion pipeline: load -> chunk -> embed -> store.

Run:  python -m app.ingest            (ingests every .md file in data/docs)
"""
import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import settings

SOURCE_COMMENT_RE = re.compile(r"^<!--\s*source:\s*(\S+)\s*-->\s*\n?")
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CODE_RE = re.compile(r"```.*?```", re.DOTALL)


def load_markdown(path: Path) -> tuple[str, dict]:
    """Read a file and extract its metadata (source URL, title)."""
    text = path.read_text(encoding="utf-8")
    url = ""
    m = SOURCE_COMMENT_RE.match(text)
    if m:
        url = m.group(1)
        text = text[m.end():]
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    return text, {"source": path.name, "title": title, "url": url}


FENCE_RE = re.compile(r"(```[^\n]*\n.*?\n```)", re.DOTALL)


def _split_code_block(block: str, max_chars: int) -> list[str]:
    """Split a large code block on line boundaries and re-wrap every piece in a fence,
    so each chunk is a valid code block on its own."""
    header, _, rest = block.partition("\n")  # "```python"
    body = rest.rsplit("\n```", 1)[0]
    pieces, current = [], ""
    for line in body.split("\n"):
        if current and len(current) + len(line) + 1 > max_chars:
            pieces.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        pieces.append(current)
    return [f"{header}\n{p.rstrip()}\n```" for p in pieces]


def _pieces(section_text: str) -> list[str]:
    """Split a section into text pieces and code blocks.
    Text is split with overlap; short code blocks are kept whole."""
    size, overlap = settings.chunk_size, settings.chunk_overlap
    max_code = int(size * 1.5)  # code blocks up to this size are never split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)

    out: list[str] = []
    for part in FENCE_RE.split(section_text):
        if not part.strip():
            continue
        if part.startswith("```"):
            out.extend([part] if len(part) <= max_code else _split_code_block(part, size))
        else:
            out.extend(text_splitter.split_text(part))
    return out


def _pack(pieces: list[str]) -> list[str]:
    """Join small pieces (for example an intro line and its code block) up to chunk_size."""
    packed: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > settings.chunk_size:
            packed.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        packed.append(current)
    return packed


def _related(a: str, b: str) -> bool:
    """Is section b inside section a, the same section, or a sibling of it? (judged by heading path)"""
    pa, pb = a.split(" > "), b.split(" > ")
    if pb[: len(pa)] == pa:  # b is inside a (or the same)
        return True
    return len(pa) > 1 and pb[: len(pa) - 1] == pa[:-1]  # siblings under the same parent


def _merge_small(chunks: list[Document]) -> list[Document]:
    """Merge stub chunks (for example a heading with one or two lines) into the next related chunk.
    Otherwise the embedding model ranks them very close to the query although they contain no answer.
    Measured: stubs ranked in the top 2 while the real answer chunk was at rank 10-18."""
    min_chars = settings.min_chunk_chars
    max_chars = int(settings.chunk_size * 1.5)
    merged: list[Document] = []
    buf: Document | None = None
    for doc in chunks:
        if buf is None:
            buf = doc
            continue
        small = len(buf.page_content) < min_chars
        fits = len(buf.page_content) + len(doc.page_content) + 2 <= max_chars
        if small and fits and _related(buf.metadata["section"], doc.metadata["section"]):
            buf = Document(page_content=f"{buf.page_content}\n\n{doc.page_content}", metadata=buf.metadata)
        else:
            merged.append(buf)
            buf = doc
    if buf is not None:
        merged.append(buf)
    return merged


def chunk_markdown(text: str, base_meta: dict) -> list[Document]:
    """Three-stage, code-aware chunking:

    1. Split on Markdown headers, so every chunk stays inside one topic and knows its
       'section' (for example "Path Parameters > Predefined values").
    2. Inside a section, split text into ~chunk_size characters with overlap.
       Keep code blocks whole (up to 1.5x); split larger ones on lines and
       re-wrap every piece in a fence. Then join small pieces.
    3. Merge remaining stub chunks (< min_chunk_chars) into the next related chunk.
    """
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)

    chunks: list[Document] = []
    for section in header_splitter.split_text(text):
        path = [section.metadata[k] for _, k in HEADERS if k in section.metadata]
        section_name = " > ".join(path) if path else base_meta["title"]
        for content in _pack(_pieces(section.page_content)):
            content = content.strip()
            if len(content) < 40:  # a bare heading or an empty piece is useless
                continue
            chunks.append(
                Document(page_content=content, metadata={**base_meta, "section": section_name})
            )

    chunks = _merge_small(chunks)
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_index"] = i
    return chunks


def embedding_text(doc: Document) -> str:
    """Text used for the embedding: heading breadcrumb + prose (code removed).
    Code blocks diluted the whole embedding (measured: root chunk score 0.29 -> 0.45 after removal).
    The store keeps the ORIGINAL text (for generation and grading); this is only for building the vector.
    If there is very little prose (the chunk is almost all code), the code is kept."""
    prose = re.sub(r"\n{3,}", "\n\n", CODE_RE.sub("", doc.page_content)).strip()
    body = prose if len(prose) >= 60 else doc.page_content
    return f"{doc.metadata['section']}\n{body}"


def make_id(doc: Document) -> str:
    raw = f"{doc.metadata['source']}|{doc.metadata['chunk_index']}|{doc.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def ingest_file(path: Path, store=None) -> int:
    """Ingest one file. The file's old chunks are deleted first,
    so ingesting the same file again creates no duplicates (idempotent)."""
    if store is None:
        from app.vectorstore import get_vectorstore

        store = get_vectorstore()
    from app.llm import get_embeddings

    text, meta = load_markdown(path)
    chunks = chunk_markdown(text, meta)
    if not chunks:
        raise ValueError(f"{path.name}: no text chunks were produced (is the file empty?)")

    store.delete(where={"source": meta["source"]})
    # We compute the embeddings ourselves (from embedding_text), so the store keeps the original text.
    vectors = get_embeddings().embed_documents([embedding_text(c) for c in chunks])
    store._collection.upsert(  # noqa: SLF001 - the LangChain wrapper does not accept precomputed embeddings
        ids=[make_id(c) for c in chunks],
        embeddings=vectors,
        documents=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


def ingest_directory(directory: str | Path | None = None, store=None) -> dict[str, int]:
    directory = Path(directory or settings.docs_dir)
    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No .md files found in {directory}. Run first: python -m scripts.fetch_docs"
        )
    return {f.name: ingest_file(f, store) for f in files}


def list_documents(store=None) -> list[dict]:
    """Which documents are in the store (grouped by source)."""
    if store is None:
        from app.vectorstore import get_vectorstore

        store = get_vectorstore()

    data = store.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for meta in data["metadatas"] or []:
        entry = docs.setdefault(
            meta["source"],
            {"source": meta["source"], "title": meta.get("title", ""), "url": meta.get("url", ""), "chunks": 0},
        )
        entry["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["source"])


if __name__ == "__main__":
    print(f"Chunking: size={settings.chunk_size}, overlap={settings.chunk_overlap}, min={settings.min_chunk_chars}")
    print("Loading the embedding model (the first run takes a little longer)...\n")
    result = ingest_directory()
    for name, n in result.items():
        print(f"  {name:40s} {n:3d} chunks")
    print(f"\nTotal: {sum(result.values())} chunks, {len(result)} documents -> {settings.chroma_dir}/")