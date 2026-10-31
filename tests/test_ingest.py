"""Chunking + ingestion (pure logic, embeddings are fake)."""
import pytest

import app.llm as llm_module
from app.ingest import chunk_markdown, embedding_text, ingest_file, load_markdown, make_id
from langchain_core.documents import Document

META = {"source": "t.md", "title": "T", "url": ""}
PROSE = "word " * 70  # ~350 chars


def test_load_markdown_reads_source_comment_and_title(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text("<!-- source: https://example.com/x -->\n# My Title\n\nBody text.\n", encoding="utf-8")
    text, meta = load_markdown(f)
    assert text.startswith("# My Title")
    assert meta == {"source": "guide.md", "title": "My Title", "url": "https://example.com/x"}


def test_load_markdown_without_comment_or_heading(tmp_path):
    f = tmp_path / "plain.md"
    f.write_text("just text, no heading", encoding="utf-8")
    _, meta = load_markdown(f)
    assert meta["url"] == "" and meta["title"] == "plain"


def test_chunks_carry_heading_breadcrumb():
    text = f"# T\n\n## A\n\n{PROSE}\n\n## B\n\n{PROSE}\n"
    chunks = chunk_markdown(text, dict(META))
    assert [c.metadata["section"] for c in chunks] == ["T > A", "T > B"]
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
    assert all(c.metadata["source"] == "t.md" for c in chunks)


def test_small_sibling_sections_are_merged():
    text = (
        "# T\n\n## A\n\nShort text about A that is more than forty characters long.\n\n"
        "## B\n\nShort text about B that is also more than forty characters long.\n"
    )
    chunks = chunk_markdown(text, dict(META))
    assert len(chunks) == 1
    assert "about A" in chunks[0].page_content and "about B" in chunks[0].page_content


def test_long_code_block_is_split_into_valid_fenced_chunks():
    code = "\n".join(f"value_{i} = {i}" for i in range(150))
    text = f"# T\n\n## Code\n\n```python\n{code}\n```\n"
    chunks = chunk_markdown(text, dict(META))
    assert len(chunks) >= 2
    assert all(c.page_content.count("```") % 2 == 0 for c in chunks)


def test_chunk_size_is_bounded():
    paragraphs = "\n\n".join(f"Paragraph {i}. " + "sentence goes here. " * 15 for i in range(30))
    chunks = chunk_markdown(f"# T\n\n## Long\n\n{paragraphs}\n", dict(META))
    assert len(chunks) > 3
    assert max(len(c.page_content) for c in chunks) <= 1200  # 1.5 x chunk_size


def test_embedding_text_drops_code_but_keeps_breadcrumb():
    d = Document(
        page_content=f"## A\n\n{PROSE}\n\n```python\nprint('hi')\n```",
        metadata={"section": "T > A"},
    )
    out = embedding_text(d)
    assert out.startswith("T > A\n") and "```" not in out and "word" in out


def test_embedding_text_keeps_code_when_almost_no_prose():
    d = Document(page_content="```python\nprint('hi')\n```", metadata={"section": "T > A"})
    assert "print('hi')" in embedding_text(d)


def test_make_id_is_deterministic_and_content_sensitive():
    a = Document(page_content="x", metadata={"source": "s.md", "chunk_index": 0})
    b = Document(page_content="y", metadata={"source": "s.md", "chunk_index": 0})
    assert make_id(a) == make_id(a)
    assert make_id(a) != make_id(b)


# ------------------------------------------------------------------ ingest_file


class FakeCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


class FakeStore:
    def __init__(self):
        self.deleted = []
        self._collection = FakeCollection()

    def delete(self, where):
        self.deleted.append(where)


@pytest.fixture
def embeddings(monkeypatch):
    embedded = []

    class FakeEmb:
        def embed_documents(self, texts):
            embedded.extend(texts)
            return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(llm_module, "get_embeddings", lambda: FakeEmb())
    return embedded


def test_ingest_file_stores_original_text_but_embeds_clean_text(tmp_path, embeddings):
    f = tmp_path / "guide.md"
    f.write_text(f"# T\n\n## A\n\n{PROSE}\n\n```python\nprint('hi')\n```\n", encoding="utf-8")
    store = FakeStore()

    n = ingest_file(f, store)

    up = store._collection.upserts[0]
    assert n == len(up["ids"]) == len(up["documents"]) == len(up["embeddings"])
    assert any("```" in d for d in up["documents"])  # the store keeps the original text (with code)
    assert all("```" not in e for e in embeddings)  # code is removed from the text that gets embedded
    assert store.deleted == [{"source": "guide.md"}]


def test_ingest_is_idempotent(tmp_path, embeddings):
    f = tmp_path / "guide.md"
    f.write_text(f"# T\n\n## A\n\n{PROSE}\n", encoding="utf-8")
    store = FakeStore()
    ingest_file(f, store)
    ingest_file(f, store)
    first, second = store._collection.upserts
    assert first["ids"] == second["ids"]  # same ids -> upsert creates no duplicates
    assert len(store.deleted) == 2  # old chunks are deleted first on every run


def test_ingest_empty_file_raises(tmp_path, embeddings):
    f = tmp_path / "empty.md"
    f.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        ingest_file(f, FakeStore())