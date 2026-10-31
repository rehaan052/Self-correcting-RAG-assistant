"""API tests (FastAPI TestClient). SQLite and the docs folder live in tmp, the LLM is fake."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import db, main
from app.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "feedback.db"))
    monkeypatch.setattr(settings, "docs_dir", str(tmp_path / "docs"))
    main._sessions.clear()
    with TestClient(main.app) as c:  # runs the lifespan -> db.init_db()
        yield c
    main._sessions.clear()


@pytest.fixture
def stub_query(monkeypatch):
    """A direct answer in place of the graph, for /query tests."""
    calls = []

    def fake_run(question, history):
        calls.append((question, list(history)))
        return {
            "answer": f"answer to: {question}",
            "status": "answered",
            "grounded": True,
            "retry_count": 1,
            "used_web_search": False,
            "sources": [{"id": 1, "source": "doc.md", "section": "S", "url": "", "score": 0.5, "snippet": "s"}],
            "trace": ["analyze_query: ok"],
        }

    monkeypatch.setattr(main, "run_query", fake_run)
    return calls


def ask(client, question, session_id=None):
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id
    return client.post("/query", json=body)


# ------------------------------------------------------------------ /query


def test_query_ok(client, stub_query):
    r = ask(client, "hello")
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "answer to: hello"
    assert body["status"] == "answered" and body["grounded"] is True
    assert body["retries"] == 1 and body["used_web_search"] is False
    assert body["sources"][0]["id"] == 1
    assert len(body["query_id"]) == 32 and len(body["session_id"]) == 32
    assert body["trace"] == ["analyze_query: ok"]


def test_each_new_query_gets_new_session_and_query_id(client, stub_query):
    a, b = ask(client, "one").json(), ask(client, "two").json()
    assert a["session_id"] != b["session_id"]
    assert a["query_id"] != b["query_id"]


def test_follow_up_receives_history(client, stub_query):
    first = ask(client, "first").json()
    ask(client, "second", first["session_id"])
    assert stub_query[0][1] == []
    assert stub_query[1][1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer to: first"},
    ]


def test_sessions_are_isolated(client, stub_query):
    ask(client, "first", "session-a")
    ask(client, "other", "session-b")
    assert stub_query[1][1] == []


def test_history_is_trimmed(client, stub_query, monkeypatch):
    monkeypatch.setattr(settings, "max_history_turns", 2)
    for i in range(5):
        ask(client, f"q{i}", "s1")
    assert len(stub_query[4][1]) == 4  # 2 turns = 4 messages


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "hi", "session_id": "s" * 65},
    ],
)
def test_query_validation_422(client, stub_query, body):
    assert client.post("/query", json=body).status_code == 422
    assert stub_query == []  # the request never reached the graph


def test_pipeline_failure_returns_502(client, monkeypatch):
    def boom(question, history):
        raise RuntimeError("groq down")

    monkeypatch.setattr(main, "run_query", boom)
    r = ask(client, "hello")
    assert r.status_code == 502
    assert "RuntimeError" in r.json()["detail"]


def test_query_end_to_end_through_real_graph(client, fake, doc):
    fake.retriever.on([doc("body", source="a.md", section="Intro")])
    body = ask(client, "real graph question").json()
    assert body["status"] == "answered" and body["grounded"] is True and body["retries"] == 0
    assert body["sources"][0]["source"] == "a.md"
    assert any(t.startswith("grade_documents") for t in body["trace"])


def test_query_end_to_end_fallback(client, fake):
    fake.llm.on("grade", fake.llm.grades(False))
    body = ask(client, "unknown topic").json()
    assert body["status"] == "no_answer" and body["retries"] == 2 and body["sources"] == []


# ------------------------------------------------------------------ /feedback


def test_feedback_saved(client, stub_query):
    qid = ask(client, "hello").json()["query_id"]
    r = client.post("/feedback", json={"query_id": qid, "rating": "up", "comment": "good"})
    assert r.status_code == 201
    assert r.json() == {"id": 1, "status": "saved"}
    with sqlite3.connect(db.DB_PATH) as conn:
        assert conn.execute("SELECT query_id, rating, comment FROM feedback").fetchall() == [(qid, "up", "good")]


def test_feedback_comment_is_optional_and_repeatable(client, stub_query):
    qid = ask(client, "hello").json()["query_id"]
    assert client.post("/feedback", json={"query_id": qid, "rating": "down"}).json()["id"] == 1
    assert client.post("/feedback", json={"query_id": qid, "rating": "up"}).json()["id"] == 2


def test_feedback_unknown_query_id_404(client):
    assert client.post("/feedback", json={"query_id": "nope", "rating": "up"}).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"query_id": "x", "rating": "meh"},
        {"query_id": "x"},
        {"rating": "up"},
        {"query_id": "x", "rating": "up", "comment": "c" * 1001},
    ],
)
def test_feedback_validation_422(client, body):
    assert client.post("/feedback", json=body).status_code == 422


# ------------------------------------------------------------------ /ingest


@pytest.fixture
def fake_ingest(monkeypatch):
    seen = []

    def fake(path):
        seen.append(path)
        return 3

    monkeypatch.setattr(main, "ingest_file", fake)
    return seen


def upload(client, name, content):
    return client.post("/ingest", files={"file": (name, content, "text/markdown")})


def test_ingest_ok(client, fake_ingest, tmp_path):
    r = upload(client, "notes.md", b"# Notes\n\nhello world")
    assert r.status_code == 201
    assert r.json() == {"source": "notes.md", "chunks": 3}
    saved = tmp_path / "docs" / "notes.md"
    assert saved.read_text() == "# Notes\n\nhello world"
    assert fake_ingest[0].name == "notes.md"


def test_ingest_txt_is_saved_as_md(client, fake_ingest):
    assert upload(client, "notes.txt", b"some text").json()["source"] == "notes.md"


def test_ingest_filename_is_sanitised(client, fake_ingest):
    source = upload(client, "my notes (v2)!.md", b"text").json()["source"]
    assert source.endswith(".md") and not any(c in source for c in " ()!")


def test_ingest_path_traversal_is_blocked(client, fake_ingest, tmp_path):
    r = upload(client, "../../evil.md", b"text")
    assert r.status_code == 201 and r.json()["source"] == "evil.md"
    assert (tmp_path / "docs" / "evil.md").exists()
    assert not (tmp_path / "evil.md").exists()


@pytest.mark.parametrize("name", ["report.pdf", "noextension"])
def test_ingest_rejects_wrong_type_400(client, fake_ingest, name):
    assert upload(client, name, b"text").status_code == 400
    assert fake_ingest == []


def test_ingest_too_large_413(client, fake_ingest):
    assert upload(client, "big.md", b"a" * 1_000_001).status_code == 413


@pytest.mark.parametrize("content", [b"\xff\xfe\xfa", b"   \n"])
def test_ingest_bad_content_400(client, fake_ingest, content):
    assert upload(client, "bad.md", content).status_code == 400


def test_ingest_pipeline_value_error_400(client, monkeypatch):
    def fail(path):
        raise ValueError("no text chunks were produced")

    monkeypatch.setattr(main, "ingest_file", fail)
    r = upload(client, "x.md", b"text")
    assert r.status_code == 400 and "chunk" in r.json()["detail"]


def test_ingest_missing_file_422(client):
    assert client.post("/ingest").status_code == 422


# ------------------------------------------------------------------ /documents, /health


def test_documents(client, monkeypatch):
    docs = [{"source": "a.md", "title": "A", "url": "", "chunks": 4}]
    monkeypatch.setattr(main, "list_documents", lambda: docs)
    r = client.get("/documents")
    assert r.status_code == 200 and r.json() == docs


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and "model" in r.json()