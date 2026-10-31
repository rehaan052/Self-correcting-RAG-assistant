"""FastAPI app. Run:  python -m uvicorn app.main:app"""
import logging
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import db
from app.config import settings
from app.graph import run_query
from app.ingest import ingest_file, list_documents
from app.schemas import (
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger("rag")
MAX_UPLOAD_BYTES = 1_000_000  # 1 MB
ALLOWED_SUFFIXES = {".md", ".txt"}

# Session history is kept in memory (lost on restart). Use Redis or a database in production.
_sessions: dict[str, list[dict]] = {}
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="Self-Correcting RAG Assistant",
    description="LangGraph RAG: query analysis -> retrieval -> grading -> generation, with retry, fallback and grounding check.",
    version="1.0.0",
    lifespan=lifespan,
)


# These endpoints are plain `def` (not async): the graph is blocking, so FastAPI runs them in a
# thread pool and one long query does not block the others.
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    session_id = req.session_id or uuid.uuid4().hex
    with _lock:
        history = list(_sessions.get(session_id, []))

    try:
        state = run_query(req.question, history)
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_query failed")
        raise HTTPException(status_code=502, detail=f"LLM/pipeline failure: {type(exc).__name__}") from exc

    answer = state.get("answer", "")
    status = state.get("status", "answered")

    history += [{"role": "user", "content": req.question}, {"role": "assistant", "content": answer}]
    with _lock:
        _sessions[session_id] = history[-settings.max_history_turns * 2 :]

    query_id = uuid.uuid4().hex
    db.log_query(query_id, session_id, req.question, answer, status)

    return QueryResponse(
        query_id=query_id,
        session_id=session_id,
        answer=answer,
        status=status,
        grounded=state.get("grounded"),
        retries=state.get("retry_count", 0),
        used_web_search=bool(state.get("used_web_search")),
        sources=state.get("sources", []),
        trace=state.get("trace", []),
    )


@app.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest(file: UploadFile = File(...)) -> IngestResponse:
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only .md or .txt files can be uploaded")

    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 1 MB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text") from exc
    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    # Safe name (no path tricks), always stored as .md so that python -m app.ingest can pick it up again
    stem = re.sub(r"[^A-Za-z0-9._-]", "-", Path(name).stem).strip(".-") or "upload"
    target = Path(settings.docs_dir) / f"{stem}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

    try:
        chunks = ingest_file(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(source=target.name, chunks=chunks)


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[dict]:
    return list_documents()


@app.post("/feedback", response_model=FeedbackResponse, status_code=201)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    if not db.query_exists(req.query_id):
        raise HTTPException(status_code=404, detail="No query found for this query_id")
    return FeedbackResponse(id=db.save_feedback(req.query_id, req.rating, req.comment))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.groq_model, "web_search": bool(settings.enable_web_search and settings.tavily_api_key)}