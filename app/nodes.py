"""LangGraph nodes. Each node takes the state and returns only the keys it changes."""
import re
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app import prompts
from app.config import settings
from app.llm import get_llm
from app.retrieval import hybrid_search
from app.state import GraphState
from app.vectorstore import get_vectorstore

# --------------------------------------------------------------------------
# Structured-output schemas (the shapes the LLM must answer in)
# --------------------------------------------------------------------------


class QueryAnalysis(BaseModel):
    standalone_question: str = Field(description="Question rewritten to be understandable without chat history")
    query_type: Literal["how_to", "conceptual", "reference", "troubleshooting"]
    search_query: str = Field(description="Short keyword-rich query for semantic search")


class ChunkGrade(BaseModel):
    id: int = Field(description="Chunk number as shown in the prompt, starting at 1")
    reason: str = Field(description="One short sentence: what useful information the chunk has, or why it does not help")
    relevant: bool


class GradeResult(BaseModel):
    grades: list[ChunkGrade]


class GroundingResult(BaseModel):
    grounded: bool = Field(description="True only if every claim in the answer is supported by the context")
    unsupported_claims: list[str]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_METHODS = ["json_schema", "function_calling"]  # whichever method works gets moved to the front


def _structured(schema: type[BaseModel], messages: list):
    """Get structured output. Try json_schema first, then function_calling.
    This keeps the pipeline working when the model or provider changes."""
    last_error: Exception | None = None
    for method in list(_METHODS):
        try:
            result = get_llm().with_structured_output(schema, method=method).invoke(messages)
            if result is None:
                raise ValueError("the model did not return output matching the schema")
            if method != _METHODS[0]:
                _METHODS.remove(method)
                _METHODS.insert(0, method)
            return result
        except Exception as exc:  # noqa: BLE001 - providers raise many different error types
            last_error = exc
    assert last_error is not None
    raise last_error


def _err(exc: Exception) -> str:
    """Short error for the trace: exception type plus the start of the message."""
    return f"{type(exc).__name__}: {str(exc)[:100]}"


def _text(message) -> str:
    content = message.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content.strip()


def _format_history(history: list[dict]) -> str:
    lines = []
    for m in history:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {str(m.get('content', ''))[:500]}")
    return "\n".join(lines)


def _label(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    section = doc.metadata.get("section")
    return f"{source} | {section}" if section else source


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] ({_label(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))


def _build_sources(answer: str, docs: list[Document]) -> list[dict]:
    """Return only the sources cited in the answer as [n] (all of them if none are cited).
    Some models write 【n†...】 style citations; those are understood too."""
    nums = {int(a or b) for a, b in re.findall(r"\[(\d+)\]|【(\d+)", answer)}
    cited = sorted(n for n in nums if 1 <= n <= len(docs))
    indexes = cited or list(range(1, len(docs) + 1))
    sources = []
    for i in indexes:
        d = docs[i - 1]
        sources.append(
            {
                "id": i,
                "source": d.metadata.get("source", "unknown"),
                "section": d.metadata.get("section", ""),
                "url": d.metadata.get("url", ""),
                "score": d.metadata.get("score"),
                "snippet": d.page_content[:200],
            }
        )
    return sources


def web_search_available() -> bool:
    return bool(settings.enable_web_search and settings.tavily_api_key)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def analyze_query(state: GraphState) -> dict:
    """Make a follow-up standalone, classify the question type, and build a search query."""
    question = state["question"]
    history = state.get("chat_history") or []
    human = f"Chat history:\n{_format_history(history) or '(none)'}\n\nLatest question: {question}"
    try:
        r = _structured(QueryAnalysis, [SystemMessage(prompts.ANALYZE_SYSTEM), HumanMessage(human)])
        standalone = r.standalone_question.strip() or question
        search_query = r.search_query.strip() or standalone
        qtype, note = r.query_type, ""
    except Exception as exc:  # noqa: BLE001
        standalone, search_query, qtype = question, question, "conceptual"
        note = f" (analysis failed: {_err(exc)}; using raw question)"
    return {
        "standalone_question": standalone,
        "search_query": search_query,
        "query_type": qtype,
        "trace": [f"analyze_query: type={qtype}, search_query='{search_query}'{note}"],
    }


def retrieve(state: GraphState) -> dict:
    """Fetch chunks. Default: hybrid (dense + BM25, fused with RRF), using BOTH the search_query
    and the standalone question. Why: the LLM-written search_query varies from run to run, while the
    original question contains the real words used in the docs.
    With HYBRID_SEARCH=false it falls back to dense-only. 'score' = cosine similarity."""
    query = state["search_query"]
    queries = list(dict.fromkeys([query, state.get("standalone_question") or query]))
    if settings.hybrid_search:
        docs, mode = hybrid_search(queries, k=settings.top_k), "hybrid"
    else:
        results = get_vectorstore().similarity_search_with_score(query, k=settings.top_k)
        docs, mode = [], "dense"
        for doc, distance in results:
            doc.metadata = {**doc.metadata, "score": round(1 - float(distance), 3)}
            docs.append(doc)
    best = max((d.metadata["score"] for d in docs), default=None)
    shown = " | ".join(f"'{q}'" for q in queries)
    return {"documents": docs, "trace": [f"retrieve[{mode}]: {shown} -> {len(docs)} chunks (best cosine {best})"]}


def grade_documents(state: GraphState) -> dict:
    """LLM judge: is each chunk relevant? All chunks go in ONE call (cheaper, rate-limit friendly).
    Every chunk's verdict and reason goes into the trace, so a wrong rejection is visible."""
    docs = state.get("documents", [])
    if not docs:
        return {"relevant_documents": [], "trace": ["grade_documents: nothing retrieved"]}

    chunks = "\n\n".join(f"[{i}] ({_label(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))
    human = f"Question: {state['standalone_question']}\n\nChunks:\n{chunks}"
    lines: list[str] = []
    try:
        result = _structured(GradeResult, [SystemMessage(prompts.GRADE_SYSTEM), HumanMessage(human)])
        grades = result.grades
        # Some models count from 0 (0..n-1) although the prompt numbers chunks 1..n: shift them back
        if grades and any(g.id == 0 for g in grades) and not any(g.id == len(docs) for g in grades):
            grades = [g.model_copy(update={"id": g.id + 1}) for g in grades]
        verdict = {g.id: g for g in grades}
        relevant = [d for i, d in enumerate(docs, 1) if i in verdict and verdict[i].relevant]
        for i, d in enumerate(docs, 1):
            g = verdict.get(i)
            mark = "KEEP" if g and g.relevant else "DROP"
            reason = g.reason.strip() if g else "(the grader did not grade this chunk)"
            lines.append(f"    chunk {i} {mark}  {_label(d)} (score {d.metadata.get('score')}): {reason[:140]}")
        note = ""
    except Exception as exc:  # noqa: BLE001
        # Fail-open: if the grader breaks, keep every chunk so the pipeline does not stop
        relevant = docs
        note = f" (grader failed: {_err(exc)}; keeping all)"
    summary = f"grade_documents: {len(relevant)}/{len(docs)} relevant{note}"
    return {"relevant_documents": relevant, "trace": [summary, *lines]}


def rewrite_query(state: GraphState) -> dict:
    """Retrieval failed: write the query again. retry_count is incremented here."""
    seen = sorted({_label(d) for d in state.get("documents", [])})[:5]
    human = (
        f"Original question: {state['standalone_question']}\n"
        f"Previous search query (failed): {state['search_query']}\n"
        f"It returned chunks from: {', '.join(seen) or 'nothing'}"
    )
    try:
        new_query = _text(get_llm().invoke([SystemMessage(prompts.REWRITE_SYSTEM), HumanMessage(human)]))
        new_query = new_query.strip().strip('"').splitlines()[0] if new_query.strip() else ""
    except Exception:  # noqa: BLE001
        new_query = ""
    new_query = new_query or state["standalone_question"]
    retry = state.get("retry_count", 0) + 1
    return {
        "search_query": new_query,
        "retry_count": retry,
        "trace": [f"rewrite_query: retry {retry}/{state.get('max_retries', settings.max_retries)} -> '{new_query}'"],
    }


def web_search(state: GraphState) -> dict:
    """Bonus: nothing useful in the vector store, so search the web with Tavily."""
    try:
        from tavily import TavilyClient

        res = TavilyClient(api_key=settings.tavily_api_key).search(
            query=state["standalone_question"], max_results=4, search_depth="basic"
        )
        docs = [
            Document(
                page_content=r["content"],
                metadata={"source": "web", "section": r.get("title", ""), "url": r.get("url", "")},
            )
            for r in res.get("results", [])
            if r.get("content")
        ]
        note = ""
    except Exception as exc:  # noqa: BLE001
        docs, note = [], f" (failed: {_err(exc)})"
    return {
        "relevant_documents": docs,
        "used_web_search": True,
        "trace": [f"web_search: {len(docs)} results{note}"],
    }


def generate(state: GraphState) -> dict:
    """Write an answer with citations, using only the relevant context."""
    docs = state["relevant_documents"]
    hint = prompts.STYLE_HINTS.get(state.get("query_type", ""), "")
    human = f"Question: {state['standalone_question']}\nStyle: {hint}\n\nContext:\n{_format_context(docs)}"
    feedback = state.get("grounding_feedback")
    if feedback:
        human += (
            f"\n\nA fact-checker found these claims in your previous draft NOT supported by the context: {feedback}\n"
            "Rewrite the answer. Remove or fix those claims and use only the context."
        )
    answer = _text(get_llm().invoke([SystemMessage(prompts.GENERATE_SYSTEM), HumanMessage(human)]))
    # gpt-oss sometimes writes citations like 【2†L1-L7】 or 【2】: normalise them to [2]
    answer = re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", answer)
    attempts = state.get("generation_attempts", 0) + 1
    return {
        "answer": answer,
        "sources": _build_sources(answer, docs),
        "generation_attempts": attempts,
        "trace": [f"generate: attempt {attempts}, {len(answer)} chars, from {len(docs)} chunks"],
    }


def check_grounding(state: GraphState) -> dict:
    """Bonus (Self-RAG style): are all claims in the answer supported by the context?"""
    human = f"Context:\n{_format_context(state['relevant_documents'])}\n\nAnswer:\n{state['answer']}"
    try:
        r = _structured(GroundingResult, [SystemMessage(prompts.GROUNDING_SYSTEM), HumanMessage(human)])
        grounded, feedback, note = r.grounded, "; ".join(r.unsupported_claims), ""
    except Exception as exc:  # noqa: BLE001
        grounded, feedback, note = True, "", f" (check failed: {_err(exc)}; skipped)"
    detail = f", unsupported: {feedback}" if feedback else ""
    return {
        "grounded": grounded,
        "grounding_feedback": feedback,
        "trace": [f"check_grounding: grounded={grounded}{detail}{note}"],
    }


def fallback(state: GraphState) -> dict:
    """Nothing relevant was found and retries are used up: say so plainly instead of guessing."""
    return {
        "answer": prompts.FALLBACK_ANSWER,
        "sources": [],
        "trace": ["fallback: no relevant context found -> 'I don't know'"],
    }