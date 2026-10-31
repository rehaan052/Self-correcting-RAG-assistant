"""Margin: Streamlit UI for the self-correcting RAG assistant.
"""
import base64
import os
import random
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Margin", page_icon="📖", layout="wide")


def load_secrets_into_env() -> None:
    """Streamlit Cloud keeps keys in its "Secrets" box. The backend reads environment variables,
    so copy them over before the backend is imported."""
    try:
        for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "GROQ_MODEL", "RAG_MODE", "DISABLE_UPLOAD", "MAX_QUERIES"):
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except Exception:  # noqa: BLE001 - no secrets file when running locally: that is fine
        pass


load_secrets_into_env()

API = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
DEMO = os.environ.get("DEMO") == "1"  # DEMO=1 uses built-in fake data instead of the API
INPROCESS = os.environ.get("RAG_MODE") == "inprocess"  # run the FastAPI app inside this process (Streamlit Cloud)
UPLOAD_DISABLED = os.environ.get("DISABLE_UPLOAD") == "1"  # public demo: nobody can add documents
MAX_QUERIES = int(os.environ.get("MAX_QUERIES") or 0)  # questions allowed per browser session, 0 = unlimited

CHAT, DOCS = "💬  Chat", "📁  Documents"

EXAMPLES = [
    "How do I run background tasks in FastAPI?",
    "How do I declare a path parameter?",
    "How many annual leave days do employees get?",
    "What is the learning budget per employee?",
]

STAGES = [
    ("Understanding your question", "query analysis"),
    ("Searching the documents", "hybrid retrieval"),
    ("Checking the chunks", "relevance grading"),
    ("Writing the answer", "generation"),
    ("Verifying the claims", "grounding check"),
]

STATUS = {
    "answered": ("Answered", "ok"),
    "answered_from_web": ("Answered from web", "web"),
    "unverified": ("Unverified", "warn"),
    "no_answer": ("No answer in docs", "bad"),
}

# ------------------------------------------------------------------ styling

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root{
  --bg:#0F1A2B; --panel:#15233A; --line:#243653; --ink:#E9EEF7; --mute:#8FA2BF;
  --hl:#F2B441; --ok:#3FD0C9; --web:#7FB2FF; --bad:#FF7A7A;
}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea{
  font-family:'IBM Plex Sans', system-ui, sans-serif;
}
.stApp{ background:var(--bg); }
[data-testid="stSidebar"]{ background:#0B1423; border-right:1px solid var(--line); }
[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2rem; max-width:980px; }

/* logo */
.brand{ display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.1rem; }
.brand svg{ flex:none; }
.brand .name{ font-family:'Fraunces', Georgia, serif; font-weight:700; font-size:1.55rem; line-height:1; color:var(--ink); }
.brand .tag{ font-size:.78rem; color:var(--mute); margin-top:.2rem; }

/* company logo */
.ea-card{ background:#FFFFFF; border-radius:10px; padding:.7rem .9rem; margin:.2rem 0 1rem; display:flex; align-items:center; justify-content:center; }
.ea-card img{ max-width:100%; max-height:42px; display:block; }
.ea-card span{ color:#0F1A2B; font-weight:700; letter-spacing:.01em; }

/* status + chips */
.pill{ display:inline-flex; align-items:center; gap:.4rem; padding:.18rem .6rem; border-radius:999px;
  font-size:.78rem; font-weight:600; border:1px solid currentColor; }
.pill.ok{ color:var(--ok); } .pill.web{ color:var(--web); }
.pill.warn{ color:var(--hl); } .pill.bad{ color:var(--bad); }
.dot{ width:.5rem; height:.5rem; border-radius:50%; background:currentColor; display:inline-block; }
.pill.ok .dot{ animation:beat 1.8s ease-in-out infinite; }
.chips{ display:flex; flex-wrap:wrap; gap:.45rem; margin:.7rem 0 .4rem; }
.chip{ font-size:.78rem; color:var(--mute); background:var(--panel); border:1px solid var(--line);
  padding:.18rem .55rem; border-radius:6px; }

/* page header */
.topbar h1{ font-family:'Fraunces', Georgia, serif; font-size:2rem; margin:0; color:var(--ink); }
.topbar p{ color:var(--mute); margin:.25rem 0 1.2rem; font-size:.92rem; }

/* hero (empty chat) */
.hero{ padding:2.2rem 0 1rem; }
.hero h2{ font-family:'Fraunces', Georgia, serif; font-size:2.6rem; line-height:1.1; margin:0 0 .7rem; color:var(--ink); }
.hero h2 mark{ background:linear-gradient(transparent 58%, rgba(242,180,65,.75) 58%); color:inherit; padding:0 .1em;
  background-size:0% 100%; background-repeat:no-repeat; animation:marker 1.1s .3s ease-out forwards; }
.hero p{ color:var(--mute); max-width:34rem; font-size:1.02rem; }

/* chat bubbles */
[data-testid="stChatMessage"]{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:1rem 1.1rem; animation:rise .3s ease both; }

/* metrics */
[data-testid="stMetric"]{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:.9rem 1rem; }
[data-testid="stMetricValue"]{ font-family:'Fraunces', Georgia, serif; }

/* sidebar buttons / nav */
[data-testid="stSidebar"] [role="radiogroup"] label{ padding:.4rem .6rem; border-radius:8px; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){ background:var(--panel); border:1px solid var(--line); }
.sec{ color:var(--mute); font-size:.8rem; margin:1.2rem 0 .4rem; }

/* thinking animation */
.txt .head{ color:var(--ink); font-weight:600; }
.txt small{ color:var(--mute); font-variant-numeric:tabular-nums; }

/* equalizer */
.wave{ display:flex; align-items:center; gap:1.2rem; padding:.9rem .4rem; }
.wave .bars{ display:flex; align-items:center; gap:4px; height:44px; }
.wave .bars i{ width:5px; height:100%; background:var(--hl); border-radius:3px; animation:eq 1s ease-in-out var(--d) infinite; }

@keyframes eq{ 0%,100%{transform:scaleY(.2)} 50%{transform:scaleY(1)} }

@keyframes rise{ from{opacity:0; transform:translateY(6px)} to{opacity:1; transform:none} }
@keyframes marker{ to{ background-size:100% 100%; } }
@keyframes beat{ 0%,100%{opacity:.35} 50%{opacity:1} }
@media (prefers-reduced-motion: reduce){
  *{ animation:none !important; }
  .hero h2 mark{ background-size:100% 100%; }
}
</style>
""",
    unsafe_allow_html=True,
)

LOGO = """
<div class="brand">
  <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
    <rect x="6" y="3" width="25" height="33" rx="3" fill="#F4EFE3"/>
    <rect x="11" y="10" width="15" height="3" rx="1.5" fill="#B9B2A0"/>
    <rect x="9" y="17" width="19" height="6" rx="1" fill="#F2B441"/>
    <rect x="11" y="18.5" width="15" height="3" rx="1.5" fill="#6B5A2A"/>
    <rect x="11" y="27" width="11" height="3" rx="1.5" fill="#B9B2A0"/>
    <circle cx="31" cy="30" r="6" fill="#0B1423" stroke="#3FD0C9" stroke-width="2"/>
    <path d="M28.6 30l1.8 1.8 3.2-3.4" stroke="#3FD0C9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <div><div class="name">Margin</div><div class="tag">Answers with receipts</div></div>
</div>
"""

# ------------------------------------------------------------------ state

ss = st.session_state
ss.setdefault("messages", [])  # {"role", "content", "meta"}
ss.setdefault("session_id", None)
ss.setdefault("archive", [])  # earlier chats: {"title", "messages", "session_id"}
ss.setdefault("view", CHAT)
ss.setdefault("demo_docs", [
    {"source": "fastapi-first-steps.md", "title": "First Steps", "url": "", "chunks": 14},
    {"source": "novacart_handbook.md", "title": "NovaCart Employee Handbook", "url": "", "chunks": 22},
])


# ------------------------------------------------------------------ API helpers


def demo_api(path: str, kwargs: dict):
    """DEMO=1 mode: fake responses. /query runs in a worker thread, so session_state is not touched for it."""
    if path == "/health":
        return {"status": "ok", "model": "demo mode"}, None
    if path == "/documents":
        return list(ss.demo_docs), None
    if path == "/ingest":
        name, data = kwargs["files"]["file"]
        source = name.rsplit(".", 1)[0] + ".md"
        chunks = max(1, len(data) // 800)
        time.sleep(1.2)
        ss.demo_docs.append({"source": source, "title": source, "url": "", "chunks": chunks})
        return {"source": source, "chunks": chunks}, None
    if path == "/feedback":
        return {"id": 1, "status": "saved"}, None
    payload = kwargs["json"]
    time.sleep(random.uniform(4, 8))  # long enough to see the thinking animation
    status = random.choice(["answered", "answered", "answered", "answered_from_web", "unverified", "no_answer"])
    retries = random.choice([0, 0, 1, 2])
    ok = status != "no_answer"
    answer = (
        f"Demo answer: you asked **{payload['question']}**. This is a fake answer, only to show the UI [1].\n\n"
        "- First point [1]\n- Second point [2]"
        if ok
        else "I don't know. I could not find the answer in the documents."
    )
    sources = (
        [
            {"id": 1, "source": "fastapi-first-steps.md", "section": "First Steps > Recap", "url": "", "score": 0.61,
             "snippet": "Create a FastAPI instance and define a path operation function."},
            {"id": 2, "source": "novacart_handbook.md", "section": "Leave > Annual Leave", "url": "https://example.com",
             "score": 0.52, "snippet": "Every full-time employee gets 24 paid leave days per year."},
        ]
        if ok
        else []
    )
    return {
        "answer": answer,
        "status": status,
        "grounded": None if status == "answered_from_web" else status != "unverified",
        "retries": retries,
        "used_web_search": status == "answered_from_web",
        "sources": sources,
        "trace": ["analyze_query: ok", "retrieve[hybrid]: 8 chunks", f"grade_documents: {3 if ok else 0}/8 relevant",
                  "generate: done" if ok else "fallback: no relevant context found"],
        "query_id": uuid.uuid4().hex,
        "session_id": payload.get("session_id") or uuid.uuid4().hex,
    }, None


@st.cache_resource(show_spinner="Setting up the knowledge base (first start takes about a minute)...")
def get_inprocess_client():
    """Run the real FastAPI app inside this process through an in-memory ASGI client (no network).
    The UI then goes through the same validation, sessions and feedback code as the HTTP API."""
    from fastapi.testclient import TestClient

    from app import db
    from app.config import settings
    from app.ingest import ingest_file, list_documents
    from app.main import app as fastapi_app

    db.init_db()  # the lifespan hook does not run without a server, so do it here
    if not list_documents():  # fresh container: the vector store starts empty
        for path in sorted(Path(settings.docs_dir).glob("*.md")):
            ingest_file(path)
    return TestClient(fastapi_app, raise_server_exceptions=False)


# Created in the main thread so that worker threads (see ask()) never need Streamlit.
_CLIENT = get_inprocess_client() if INPROCESS and not DEMO else None


def call_api(method: str, path: str, timeout: int = 120, **kwargs):
    """No Streamlit calls, so it is safe inside a thread. Returns (data, error)."""
    if DEMO:
        return demo_api(path, kwargs)
    if _CLIENT is not None:
        try:
            r = _CLIENT.request(method, path, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return None, f"The backend failed ({type(exc).__name__})"
    else:
        try:
            r = requests.request(method, f"{API}{path}", timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            return None, f"Could not connect to the API ({API}). Is the server running? ({type(exc).__name__})"
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"{r.status_code}: {detail}"
    return r.json(), None


def api(method: str, path: str, **kwargs):
    data, err = call_api(method, path, **kwargs)
    if err:
        st.error(err)
    return data


@st.cache_data(ttl=10, show_spinner=False)
def get_health():
    data, _ = call_api("GET", "/health", timeout=3)
    return data


@st.cache_data(ttl=5, show_spinner=False)
def get_docs():
    data, _ = call_api("GET", "/documents", timeout=10)
    return data or []


# ------------------------------------------------------------------ company logo

LOGO_DIR = Path(__file__).parent / "assets"
LOGO_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".svg": "image/svg+xml"}


def company_logo_html() -> str:
    """Shows assets/express_analytics_logo.(png|svg|jpg|webp) if present, otherwise a text wordmark."""
    for ext, mime in LOGO_MIME.items():
        f = LOGO_DIR / f"express_analytics_logo{ext}"
        if f.exists():
            b64 = base64.b64encode(f.read_bytes()).decode()
            return f'<div class="ea-card"><img src="data:{mime};base64,{b64}" alt="Express Analytics"></div>'
    return '<div class="ea-card"><span>Express Analytics</span></div>'


# ------------------------------------------------------------------ chat helpers


def thinking_html(active: int, elapsed: float) -> str:
    title, sub = STAGES[active]
    bars = "".join(f'<i style="--d:{(i * 0.11) % 0.9:.2f}s"></i>' for i in range(16))
    return (
        f'<div class="wave"><div class="bars">{bars}</div>'
        f'<div class="txt"><div class="head">{title}</div><small>{sub} · {elapsed:.1f}s</small></div></div>'
    )


def ask(question: str):
    """The API call runs in a background thread while the thinking animation plays."""
    box = st.empty()
    payload = {"question": question, "session_id": ss.session_id}
    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call_api, "POST", "/query", json=payload)
        while not future.done():
            elapsed = time.time() - start
            stage = min(int(elapsed / 1.6), len(STAGES) - 1)  # estimate: the API is one call and reports no per-stage progress
            box.markdown(thinking_html(stage, elapsed), unsafe_allow_html=True)
            time.sleep(0.2)
        res, err = future.result()
    box.empty()
    return res, err, time.time() - start


def typewriter(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


def render_meta(meta: dict, idx: int):
    label, cls = STATUS.get(meta["status"], (meta["status"], "warn"))
    chips = [
        f'<span class="pill {cls}"><span class="dot"></span>{label}</span>',
        f'<span class="chip">retries {meta["retries"]}</span>',
        f'<span class="chip">grounded {meta["grounded"]}</span>',
    ]
    if meta.get("elapsed") is not None:
        chips.append(f'<span class="chip">{meta["elapsed"]:.1f}s</span>')
    if meta.get("used_web_search"):
        chips.append('<span class="chip">web search used</span>')
    st.markdown('<div class="chips">' + "".join(chips) + "</div>", unsafe_allow_html=True)

    if meta["sources"]:
        with st.expander(f"Sources ({len(meta['sources'])})"):
            for s in meta["sources"]:
                link = f" · [link]({s['url']})" if s.get("url") else ""
                st.markdown(f"**[{s['id']}] {s['source']}** — {s['section']}{link}")
                st.code(s["snippet"], language=None)
    with st.expander("Trace: what the graph did"):
        st.code("\n".join(meta["trace"]), language=None)

    c1, c2, _ = st.columns([1, 1, 8])
    for col, rating, icon in ((c1, "up", "👍"), (c2, "down", "👎")):
        if col.button(icon, key=f"fb-{rating}-{idx}"):
            if api("POST", "/feedback", json={"query_id": meta["query_id"], "rating": rating}):
                st.toast("Feedback saved")


# ------------------------------------------------------------------ callbacks


def archive_current():
    if ss.messages:
        first = next((m["content"] for m in ss.messages if m["role"] == "user"), "Chat")
        ss.archive.insert(0, {"title": first[:38], "messages": ss.messages, "session_id": ss.session_id})


def new_chat():
    archive_current()
    ss.messages, ss.session_id, ss.view = [], None, CHAT


def load_chat(i: int):
    chosen = ss.archive.pop(i)
    archive_current()
    ss.messages, ss.session_id, ss.view = chosen["messages"], chosen["session_id"], CHAT


def queue_example(text: str):
    ss.queued = text


# ------------------------------------------------------------------ views


def chat_view():
    question = st.chat_input("Ask something about your documents") or ss.pop("queued", None)

    st.markdown(
        '<div class="topbar"><h1>Self-correcting RAG</h1>'
        "<p>Understand → retrieve → grade → write → verify. Answers come only from your documents.</p></div>",
        unsafe_allow_html=True,
    )

    if not ss.messages and not question:
        st.markdown(
            '<div class="hero"><h2>Ask anything. Every answer comes with <mark>proof</mark>.</h2>'
            "<p>Ingest your documents, then ask away. Each answer shows its sources and the full trace, "
            "and when the documents do not cover it, the assistant says so plainly.</p></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, text in enumerate(EXAMPLES):
            cols[i % 2].button(text, key=f"ex-{i}", on_click=queue_example, args=(text,), width="stretch")

    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                render_meta(m["meta"], i)

    if question and MAX_QUERIES and ss.get("query_count", 0) >= MAX_QUERIES:
        st.warning(f"Demo limit reached ({MAX_QUERIES} questions per session). Run the project locally for unlimited use.")
    elif question:
        ss.query_count = ss.get("query_count", 0) + 1
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            res, err, elapsed = ask(question)
            if err:
                st.error(err)
                ss.messages += [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"⚠️ {err}"},
                ]
            else:
                ss.session_id = res["session_id"]
                st.write_stream(typewriter(res["answer"]))
                meta = {**res, "elapsed": elapsed}
                ss.messages += [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": res["answer"], "meta": meta},
                ]
                render_meta(meta, len(ss.messages) - 1)


def documents_view():
    st.markdown(
        '<div class="topbar"><h1>Documents</h1><p>See what has been ingested, and add new files.</p></div>',
        unsafe_allow_html=True,
    )
    docs = get_docs()
    c = st.columns(3)
    c[0].metric("Documents", len(docs))
    c[1].metric("Chunks", sum(d["chunks"] for d in docs))
    c[2].metric("Avg chunks / doc", f"{sum(d['chunks'] for d in docs) / len(docs):.1f}" if docs else "0")

    st.markdown("&nbsp;")
    if UPLOAD_DISABLED:
        st.info("Uploads are turned off in this public demo. Clone the repo to ingest your own documents.")
    else:
        up = st.file_uploader("Ingest a new document (.md or .txt)", type=["md", "txt"])
        if up and st.button("Ingest", type="primary"):
            with st.spinner("Chunking and embedding..."):
                res = api("POST", "/ingest", files={"file": (up.name, up.getvalue())})
            if res:
                st.success(f"{res['source']}: {res['chunks']} chunks created")
                get_docs.clear()
                st.rerun()

    if docs:
        st.markdown("**Ingested**")
        st.dataframe(
            pd.DataFrame(docs).rename(columns={"source": "File", "title": "Title", "url": "URL", "chunks": "Chunks"}),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No documents ingested yet." + ("" if UPLOAD_DISABLED else " Upload a .md or .txt file above."))


# ------------------------------------------------------------------ router

{CHAT: chat_view, DOCS: documents_view}[ss.view]()

# ------------------------------------------------------------------ sidebar (rendered last so it reflects the latest state)

with st.sidebar:
    st.markdown(company_logo_html(), unsafe_allow_html=True)
    st.markdown(LOGO, unsafe_allow_html=True)

    health = get_health()
    if health:
        st.markdown(
            f'<span class="pill ok"><span class="dot"></span>Groq API online</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="pill bad"><span class="dot"></span>API offline</span>', unsafe_allow_html=True)

    st.markdown("&nbsp;")
    st.button("＋  New chat", on_click=new_chat, type="primary", width="stretch")

    st.markdown('<div class="sec">Sections</div>', unsafe_allow_html=True)
    st.radio("Section", [CHAT, DOCS], key="view", label_visibility="collapsed")

    st.markdown('<div class="sec">Recent chats</div>', unsafe_allow_html=True)
    if ss.archive:
        for i, c in enumerate(ss.archive[:8]):
            st.button(c["title"], key=f"hist-{i}", on_click=load_chat, args=(i,), width="stretch")
    else:
        st.caption("Earlier chats are saved here when you start a new one.")

    docs = get_docs()
    st.markdown('<div class="sec">Knowledge base</div>', unsafe_allow_html=True)
    st.caption(f"{len(docs)} docs · {sum(d['chunks'] for d in docs)} chunks")