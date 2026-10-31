"""SQLite: query log + feedback. Small and dependency-free (stdlib sqlite3)."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

DB_PATH = "feedback.db"


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with closing(_conn()) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL REFERENCES queries(id),
                rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                comment TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_query(query_id: str, session_id: str, question: str, answer: str, status: str) -> None:
    with closing(_conn()) as conn, conn:
        conn.execute(
            "INSERT INTO queries (id, session_id, question, answer, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (query_id, session_id, question, answer, status, _now()),
        )


def query_exists(query_id: str) -> bool:
    with closing(_conn()) as conn:
        return conn.execute("SELECT 1 FROM queries WHERE id = ?", (query_id,)).fetchone() is not None


def save_feedback(query_id: str, rating: str, comment: str | None) -> int:
    with closing(_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO feedback (query_id, rating, comment, created_at) VALUES (?, ?, ?, ?)",
            (query_id, rating, comment, _now()),
        )
        return int(cur.lastrowid)