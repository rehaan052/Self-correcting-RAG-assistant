"""Request/response shapes of the API. Validation happens here (FastAPI returns 422 on bad input by itself)."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64, description="Send the previous session_id for follow-up questions")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class Source(BaseModel):
    id: int
    source: str
    section: str = ""
    url: str = ""
    score: float | None = None
    snippet: str = ""


class QueryResponse(BaseModel):
    query_id: str
    session_id: str
    answer: str
    status: str  # answered | answered_from_web | unverified | no_answer
    grounded: bool | None
    retries: int
    used_web_search: bool
    sources: list[Source]
    trace: list[str]


class IngestResponse(BaseModel):
    source: str
    chunks: int


class DocumentInfo(BaseModel):
    source: str
    title: str
    url: str
    chunks: int


class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    id: int
    status: str = "saved"