from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache
def get_llm() -> ChatGroq:
    """temperature=0 keeps grading and answers consistent."""
    kwargs = {}
    if settings.groq_model.startswith("openai/gpt-oss"):
        # gpt-oss is a reasoning model; 'low' effort keeps grading and rewriting fast and cheap
        kwargs["reasoning_effort"] = settings.reasoning_effort
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
        **kwargs,
    )


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Local CPU embedding model (~90 MB). Downloaded on the first run."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )