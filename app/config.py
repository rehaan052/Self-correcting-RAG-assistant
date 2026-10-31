from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings in one place. Values come from the .env file."""

    # LLM + embeddings
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"
    reasoning_effort: str = "low"  # only for gpt-oss models: low = fast and cheap
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    chroma_dir: str = "chroma_db"
    collection_name: str = "tech_docs"
    docs_dir: str = "data/docs"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_chars: int = 300  # chunks shorter than this are merged into the next related chunk

    # Retrieval / graph
    top_k: int = 8  # the grader sees all chunks in one call, so 5 -> 8 is cheap
    hybrid_search: bool = True  # set False for dense-only (ablation / comparison)
    candidate_k: int = 20  # how many candidates from dense and from BM25 are fused
    rrf_k: int = 60  # the standard Reciprocal Rank Fusion constant
    max_retries: int = 2  # maximum query rewrites
    max_generation_attempts: int = 2  # maximum regenerations after the hallucination check

    # Bonus features
    tavily_api_key: str = ""
    enable_web_search: bool = True
    enable_hallucination_check: bool = True
    max_history_turns: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()