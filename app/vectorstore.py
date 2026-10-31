from functools import lru_cache

from langchain_chroma import Chroma

from app.config import settings
from app.llm import get_embeddings


@lru_cache
def get_vectorstore() -> Chroma:
    """Persistent ChromaDB. Cosine distance, kyunki embeddings normalized hain."""
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )