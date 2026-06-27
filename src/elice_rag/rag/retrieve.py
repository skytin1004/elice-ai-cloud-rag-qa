from __future__ import annotations

from elice_rag.config import Settings, get_settings
from elice_rag.providers import create_embedding_client

from .index import JsonVectorIndex
from .schemas import RetrievedChunk


def retrieve(question: str, *, top_k: int | None = None, settings: Settings | None = None) -> list[RetrievedChunk]:
    settings = settings or get_settings()
    embedding_client = create_embedding_client(settings)
    index = JsonVectorIndex.load(settings.index_path)
    return index.search(question, embedding_client, top_k=top_k or settings.rag_top_k)

