from __future__ import annotations

from .schemas import Citation, RetrievedChunk


def citations_from_chunks(chunks: list[RetrievedChunk], *, max_citations: int = 5) -> list[Citation]:
    seen: set[str] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        key = chunk.source_url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                source_url=chunk.source_url,
                title=chunk.title,
                heading=chunk.heading,
                chunk_id=chunk.chunk_id,
            )
        )
        if len(citations) >= max_citations:
            break
    return citations
