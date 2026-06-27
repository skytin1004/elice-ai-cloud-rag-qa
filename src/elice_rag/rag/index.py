from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from elice_rag.providers.base import EmbeddingClient

from .schemas import Chunk, RetrievedChunk


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return numerator / (a_norm * b_norm)


@dataclass
class JsonVectorIndex:
    path: Path
    chunks: list[Chunk]
    vectors: list[list[float]]

    @classmethod
    def load(cls, path: str | Path) -> "JsonVectorIndex":
        index_path = Path(path)
        data = json.loads(index_path.read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(item) for item in data["chunks"]]
        return cls(index_path, chunks, data["vectors"])

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedding_client: EmbeddingClient,
        path: str | Path,
    ) -> "JsonVectorIndex":
        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        vectors = embedding_client.embed([chunk.text for chunk in chunks])
        data = {
            "embedding_provider": embedding_client.provider_name,
            "embedding_model": embedding_client.model_name,
            "chunks": [chunk.model_dump() for chunk in chunks],
            "vectors": vectors,
        }
        index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return cls(index_path, chunks, vectors)

    def search(
        self,
        query: str,
        embedding_client: EmbeddingClient,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        query_vector = embedding_client.embed([query])[0]
        scored: list[tuple[float, Chunk]] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            scored.append((_cosine(query_vector, vector), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                source_url=chunk.source_url,
                title=chunk.title,
                heading=chunk.heading,
                text=chunk.text,
                score=round(score, 6),
            )
            for score, chunk in scored[:top_k]
        ]


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunks_path = Path(path)
    if not chunks_path.exists():
        return chunks
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk.model_validate_json(line))
    return chunks


def write_chunks(chunks: list[Chunk], path: str | Path) -> None:
    chunks_path = Path(path)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")

