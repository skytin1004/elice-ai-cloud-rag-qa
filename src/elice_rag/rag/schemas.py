from __future__ import annotations

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    id: str
    title: str
    url: str


class RawDocument(SourceDocument):
    fetched_at: str
    html: str


class CleanDocument(SourceDocument):
    text: str
    headings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    source_url: str
    title: str
    heading: str
    text: str
    chunk_index: int


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    source_url: str
    title: str
    heading: str
    text: str
    score: float


class Citation(BaseModel):
    source_url: str
    title: str
    heading: str
    chunk_id: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    model: str | None = Field(default=None, min_length=1)


class QueryResponse(BaseModel):
    status: str
    answer: str
    citations: list[Citation]
    confidence: str
    retrieved_context: list[RetrievedChunk]
    model: str | None = None
