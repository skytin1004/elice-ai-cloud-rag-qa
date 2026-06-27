from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from elice_rag.config import get_settings
from elice_rag.rag.generate import RAGPipeline
from elice_rag.rag.schemas import QueryRequest, QueryResponse


app = FastAPI(
    title="Elice Citation RAG QA",
    version="0.1.0",
    description="Citation-grounded RAG service with refusal behavior for insufficient context.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sources")
def sources() -> list[dict[str, str]]:
    settings = get_settings()
    if not settings.sources_path.exists():
        return []
    return json.loads(settings.sources_path.read_text(encoding="utf-8"))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    if not settings.index_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Vector index not found. Run: python -m elice_rag.rag.ingest all",
        )
    pipeline = RAGPipeline.from_settings(settings)
    return pipeline.query(request.question, top_k=request.top_k, model=request.model)


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    settings = get_settings()
    if not settings.index_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Vector index not found. Run: python -m elice_rag.rag.ingest all",
        )
    pipeline = RAGPipeline.from_settings(settings)

    def event_stream():
        for item in pipeline.stream_query_events(
            request.question,
            top_k=request.top_k,
            model=request.model,
        ):
            yield _sse(item["event"], item["data"])

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
