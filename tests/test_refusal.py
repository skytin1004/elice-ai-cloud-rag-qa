from pathlib import Path

from elice_rag.config import Settings
from elice_rag.providers.local_embeddings import HashingEmbeddingClient
from elice_rag.rag.generate import RAGPipeline
from elice_rag.rag.index import JsonVectorIndex
from elice_rag.rag.schemas import Chunk


def test_pipeline_refuses_when_score_is_too_low(tmp_path: Path):
    chunks = [
        Chunk(
            chunk_id="a",
            source_id="a",
            source_url="https://example.com/a",
            title="A",
            heading="A",
            text="Serverless API Key는 API Key 관리 메뉴에서 발급합니다.",
            chunk_index=0,
        )
    ]
    index_path = tmp_path / "index.json"
    JsonVectorIndex.build(chunks, HashingEmbeddingClient(), index_path)
    settings = Settings(
        llm_provider="mock",
        embedding_provider="local",
        rag_min_score=0.99,
        index_path=index_path,
    )

    response = RAGPipeline(settings).query("오늘 서울 날씨는?")

    assert response.status == "insufficient_context"
    assert response.citations == []


def test_pipeline_streams_metadata_tokens_and_done_event(tmp_path: Path):
    chunks = [
        Chunk(
            chunk_id="a",
            source_id="a",
            source_url="https://example.com/a",
            title="API Key",
            heading="API Key 발급",
            text="Serverless API Key는 API Key 관리 메뉴에서 발급합니다.",
            chunk_index=0,
        )
    ]
    index_path = tmp_path / "index.json"
    JsonVectorIndex.build(chunks, HashingEmbeddingClient(), index_path)
    settings = Settings(
        llm_provider="mock",
        embedding_provider="local",
        rag_min_score=0.0,
        index_path=index_path,
    )

    events = list(
        RAGPipeline(settings).stream_query_events(
            "Serverless API Key는 어디에서 발급하나요?"
        )
    )

    assert events[0]["event"] == "metadata"
    assert any(event["event"] == "token" for event in events)
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "answered"
    assert events[-1]["data"]["citations"][0]["source_url"] == "https://example.com/a"


def test_insufficient_phrase_inside_substantive_answer_is_not_refusal():
    answer = (
        "문서에 따르면 Serverless API Key는 API Key 관리 메뉴에서 발급합니다. "
        "다만 문서에서 구체적인 내부 승인 절차는 확인할 수 없습니다."
    )

    assert not RAGPipeline._looks_insufficient(answer)


def test_short_insufficient_answer_is_refusal():
    assert RAGPipeline._looks_insufficient("제공된 문서에서 해당 내용을 확인할 수 없습니다.")
