from pathlib import Path

from elice_rag.providers.local_embeddings import HashingEmbeddingClient
from elice_rag.rag.index import JsonVectorIndex
from elice_rag.rag.schemas import Chunk


def test_json_vector_index_retrieves_relevant_chunk(tmp_path: Path):
    chunks = [
        Chunk(
            chunk_id="a",
            source_id="a",
            source_url="https://example.com/api-key",
            title="API Key",
            heading="API Key",
            text="Serverless API Key는 API Key 관리 메뉴에서 발급합니다.",
            chunk_index=0,
        ),
        Chunk(
            chunk_id="b",
            source_id="b",
            source_url="https://example.com/runbox",
            title="Runbox",
            heading="인스턴스",
            text="런박스 인스턴스는 GPU 워크로드를 실행합니다.",
            chunk_index=0,
        ),
    ]
    client = HashingEmbeddingClient()
    index = JsonVectorIndex.build(chunks, client, tmp_path / "index.json")

    results = index.search("Serverless API Key 발급 위치", client, top_k=1)

    assert results[0].chunk_id == "a"

