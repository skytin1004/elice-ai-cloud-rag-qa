from elice_rag.rag.chunk import heading_aware_chunks
from elice_rag.rag.schemas import CleanDocument


def test_heading_aware_chunks_preserve_metadata():
    document = CleanDocument(
        id="doc",
        title="Doc Title",
        url="https://example.com/doc",
        text="## API Key\nAPI Key는 Serverless 탭에서 발급합니다.\n## 비용\n호출량 기준으로 과금됩니다.",
    )

    chunks = heading_aware_chunks(document, target_tokens=20, overlap_tokens=5)

    assert chunks
    assert chunks[0].source_url == "https://example.com/doc"
    assert chunks[0].heading == "API Key"
    assert chunks[0].chunk_id.startswith("doc-")

