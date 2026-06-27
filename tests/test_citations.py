from elice_rag.rag.citations import citations_from_chunks
from elice_rag.rag.schemas import RetrievedChunk


def test_citations_are_deduplicated_by_url_and_heading():
    chunks = [
        RetrievedChunk(
            chunk_id="a",
            source_id="a",
            source_url="https://example.com/a",
            title="A",
            heading="Intro",
            text="one",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="b",
            source_id="a",
            source_url="https://example.com/a",
            title="A",
            heading="Intro",
            text="two",
            score=0.8,
        ),
    ]

    citations = citations_from_chunks(chunks)

    assert len(citations) == 1
    assert citations[0].chunk_id == "a"

