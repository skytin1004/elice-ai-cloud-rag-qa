from elice_rag.eval.gold_set import GoldExample
from elice_rag.eval.metrics import score_example, summarize
from elice_rag.rag.schemas import Citation, QueryResponse, RetrievedChunk


def test_eval_metrics_score_expected_source_hits():
    example = GoldExample(
        id="q",
        question="API Key?",
        type="factual",
        expected_sources=["https://example.com/api-key"],
    )
    response = QueryResponse(
        status="answered",
        answer="API Key는 관리 메뉴에서 발급합니다.",
        citations=[
            Citation(
                source_url="https://example.com/api-key",
                title="API Key",
                heading="발급",
                chunk_id="c1",
            )
        ],
        confidence="high",
        retrieved_context=[
            RetrievedChunk(
                chunk_id="c1",
                source_id="api-key",
                source_url="https://example.com/api-key",
                title="API Key",
                heading="발급",
                text="API Key는 관리 메뉴에서 발급합니다.",
                score=0.8,
            )
        ],
    )

    score = score_example(example, response)
    summary = summarize([score])

    assert score.retrieval_hit is True
    assert score.citation_hit is True
    assert summary["retrieval_recall_at_k"] == 1.0
    assert summary["citation_hit_rate"] == 1.0


def test_eval_metrics_summarize_optional_judge_scores():
    example = GoldExample(
        id="q",
        question="API Key?",
        type="factual",
        expected_sources=["https://example.com/api-key"],
    )
    response = QueryResponse(
        status="answered",
        answer="API Key는 관리 메뉴에서 발급합니다.",
        citations=[
            Citation(
                source_url="https://example.com/api-key",
                title="API Key",
                heading="발급",
                chunk_id="c1",
            )
        ],
        confidence="high",
        retrieved_context=[
            RetrievedChunk(
                chunk_id="c1",
                source_id="api-key",
                source_url="https://example.com/api-key",
                title="API Key",
                heading="발급",
                text="API Key는 관리 메뉴에서 발급합니다.",
                score=0.8,
            )
        ],
    )

    score = score_example(example, response)
    score.judge_groundedness = 1.0
    score.judge_correctness = 0.5
    score.judge_score = 0.75
    score.judge_pass = True

    summary = summarize([score])

    assert summary["judge_groundedness"] == 1.0
    assert summary["judge_correctness"] == 0.5
    assert summary["judge_score"] == 0.75
    assert summary["judge_pass_rate"] == 1.0
