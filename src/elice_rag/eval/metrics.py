from __future__ import annotations

from dataclasses import dataclass

from elice_rag.rag.schemas import QueryResponse

from .gold_set import GoldExample


@dataclass
class ExampleScore:
    id: str
    question: str
    response_status: str
    retrieval_hit: bool
    citation_hit: bool
    refusal_correct: bool | None
    faithfulness: float
    answer: str
    judge_groundedness: float | None = None
    judge_correctness: float | None = None
    judge_score: float | None = None
    judge_pass: bool | None = None
    judge_rationale: str | None = None


def source_matches(actual_url: str, expected_url: str) -> bool:
    return actual_url.rstrip("/") == expected_url.rstrip("/")


def retrieval_hit(example: GoldExample, response: QueryResponse) -> bool:
    if example.should_refuse:
        return False
    return any(
        source_matches(chunk.source_url, expected)
        for chunk in response.retrieved_context
        for expected in example.expected_sources
    )


def citation_hit(example: GoldExample, response: QueryResponse) -> bool:
    if example.should_refuse:
        return False
    return any(
        source_matches(citation.source_url, expected)
        for citation in response.citations
        for expected in example.expected_sources
    )


def refusal_correct(example: GoldExample, response: QueryResponse) -> bool | None:
    if not example.should_refuse:
        return None
    return response.status == "insufficient_context"


def faithfulness_score(example: GoldExample, response: QueryResponse) -> float:
    if response.status == "insufficient_context":
        return 1.0 if example.should_refuse else 0.0
    if not response.citations:
        return 0.0
    if not response.retrieved_context:
        return 0.0
    if example.expected_sources and citation_hit(example, response):
        return 1.0
    return 0.5


def score_example(example: GoldExample, response: QueryResponse) -> ExampleScore:
    return ExampleScore(
        id=example.id,
        question=example.question,
        response_status=response.status,
        retrieval_hit=retrieval_hit(example, response),
        citation_hit=citation_hit(example, response),
        refusal_correct=refusal_correct(example, response),
        faithfulness=faithfulness_score(example, response),
        answer=response.answer,
    )


def summarize(scores: list[ExampleScore]) -> dict[str, float]:
    answerable = [score for score in scores if score.refusal_correct is None]
    refusal = [score for score in scores if score.refusal_correct is not None]
    summary = {
        "total": float(len(scores)),
        "retrieval_recall_at_k": _mean([score.retrieval_hit for score in answerable]),
        "citation_hit_rate": _mean([score.citation_hit for score in answerable]),
        "refusal_accuracy": _mean([score.refusal_correct for score in refusal]),
        "faithfulness": _mean([score.faithfulness for score in scores]),
    }
    judged = [score for score in scores if score.judge_score is not None]
    if judged:
        summary.update(
            {
                "judge_groundedness": _mean(
                    [score.judge_groundedness for score in judged]
                ),
                "judge_correctness": _mean(
                    [score.judge_correctness for score in judged]
                ),
                "judge_score": _mean([score.judge_score for score in judged]),
                "judge_pass_rate": _mean([score.judge_pass for score in judged]),
            }
        )
    return summary


def _mean(values: list[bool | float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    return round(sum(clean) / len(clean), 4)
