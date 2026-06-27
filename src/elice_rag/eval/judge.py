from __future__ import annotations

import json
import re
from dataclasses import dataclass

from elice_rag.providers.base import ChatClient
from elice_rag.rag.schemas import QueryResponse

from .gold_set import GoldExample


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a citation-based RAG system.
Evaluate only whether the answer is supported by the retrieved context and whether it
satisfies the expected acceptance criteria. Do not use outside knowledge.

Rubric:
- groundedness: 1.0 if all factual claims are supported by retrieved context/citations,
  0.5 if mostly supported but has minor unsupported claims, 0.0 if unsupported.
- correctness: 1.0 if the answer satisfies the acceptance criteria or correctly refuses,
  0.5 if partially correct, 0.0 if incorrect or refuses when the document evidence is sufficient.
- judge_pass: true only when both groundedness and correctness are at least 0.75.

Return JSON only with this schema:
{
  "groundedness": 0.0,
  "correctness": 0.0,
  "judge_pass": false,
  "rationale": "short reason"
}
"""


@dataclass(frozen=True)
class JudgeResult:
    groundedness: float
    correctness: float
    judge_score: float
    judge_pass: bool
    rationale: str
    raw_response: str


def judge_example(
    *,
    example: GoldExample,
    response: QueryResponse,
    judge_client: ChatClient,
    judge_model: str | None = None,
    max_context_chars: int = 6000,
) -> JudgeResult:
    raw = judge_client.generate(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(
                    example=example,
                    response=response,
                    max_context_chars=max_context_chars,
                ),
            },
        ],
        temperature=0.0,
        model=judge_model,
    )
    return parse_judge_response(raw)


def parse_judge_response(raw: str) -> JudgeResult:
    data = _extract_json_object(raw)
    groundedness = _score(data.get("groundedness"))
    correctness = _score(data.get("correctness"))
    judge_score = round((groundedness + correctness) / 2, 4)
    judge_pass = _bool(
        data.get("judge_pass"),
        default=groundedness >= 0.75 and correctness >= 0.75,
    )
    rationale = str(data.get("rationale", "")).strip()
    return JudgeResult(
        groundedness=groundedness,
        correctness=correctness,
        judge_score=judge_score,
        judge_pass=judge_pass,
        rationale=rationale[:500],
        raw_response=raw,
    )


def _build_user_prompt(
    *,
    example: GoldExample,
    response: QueryResponse,
    max_context_chars: int,
) -> str:
    return "\n\n".join(
        [
            f"Question:\n{example.question}",
            f"Question type: {example.type}",
            f"Should refuse: {example.should_refuse}",
            "Expected sources:\n" + _format_list(example.expected_sources),
            "Acceptance criteria:\n" + _format_list(example.acceptance_criteria),
            f"Response status:\n{response.status}",
            f"Answer:\n{response.answer}",
            "Citations:\n" + _format_citations(response),
            "Retrieved context:\n"
            + _format_context(response, max_context_chars=max_context_chars),
        ]
    )


def _format_list(values: list[str]) -> str:
    if not values:
        return "- (none)"
    return "\n".join(f"- {value}" for value in values)


def _format_citations(response: QueryResponse) -> str:
    if not response.citations:
        return "- (none)"
    lines = []
    for index, citation in enumerate(response.citations, start=1):
        lines.append(
            f"[{index}] title={citation.title}; heading={citation.heading}; "
            f"url={citation.source_url}; chunk_id={citation.chunk_id}"
        )
    return "\n".join(lines)


def _format_context(response: QueryResponse, *, max_context_chars: int) -> str:
    if not response.retrieved_context:
        return "- (none)"
    remaining = max_context_chars
    parts = []
    for index, chunk in enumerate(response.retrieved_context, start=1):
        header = (
            f"[{index}] title={chunk.title}; heading={chunk.heading}; "
            f"url={chunk.source_url}; score={chunk.score:.4f}; chunk_id={chunk.chunk_id}\n"
        )
        budget = max(0, remaining - len(header))
        if budget <= 0:
            break
        text = chunk.text[:budget]
        parts.append(header + text)
        remaining -= len(header) + len(text)
        if remaining <= 0:
            break
    return "\n\n".join(parts)


def _extract_json_object(raw: str) -> dict:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            text = object_match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge response was not valid JSON: {raw[:300]}") from exc
    if not isinstance(data, dict):
        raise ValueError("Judge response JSON must be an object")
    return data


def _score(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(1.0, max(0.0, number)), 4)


def _bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default
