from __future__ import annotations

import argparse
import os

from elice_rag.config import get_settings
from elice_rag.providers import create_chat_client
from elice_rag.rag.generate import RAGPipeline

from .gold_set import load_gold_set
from .judge import judge_example
from .metrics import score_example, summarize
from .report import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--gold", default="eval/gold_set.jsonl")
    parser.add_argument("--out", default="eval/reports/baseline.md")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--rerank-mode", choices=["none", "keyword"])
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run optional LLM-as-a-judge groundedness/correctness scoring.",
    )
    parser.add_argument(
        "--judge-model",
        help="Override the model used for LLM-as-a-judge. Defaults to the chat model.",
    )
    parser.add_argument("--judge-max-context-chars", type=int, default=6000)
    args = parser.parse_args()

    if args.top_k is not None:
        os.environ["RAG_TOP_K"] = str(args.top_k)
    if args.min_score is not None:
        os.environ["RAG_MIN_SCORE"] = str(args.min_score)
    if args.rerank_mode is not None:
        os.environ["RAG_RERANK_MODE"] = args.rerank_mode

    settings = get_settings()
    pipeline = RAGPipeline.from_settings(settings)
    examples = load_gold_set(args.gold)
    judge_client = create_chat_client(settings) if args.judge else None

    scores = []
    for example in examples:
        response = pipeline.query(example.question, top_k=settings.rag_top_k)
        score = score_example(example, response)
        if judge_client is not None:
            judge = judge_example(
                example=example,
                response=response,
                judge_client=judge_client,
                judge_model=args.judge_model,
                max_context_chars=args.judge_max_context_chars,
            )
            score.judge_groundedness = judge.groundedness
            score.judge_correctness = judge.correctness
            score.judge_score = judge.judge_score
            score.judge_pass = judge.judge_pass
            score.judge_rationale = judge.rationale
        scores.append(score)

    summary = summarize(scores)
    if args.judge:
        judge_model = args.judge_model
        if judge_model is None and judge_client is not None:
            judge_model = judge_client.model_name
        judge_policy = (
            "enabled; fixed rubric JSON judge; temperature=0.0; "
            f"model={judge_model}; max_context_chars={args.judge_max_context_chars}"
        )
    else:
        judge_model = None
        judge_policy = "not used; deterministic metric implementation"
    write_reports(
        scores=scores,
        summary=summary,
        settings=settings,
        out_path=args.out,
        label=args.label,
        judge_policy=judge_policy,
        judge_model=judge_model,
    )
    print(summary)


if __name__ == "__main__":
    main()
