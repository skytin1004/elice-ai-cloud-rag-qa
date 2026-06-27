from __future__ import annotations

import argparse
import os

from elice_rag.config import get_settings
from elice_rag.rag.generate import RAGPipeline

from .gold_set import load_gold_set
from .metrics import score_example, summarize
from .report import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--gold", default="eval/gold_set.jsonl")
    parser.add_argument("--out", default="eval/reports/baseline.md")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-score", type=float)
    args = parser.parse_args()

    if args.top_k is not None:
        os.environ["RAG_TOP_K"] = str(args.top_k)
    if args.min_score is not None:
        os.environ["RAG_MIN_SCORE"] = str(args.min_score)

    settings = get_settings()
    pipeline = RAGPipeline.from_settings(settings)
    examples = load_gold_set(args.gold)

    scores = []
    for example in examples:
        response = pipeline.query(example.question, top_k=settings.rag_top_k)
        scores.append(score_example(example, response))

    summary = summarize(scores)
    write_reports(
        scores=scores,
        summary=summary,
        settings=settings,
        out_path=args.out,
        label=args.label,
    )
    print(summary)


if __name__ == "__main__":
    main()

