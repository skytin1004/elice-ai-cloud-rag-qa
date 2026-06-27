from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from elice_rag.config import get_settings
from elice_rag.rag.generate import RAGPipeline
from elice_rag.rag.ingest import build_index, download_sources, process_raw_documents

from .gold_set import load_gold_set
from .metrics import score_example, summarize
from .report import write_reports


def _run_eval(label: str, out_path: Path) -> dict[str, float]:
    settings = get_settings()
    pipeline = RAGPipeline.from_settings(settings)
    examples = load_gold_set("eval/gold_set.jsonl")
    scores = [score_example(example, pipeline.query(example.question)) for example in examples]
    summary = summarize(scores)
    write_reports(
        scores=scores,
        summary=summary,
        settings=settings,
        out_path=out_path,
        label=label,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs improved RAG experiment")
    parser.add_argument("--out", default="eval/reports/experiment.md")
    parser.add_argument("--baseline-out", default="eval/reports/baseline-fixed.md")
    parser.add_argument("--improved-out", default="eval/reports/improved-heading.md")
    parser.add_argument("--redownload", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.redownload or not settings.raw_dir.exists():
        download_sources(settings)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["RAG_TOP_K"] = "3"
    os.environ["RAG_MIN_SCORE"] = "0.0"
    process_raw_documents(get_settings(), strategy="fixed")
    build_index(get_settings())
    baseline = _run_eval("baseline-fixed-top3", Path(args.baseline_out))

    os.environ["RAG_TOP_K"] = "5"
    os.environ["RAG_MIN_SCORE"] = "0.08"
    process_raw_documents(get_settings(), strategy="heading")
    build_index(get_settings())
    improved = _run_eval("improved-heading-threshold", Path(args.improved_out))

    payload = {"baseline": baseline, "improved": improved}
    out_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    out_path.write_text(_render_comparison(baseline, improved), encoding="utf-8", newline="\n")
    print(payload)


def _render_comparison(baseline: dict[str, float], improved: dict[str, float]) -> str:
    metrics = [
        "retrieval_recall_at_k",
        "citation_hit_rate",
        "refusal_accuracy",
        "faithfulness",
    ]
    lines = [
        "# Before/After Experiment",
        "",
        "Hypothesis: heading-aware chunking plus a score threshold improves citation quality and refusal behavior compared with naive fixed-size chunking.",
        "",
        "| Metric | Baseline fixed/top-3 | Improved heading/top-5 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        before = baseline.get(metric, 0.0)
        after = improved.get(metric, 0.0)
        lines.append(f"| {metric} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    retrieval_delta = improved.get("retrieval_recall_at_k", 0.0) - baseline.get(
        "retrieval_recall_at_k", 0.0
    )
    citation_delta = improved.get("citation_hit_rate", 0.0) - baseline.get(
        "citation_hit_rate", 0.0
    )
    refusal_delta = improved.get("refusal_accuracy", 0.0) - baseline.get(
        "refusal_accuracy", 0.0
    )
    faithfulness_delta = improved.get("faithfulness", 0.0) - baseline.get(
        "faithfulness", 0.0
    )
    if retrieval_delta >= 0 and citation_delta >= 0 and refusal_delta >= 0:
        analysis = (
            "Analysis: the hypothesis is supported by this provider configuration. "
            "The improved setting preserved or increased retrieval, citation, and refusal metrics."
        )
    elif retrieval_delta >= 0 and citation_delta < 0:
        analysis = (
            "Analysis: the hypothesis is only partially supported. The improved setting "
            "retrieved expected sources more often, but citation selection became less stable. "
            "This suggests retrieval context selection and final citation selection should be tuned separately."
        )
    elif retrieval_delta < 0 or citation_delta < 0 or faithfulness_delta < 0:
        analysis = (
            "Analysis: the hypothesis is not supported for this provider configuration. "
            "The fixed-size baseline outperformed the heading-aware setting on at least one key metric. "
            "A likely cause is that wider fixed chunks provided enough semantic context for the embedding model, "
            "while narrower heading chunks sometimes shifted the top-k context or citation away from the expected source."
        )
    else:
        analysis = (
            "Analysis: this report is generated automatically from the same gold set and provider configuration. "
            "Any metric regression should be analyzed by inspecting the per-example baseline and improved reports next to this comparison."
        )
    lines.extend(
        [
            "",
            analysis,
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
