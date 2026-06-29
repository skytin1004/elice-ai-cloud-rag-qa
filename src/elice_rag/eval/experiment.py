from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path

from elice_rag.config import get_settings
from elice_rag.rag.generate import RAGPipeline
from elice_rag.rag.ingest import build_index, download_sources, process_raw_documents

from .gold_set import load_gold_set
from .metrics import score_example, summarize
from .report import write_reports


@dataclass(frozen=True)
class ExperimentConfig:
    key: str
    label: str
    hypothesis: str
    strategy: str
    top_k: int
    min_score: float
    rerank_mode: str
    report_path: Path


def _run_eval(label: str, out_path: Path) -> dict[str, float]:
    settings = get_settings()
    pipeline = RAGPipeline.from_settings(settings)
    examples = load_gold_set("eval/gold_set.jsonl")
    scores = [
        score_example(
            example,
            pipeline.query(example.question, top_k=settings.rag_top_k),
        )
        for example in examples
    ]
    summary = summarize(scores)
    write_reports(
        scores=scores,
        summary=summary,
        settings=settings,
        out_path=out_path,
        label=label,
    )
    return summary


@contextmanager
def _preserve_files(paths: list[Path]) -> Iterator[None]:
    snapshots: dict[Path, bytes | None] = {}
    for path in paths:
        snapshots[path] = path.read_bytes() if path.exists() else None
    try:
        yield
    finally:
        for path, content in snapshots.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)


def _build_experiments(args: argparse.Namespace) -> list[ExperimentConfig]:
    sweep_dir = Path(args.sweep_dir or Path(args.out).parent)
    prefix = args.report_prefix
    return [
        ExperimentConfig(
            key="baseline-fixed-top3",
            label="baseline-fixed-top3",
            hypothesis="Fixed-size chunks provide broad context and are a simple baseline.",
            strategy="fixed",
            top_k=3,
            min_score=0.0,
            rerank_mode="none",
            report_path=Path(args.baseline_out),
        ),
        ExperimentConfig(
            key="heading-top5-threshold",
            label="heading-top5-threshold",
            hypothesis="Heading-aware chunks plus a threshold improve citation precision and refusal behavior.",
            strategy="heading",
            top_k=5,
            min_score=0.08,
            rerank_mode="none",
            report_path=Path(args.improved_out),
        ),
        ExperimentConfig(
            key="heading-top8-relaxed",
            label="heading-top8-relaxed",
            hypothesis="Increasing top-k and relaxing the threshold can recover expected sources missed by narrower heading chunks.",
            strategy="heading",
            top_k=8,
            min_score=0.04,
            rerank_mode="none",
            report_path=sweep_dir / f"{prefix}heading-top8-relaxed.md",
        ),
        ExperimentConfig(
            key="heading-top5-strict",
            label="heading-top5-strict",
            hypothesis="A stricter threshold should reduce weak evidence but may lower recall.",
            strategy="heading",
            top_k=5,
            min_score=0.16,
            rerank_mode="none",
            report_path=sweep_dir / f"{prefix}heading-top5-strict.md",
        ),
        ExperimentConfig(
            key="heading-top8-keyword-rerank",
            label="heading-top8-keyword-rerank",
            hypothesis="Keyword-aware reranking can improve citation ordering after broad top-k retrieval.",
            strategy="heading",
            top_k=8,
            min_score=0.04,
            rerank_mode="keyword",
            report_path=sweep_dir / f"{prefix}heading-top8-keyword-rerank.md",
        ),
    ]


def _apply_config(config: ExperimentConfig) -> None:
    os.environ["RAG_TOP_K"] = str(config.top_k)
    os.environ["RAG_MIN_SCORE"] = str(config.min_score)
    os.environ["RAG_RERANK_MODE"] = config.rerank_mode


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part C RAG experiment log")
    parser.add_argument("--out", default="eval/reports/experiment.md")
    parser.add_argument("--baseline-out", default="eval/reports/baseline-fixed.md")
    parser.add_argument("--improved-out", default="eval/reports/improved-heading.md")
    parser.add_argument("--sweep-dir", default="")
    parser.add_argument("--report-prefix", default="")
    parser.add_argument("--redownload", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.redownload or not settings.raw_dir.exists():
        download_sources(settings)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    experiments = _build_experiments(args)

    env_snapshot = {
        key: os.environ.get(key)
        for key in ["RAG_TOP_K", "RAG_MIN_SCORE", "RAG_RERANK_MODE"]
    }
    results: list[dict] = []
    try:
        with _preserve_files([settings.chunks_path, settings.index_path]):
            for experiment in experiments:
                _apply_config(experiment)
                process_raw_documents(get_settings(), strategy=experiment.strategy)
                build_index(get_settings())
                summary = _run_eval(experiment.label, experiment.report_path)
                results.append(
                    {
                        "config": {
                            "key": experiment.key,
                            "label": experiment.label,
                            "hypothesis": experiment.hypothesis,
                            "strategy": experiment.strategy,
                            "top_k": experiment.top_k,
                            "min_score": experiment.min_score,
                            "rerank_mode": experiment.rerank_mode,
                            "report_path": str(experiment.report_path),
                        },
                        "summary": summary,
                    }
                )
    finally:
        _restore_env(env_snapshot)

    payload = {"experiments": results}
    out_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    out_path.write_text(_render_comparison(results), encoding="utf-8", newline="\n")
    print(payload)


def _render_comparison(results: list[dict]) -> str:
    baseline = results[0]["summary"]
    analyses = _experiment_analyses(results)
    lines = [
        "# Part C Experiment Log",
        "",
        "The same Gold Set and metric implementation were used for every run. The goal is not to hide failed attempts, but to observe how retrieval, citation, and refusal behavior change under controlled configuration changes.",
        "",
        "## Summary",
        "",
        "| Experiment | Strategy | Top-k | Min score | Rerank | Recall@k | Citation hit | Refusal | Faithfulness | Faithfulness delta |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        config = result["config"]
        summary = result["summary"]
        delta = summary.get("faithfulness", 0.0) - baseline.get("faithfulness", 0.0)
        lines.append(
            "| {label} | {strategy} | {top_k} | {min_score:.2f} | {rerank_mode} | {recall:.4f} | {citation:.4f} | {refusal:.4f} | {faithfulness:.4f} | {delta:+.4f} |".format(
                label=config["label"],
                strategy=config["strategy"],
                top_k=config["top_k"],
                min_score=config["min_score"],
                rerank_mode=config["rerank_mode"],
                recall=summary.get("retrieval_recall_at_k", 0.0),
                citation=summary.get("citation_hit_rate", 0.0),
                refusal=summary.get("refusal_accuracy", 0.0),
                faithfulness=summary.get("faithfulness", 0.0),
                delta=delta,
            )
        )

    recovery_lines = _render_recovery_comparison(results)
    if recovery_lines:
        lines.extend(["", *recovery_lines])

    lines.extend(["", "## Per-Experiment Notes", ""])
    for result in results:
        config = result["config"]
        summary = result["summary"]
        analysis = analyses[config["label"]]
        lines.extend(
            [
                f"### {config['label']}",
                "",
                f"- Hypothesis: {config['hypothesis']}",
                f"- Result: recall={summary.get('retrieval_recall_at_k', 0.0):.4f}, citation={summary.get('citation_hit_rate', 0.0):.4f}, refusal={summary.get('refusal_accuracy', 0.0):.4f}, faithfulness={summary.get('faithfulness', 0.0):.4f}",
                f"- Analysis: {analysis}",
                f"- Next step: {_next_step_for(config['label'])}",
                "",
            ]
        )

    best = max(results, key=lambda item: item["summary"].get("faithfulness", 0.0))
    improved = results[1]["summary"] if len(results) > 1 else baseline
    retrieval_delta = improved.get("retrieval_recall_at_k", 0.0) - baseline.get(
        "retrieval_recall_at_k", 0.0
    )
    citation_delta = improved.get("citation_hit_rate", 0.0) - baseline.get(
        "citation_hit_rate", 0.0
    )
    faithfulness_delta = improved.get("faithfulness", 0.0) - baseline.get(
        "faithfulness", 0.0
    )
    if retrieval_delta < 0 or citation_delta < 0 or faithfulness_delta < 0:
        primary_analysis = (
            "The primary heading-aware hypothesis was not supported by this run. "
            "The fixed-size baseline remained stronger on at least one key metric, which suggests that wider chunks can be beneficial for this corpus/provider combination."
        )
    else:
        primary_analysis = (
            "The primary heading-aware hypothesis was supported by this run. "
            "The improved setting preserved or increased the main retrieval and citation metrics."
        )

    lines.extend(
        [
            "",
            "## Analysis",
            "",
            primary_analysis,
            "",
            f"The highest faithfulness score in this sweep was `{best['config']['label']}`. This does not automatically make it the production choice; implementation complexity, citation interpretability, and possible Gold Set overfitting still need to be considered.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_recovery_comparison(results: list[dict]) -> list[str]:
    by_label = {result["config"]["label"]: result for result in results}
    failed = by_label.get("heading-top5-threshold")
    recovered = by_label.get("baseline-fixed-top3")
    if failed is None or recovered is None:
        return []

    failed_summary = failed["summary"]
    recovered_summary = recovered["summary"]
    metrics = [
        ("retrieval_recall_at_k", "Retrieval recall@k"),
        ("citation_hit_rate", "Citation hit rate"),
        ("refusal_accuracy", "Refusal accuracy"),
        ("faithfulness", "Faithfulness"),
    ]
    lines = [
        "## Recovery Comparison",
        "",
        "The initial heading-aware hypothesis failed against the fixed-size control. The recovery comparison treats that failed heading-aware setting as the new before state, then checks whether returning to a broad-context fixed chunking strategy restores the measured quality.",
        "",
        "| Metric | Failed heading/top-5 | Broad fixed/top-3 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key, label in metrics:
        before = failed_summary.get(key, 0.0)
        after = recovered_summary.get(key, 0.0)
        lines.append(f"| {label} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    lines.extend(
        [
            "",
            "This is not a hidden replacement of the failed experiment. It is the follow-up design decision: preserve broad local context for answer generation, then handle stricter evidence selection as a separate next step.",
        ]
    )
    return lines


def _experiment_analyses(results: list[dict]) -> dict[str, str]:
    baseline = results[0]["summary"]
    analyses: dict[str, str] = {}
    for result in results:
        label = result["config"]["label"]
        summary = result["summary"]
        recall_delta = summary.get("retrieval_recall_at_k", 0.0) - baseline.get(
            "retrieval_recall_at_k", 0.0
        )
        citation_delta = summary.get("citation_hit_rate", 0.0) - baseline.get(
            "citation_hit_rate", 0.0
        )
        faithfulness_delta = summary.get("faithfulness", 0.0) - baseline.get(
            "faithfulness", 0.0
        )
        if label == "baseline-fixed-top3":
            analyses[label] = (
                "This is the control run. It provides the comparison point for the later changes."
            )
        elif recall_delta < 0 or citation_delta < 0 or faithfulness_delta < 0:
            analyses[label] = (
                "The change did not improve the target metrics. A likely cause is that the fixed-size baseline retained broader local context, while the changed setting shifted the retrieved or cited source away from the expected evidence."
            )
        elif recall_delta == 0 and citation_delta == 0 and faithfulness_delta == 0:
            analyses[label] = (
                "The change was neutral against this Gold Set. It may still matter for latency, cost, or edge cases, but it did not create measurable quality gain here."
            )
        else:
            analyses[label] = (
                "The change improved at least one target metric without reducing the measured guardrail metric in this run."
            )
    return analyses


def _next_step_for(label: str) -> str:
    if label == "baseline-fixed-top3":
        return "Keep it as the control setting and compare every later change against it."
    if "top8" in label and "rerank" not in label:
        return "Inspect missed examples before increasing context further; larger top-k alone may add noise."
    if "strict" in label:
        return "Tune the threshold per query type instead of applying a single stricter global cutoff."
    if "rerank" in label:
        return "Replace simple keyword overlap with BM25, heading proximity, and source diversity features."
    return "Separate answer context selection from final citation selection."


if __name__ == "__main__":
    main()
