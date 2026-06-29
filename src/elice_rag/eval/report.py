from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from elice_rag.config import Settings
from elice_rag.providers import create_chat_client, create_embedding_client

from .metrics import ExampleScore


def write_reports(
    *,
    scores: list[ExampleScore],
    summary: dict[str, float],
    settings: Settings,
    out_path: str | Path,
    label: str,
    judge_policy: str = "not used; deterministic metric implementation",
    judge_model: str | None = None,
) -> None:
    markdown_path = Path(out_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = markdown_path.with_suffix(".json")
    chat_client = create_chat_client(settings)
    embedding_client = create_embedding_client(settings)

    payload = {
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": {
            "llm_provider": settings.llm_provider,
            "llm_model": chat_client.model_name,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": embedding_client.model_name,
            "azure_chat_deployment": settings.azure_openai_chat_deployment_name,
            "elice_chat_model": settings.elice_chat_model,
            "judge_model": judge_model,
        },
        "retrieval": {
            "top_k": settings.rag_top_k,
            "min_score": settings.rag_min_score,
            "rerank_mode": settings.rag_rerank_mode,
        },
        "reproducibility": {
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "python_version": platform.python_version(),
            "seed": "not used; deterministic retrieval/eval path",
            "judge": judge_policy,
        },
        "summary": summary,
        "examples": [score.__dict__ for score in scores],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8", newline="\n")


def _render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        f"# Eval Report: {payload['label']}",
        "",
        "## Provider",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- LLM provider: `{payload['provider']['llm_provider']}`",
        f"- LLM model/deployment: `{payload['provider']['llm_model']}`",
        f"- Embedding provider: `{payload['provider']['embedding_provider']}`",
        f"- Embedding model: `{payload['provider']['embedding_model']}`",
        f"- Top-k: `{payload['retrieval']['top_k']}`",
        f"- Min score: `{payload['retrieval']['min_score']}`",
        f"- Rerank mode: `{payload['retrieval'].get('rerank_mode', 'none')}`",
        f"- Git commit: `{payload['reproducibility']['git_commit']}`",
        f"- Git branch: `{payload['reproducibility']['git_branch']}`",
        f"- Python: `{payload['reproducibility']['python_version']}`",
        f"- Seed policy: `{payload['reproducibility']['seed']}`",
        f"- Judge policy: `{payload['reproducibility']['judge']}`",
        "",
        "## Summary",
        "",
        "| Metric | Score |",
        "|---|---:|",
        f"| Retrieval recall@k | {summary['retrieval_recall_at_k']:.4f} |",
        f"| Citation hit rate | {summary['citation_hit_rate']:.4f} |",
        f"| Refusal accuracy | {summary['refusal_accuracy']:.4f} |",
        f"| Faithfulness | {summary['faithfulness']:.4f} |",
    ]
    if "judge_score" in summary:
        lines.extend(
            [
                f"| Judge groundedness | {summary['judge_groundedness']:.4f} |",
                f"| Judge correctness | {summary['judge_correctness']:.4f} |",
                f"| Judge score | {summary['judge_score']:.4f} |",
                f"| Judge pass rate | {summary['judge_pass_rate']:.4f} |",
            ]
        )
    lines.extend(["", "## Examples", ""])
    has_judge = any(example.get("judge_score") is not None for example in payload["examples"])
    if has_judge:
        lines.extend(
            [
                "| ID | Status | Retrieval | Citation | Refusal | Faithfulness | Judge | Judge pass |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| ID | Status | Retrieval | Citation | Refusal | Faithfulness |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
    for example in payload["examples"]:
        refusal = example["refusal_correct"]
        refusal_text = "" if refusal is None else str(refusal)
        if has_judge:
            judge_score = example["judge_score"]
            judge_text = "" if judge_score is None else f"{judge_score:.2f}"
            judge_pass = example["judge_pass"]
            judge_pass_text = "" if judge_pass is None else str(judge_pass)
            lines.append(
                "| {id} | {status} | {retrieval} | {citation} | {refusal} | {faithfulness:.2f} | {judge} | {judge_pass} |".format(
                    id=example["id"],
                    status=example["response_status"],
                    retrieval=str(example["retrieval_hit"]),
                    citation=str(example["citation_hit"]),
                    refusal=refusal_text,
                    faithfulness=example["faithfulness"],
                    judge=judge_text,
                    judge_pass=judge_pass_text,
                )
            )
        else:
            lines.append(
                "| {id} | {status} | {retrieval} | {citation} | {refusal} | {faithfulness:.2f} |".format(
                    id=example["id"],
                    status=example["response_status"],
                    retrieval=str(example["retrieval_hit"]),
                    citation=str(example["citation_hit"]),
                    refusal=refusal_text,
                    faithfulness=example["faithfulness"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"
