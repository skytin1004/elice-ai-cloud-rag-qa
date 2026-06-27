from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class GoldExample(BaseModel):
    id: str
    question: str
    type: str
    expected_sources: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    should_refuse: bool = False


def load_gold_set(path: str | Path) -> list[GoldExample]:
    examples: list[GoldExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(GoldExample.model_validate_json(line))
    return examples

