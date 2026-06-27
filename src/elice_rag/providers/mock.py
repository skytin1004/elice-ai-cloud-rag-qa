from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .local_embeddings import HashingEmbeddingClient


@dataclass
class MockChatClient:
    provider_name: str = "mock"
    model_name: str = "extractive-mock-v1"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        prompt = "\n".join(message.get("content", "") for message in messages)
        context_match = re.search(
            r"<context>\s*(.*?)\s*</context>", prompt, flags=re.DOTALL | re.IGNORECASE
        )
        context = context_match.group(1).strip() if context_match else prompt
        lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip()
            and not line.strip().startswith(("[", "title=", "url=", "heading=", "chunk_id=", "---"))
        ]
        sentences = re.split(r"(?<=[.!?。])\s+|\n+", "\n".join(lines))
        for sentence in sentences:
            cleaned = re.sub(r"\[[^\]]+\]", "", sentence).strip()
            if len(cleaned) > 25:
                return f"문서 근거에 따르면 {cleaned[:500]}"
        return "제공된 문서에서 해당 내용을 확인할 수 없습니다."

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> Iterator[str]:
        answer = self.generate(messages, temperature=temperature, model=model)
        for part in re.split(r"(\s+)", answer):
            if part:
                yield part


class MockEmbeddingClient(HashingEmbeddingClient):
    provider_name = "mock"
    model_name = "hashing-ngrams-v1"
