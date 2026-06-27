from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class ChatClient(Protocol):
    provider_name: str
    model_name: str

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        ...

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> Iterator[str]:
        ...


class EmbeddingClient(Protocol):
    provider_name: str
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...
