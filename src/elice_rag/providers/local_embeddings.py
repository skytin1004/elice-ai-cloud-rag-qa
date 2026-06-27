from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


def _features(text: str) -> list[str]:
    lowered = text.lower()
    tokens = TOKEN_RE.findall(lowered)
    feats: list[str] = []
    for token in tokens:
        if len(token) >= 2:
            feats.append(f"tok:{token}")
        if len(token) >= 4:
            for i in range(0, len(token) - 2):
                feats.append(f"tri:{token[i:i+3]}")
    return feats


@dataclass
class HashingEmbeddingClient:
    """Small deterministic embedding fallback for reproducible local testing."""

    dimensions: int = 384
    provider_name: str = "local"
    model_name: str = "hashing-ngrams-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in _features(text):
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

