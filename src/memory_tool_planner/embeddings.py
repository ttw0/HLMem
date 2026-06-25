from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .config import OpenAISettings
from .openai_protocol import OpenAIProtocolClient


def hash_embedding(text: str, dims: int = 128) -> list[float]:
    vec = [0.0] * dims
    tokens = [token for token in text.lower().replace("\n", " ").split(" ") if token]
    for token in tokens or [text]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i, byte in enumerate(digest):
            idx = (i + byte) % dims
            vec[idx] += 1.0 if byte % 2 == 0 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) or 1.0) * (math.sqrt(sum(y * y for y in b)) or 1.0))


@dataclass
class EmbeddingProvider:
    settings: OpenAISettings

    @classmethod
    def from_env(cls) -> "EmbeddingProvider":
        return cls(OpenAISettings.from_env())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.settings.offline:
            return [hash_embedding(text) for text in texts]
        return OpenAIProtocolClient(self.settings).embed(texts)
