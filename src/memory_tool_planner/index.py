from __future__ import annotations

from pathlib import Path

from .builder import read_jsonl, write_jsonl
from .embeddings import EmbeddingProvider
from .schema import search_text


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    return text[:max_chars]


def index_store(
    input_path: str | Path,
    output_path: str | Path,
    provider: EmbeddingProvider | None = None,
    batch_size: int = 64,
    max_text_chars: int | None = None,
) -> None:
    provider = provider or EmbeddingProvider.from_env()
    rows = read_jsonl(input_path)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [_truncate(search_text(row), max_text_chars) for row in batch]
        embeddings = provider.embed(texts)
        for row, embedding, text in zip(batch, embeddings, texts):
            row["search_text"] = text
            row["embedding"] = embedding
    write_jsonl(output_path, rows)
