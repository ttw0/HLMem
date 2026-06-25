from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .core_memory import build_core_memory_from_memcells, write_json
from .memcell import (
    extract_memcells_from_messages,
    load_locomo_conversations,
    memcells_to_episodic_rows,
)
from .openai_protocol import OpenAIProtocolClient


def sample_memories() -> list[dict]:
    return [
        {
            "store": "core",
            "memory_id": "core_smoke_001",
            "content": "用户不吃香菜。",
            "entity": "用户",
            "field": "allergy",
            "confidence": 0.9,
            "updated_at": "2026-05-01T10:00:00+08:00",
            "evidence_ids": ["smoke_dialog_001_turn_01"],
        },
        {
            "store": "core",
            "memory_id": "core_smoke_002",
            "content": "妈妈偏好清淡、适合长辈的餐厅。",
            "entity": "妈妈",
            "field": "preference",
            "confidence": 0.8,
            "updated_at": "2026-05-02T10:00:00+08:00",
            "evidence_ids": ["smoke_dialog_001_turn_02"],
        },
        {
            "store": "episodic",
            "memory_id": "epi_smoke_001",
            "event": "用户计划和妈妈去北京。",
            "time": "2026-05-09",
            "location": "北京",
            "participants": ["用户", "妈妈"],
            "status": "planned",
            "evidence_ids": ["smoke_dialog_003_turn_02"],
        },
    ]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sample_store(output: str | Path) -> None:
    write_jsonl(output, sample_memories())


def build_locomo_memcell(
    input_path: str | Path,
    output_path: str | Path,
    *,
    memcells_output: str | Path | None = None,
    core_slices_output: str | Path | None = None,
    merged_core_output: str | Path | None = None,
    limit_conversations: int | None = None,
    boundary_mode: str = "heuristic",
    episode_mode: str = "heuristic",
    core_mode: str = "llm",
    include_core: bool = True,
    max_core_slices: int | None = None,
) -> dict[str, int]:
    conversations = load_locomo_conversations(input_path, limit_conversations=limit_conversations)
    client = OpenAIProtocolClient.from_env() if "llm" in {boundary_mode, episode_mode, core_mode} else None

    all_memcells = []
    for conv_id, messages in conversations.items():
        all_memcells.extend(
            extract_memcells_from_messages(
                messages,
                conv_id=conv_id,
                boundary_mode=boundary_mode,
                episode_mode=episode_mode,
                client=client,
            )
        )

    rows = memcells_to_episodic_rows(all_memcells)
    core_rows: list[dict] = []
    profile_slices: list[dict] = []
    merged_core: dict[str, list[dict]] = {"not_agent": [], "is_agent": []}
    if include_core:
        core_rows, profile_slices, merged_core = build_core_memory_from_memcells(
            all_memcells,
            mode=core_mode,
            client=client,
            max_slices=max_core_slices,
        )
        rows.extend(core_rows)

    write_jsonl(output_path, rows)
    if memcells_output:
        write_jsonl(memcells_output, [memcell.to_dict() for memcell in all_memcells])
    if core_slices_output:
        write_json(core_slices_output, profile_slices)
    if merged_core_output:
        write_json(merged_core_output, merged_core)
    return {
        "memcells": len(all_memcells),
        "episodic": len([row for row in rows if row.get("store") == "episodic"]),
        "core": len(core_rows),
        "total": len(rows),
    }


def extract_memories_with_llm(text: str, client: OpenAIProtocolClient) -> list[dict]:
    prompt = (
        "Extract core and episodic memories from the text. "
        "Return JSON array only. Each item must include store and evidence_ids. Text:\n"
        f"{text}"
    )
    content = client.chat(
        [
            {"role": "system", "content": "You extract structured memories for a memory system."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("LLM memory extraction must return a JSON array")
    return parsed
