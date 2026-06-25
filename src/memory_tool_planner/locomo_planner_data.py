from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .evidence import scoped_locomo_evidence_ids
from .schema import memory_tool_schema


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _conversation_speakers(item: dict[str, Any]) -> list[str]:
    speakers: list[str] = []
    for key in ["speaker_a", "speaker_b"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in speakers:
            speakers.append(value.strip())
    conversation = item.get("conversation", {})
    if isinstance(conversation, dict):
        for messages in conversation.values():
            if not isinstance(messages, list):
                continue
            for message in messages:
                if not isinstance(message, dict):
                    continue
                speaker = message.get("speaker")
                if isinstance(speaker, str) and speaker.strip() and speaker.strip() not in speakers:
                    speakers.append(speaker.strip())
    return speakers


def _question_persons(question: str, speakers: list[str]) -> list[str]:
    matches: list[tuple[int, str]] = []
    for speaker in speakers:
        pattern = rf"(?<![A-Za-z]){re.escape(speaker)}(?:'s)?(?![A-Za-z])"
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            matches.append((match.start(), speaker))
    return [speaker for _, speaker in sorted(matches)]


def iter_locomo_qa(
    locomo_path: str | Path,
    limit_conversations: int | None = None,
    limit_qa: int | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(locomo_path).read_text(encoding="utf-8"))
    samples = []
    for conv_index, item in enumerate(data):
        if limit_conversations is not None and conv_index >= limit_conversations:
            break
        if split in {"train"} and conv_index != 0:
            continue
        if split in {"valid", "eval"} and conv_index != 1:
            continue
        if split == "test" and conv_index < 2:
            continue
        sample_id = item.get("sample_id", str(conv_index))
        speakers = _conversation_speakers(item)
        for qa_index, qa in enumerate(item.get("qa", [])):
            if limit_qa is not None and len(samples) >= limit_qa:
                return samples
            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            annotated_person = str(qa.get("person", "")).strip()
            question_persons = _question_persons(question, speakers)
            if annotated_person:
                persons = [annotated_person] + [person for person in question_persons if person != annotated_person]
            else:
                persons = question_persons
            evidence = qa.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = [evidence]
            samples.append(
                {
                    "sample_id": sample_id,
                    "question_id": f"locomo_{conv_index}_qa{qa_index}",
                    "conversation_index": conv_index,
                    "qa_index": qa_index,
                    "question": question,
                    "person": persons[0] if persons else "",
                    "persons": persons,
                    "answer": qa.get("answer"),
                    "evidence": scoped_locomo_evidence_ids(evidence, conv_index),
                    "category": qa.get("category"),
                    "split": split or "all",
                }
            )
    return samples


def retrieved_evidence(tool_results: dict[str, Any]) -> set[str]:
    values = set()
    for result in tool_results.get("tool_results", []):
        if result.get("status") != "ok":
            continue
        for memory in result.get("result", {}).get("memories", []):
            for evidence_id in memory.get("evidence_ids", []):
                values.add(str(evidence_id))
    return values


def prepare_planner_data(
    locomo_path: str | Path,
    output_path: str | Path,
    *,
    limit_conversations: int | None = None,
    limit_qa: int | None = None,
    include_schema: bool = True,
    split: str | None = None,
) -> dict[str, Any]:
    rows = []
    for sample in iter_locomo_qa(locomo_path, limit_conversations=limit_conversations, limit_qa=limit_qa, split=split):
        row = {
            **sample,
            "planner_input": {
                "question_id": sample["question_id"],
                "question": sample["question"],
                "person": sample["person"],
                "persons": sample["persons"],
                "available_memory_tools": memory_tool_schema() if include_schema else [],
            },
        }
        rows.append(row)
    write_jsonl(output_path, rows)
    summary = {
        "output": str(output_path),
        "split": split or "all",
        "samples": len(rows),
        "label_status": "unlabeled_teacher_required",
    }
    return summary
