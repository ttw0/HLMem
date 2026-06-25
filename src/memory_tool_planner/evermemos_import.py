from __future__ import annotations

import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .core_memory import (
    build_core_memory_from_memcells,
    convert_to_core_memory,
    core_memory_rows,
    extract_core_profile_slices,
    merge_core_profile_slices,
)
from .evidence import scoped_locomo_evidence_ids
from .memcell import load_locomo_conversations
from .memcell import MemCell
from .openai_protocol import OpenAIProtocolClient
from .schema import search_text


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _conversation_index(path: Path) -> int:
    match = re.search(r"memcell_list_conv_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"cannot parse conversation index from {path}")
    return int(match.group(1))


def _normalize_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return parsed.isoformat().replace("+00:00", "Z")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _message_key(message: dict[str, Any], *, include_content: bool = True) -> tuple[str, str, str] | tuple[str, str]:
    timestamp = _normalize_time(message.get("timestamp"))
    speaker = _normalize_text(message.get("speaker_name") or message.get("user_name") or message.get("speaker_id"))
    if include_content:
        return timestamp, speaker, _normalize_text(message.get("content"))
    return timestamp, speaker


def _locomo_indexes(locomo_path: str | Path) -> dict[int, dict[str, Any]]:
    conversations = load_locomo_conversations(locomo_path)
    indexed: dict[int, dict[str, Any]] = {}
    for conv_id, messages in conversations.items():
        exact: dict[tuple[str, str, str], dict[str, Any]] = {}
        loose: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for message in messages:
            exact[_message_key(message, include_content=True)] = message
            loose.setdefault(_message_key(message, include_content=False), []).append(message)
        indexed[int(conv_id)] = {"messages": messages, "exact": exact, "loose": loose}
    return indexed


def _match_locomo_message(raw: dict[str, Any], index: dict[str, Any], cursor: int) -> tuple[dict[str, Any] | None, int]:
    exact = index["exact"].get(_message_key(raw, include_content=True))
    if exact:
        messages = index["messages"]
        try:
            return exact, messages.index(exact) + 1
        except ValueError:
            return exact, cursor

    loose_candidates = index["loose"].get(_message_key(raw, include_content=False), [])
    if loose_candidates:
        messages = index["messages"]
        later = [item for item in loose_candidates if messages.index(item) >= cursor]
        selected = later[0] if later else loose_candidates[0]
        return selected, messages.index(selected) + 1

    messages = index["messages"]
    if cursor < len(messages):
        raw_speaker = _normalize_text(raw.get("speaker_name") or raw.get("user_name") or raw.get("speaker_id"))
        for offset in range(cursor, min(cursor + 5, len(messages))):
            candidate = messages[offset]
            candidate_speaker = _normalize_text(candidate.get("speaker_name") or candidate.get("user_name") or candidate.get("speaker_id"))
            if candidate_speaker == raw_speaker:
                return candidate, offset + 1
    return None, cursor


def _strip_large_embeddings(memcell: dict[str, Any]) -> None:
    event_log = memcell.get("event_log")
    if isinstance(event_log, dict):
        event_log.pop("fact_embeddings", None)


def _atomic_facts(memcell: dict[str, Any]) -> list[str]:
    event_log = memcell.get("event_log")
    if not isinstance(event_log, dict):
        return []
    facts = event_log.get("atomic_fact")
    if not isinstance(facts, list):
        return []
    return [str(fact).strip() for fact in facts if str(fact).strip()]


def _time_range(original_data: list[dict[str, Any]], fallback: str) -> dict[str, Any]:
    timestamps = [_normalize_time(item.get("timestamp")) for item in original_data if item.get("timestamp")]
    if not timestamps:
        return {"start_time": fallback, "end_time": fallback, "message_count": len(original_data)}
    return {"start_time": timestamps[0], "end_time": timestamps[-1], "message_count": len(original_data)}


def _episode_row(memcell: dict[str, Any]) -> dict[str, Any]:
    event_id = str(memcell.get("event_id") or memcell.get("memcell_id") or "")
    conversation_index = int(memcell["conversation_index"])
    original_data = memcell.get("original_data") if isinstance(memcell.get("original_data"), list) else []
    evidence_ids = scoped_locomo_evidence_ids([item.get("dia_id") for item in original_data if item.get("dia_id")], conversation_index)
    atomic_facts = _atomic_facts(memcell)
    timestamp = _normalize_time(memcell.get("timestamp"))
    row = {
        "store": "episodic",
        "memory_id": f"epi_evermemos_locomo_{conversation_index}_{event_id}",
        "memcell_id": event_id,
        "source": "evermemos_locomo_baseline",
        "conversation_id": f"locomo_{conversation_index}",
        "conversation_index": conversation_index,
        "event": str(memcell.get("episode") or memcell.get("summary") or ""),
        "subject": str(memcell.get("subject") or ""),
        "summary": str(memcell.get("summary") or ""),
        "atomic_facts": atomic_facts,
        "time": timestamp,
        "time_range": _time_range(original_data, timestamp),
        "location": "",
        "participants": memcell.get("participants") or memcell.get("user_id_list") or [],
        "status": "completed",
        "evidence_ids": evidence_ids,
        "metadata": {
            "event_id": event_id,
            "type": memcell.get("type"),
            "keywords": memcell.get("keywords"),
            "linked_entities": memcell.get("linked_entities"),
        },
    }
    row["search_text"] = search_text(row) + " " + " ".join(atomic_facts)
    return row


def _memcell_from_row(row: dict[str, Any]) -> MemCell:
    memcell_id = str(row.get("memcell_id") or row.get("event_id") or "")
    user_id_list = row.get("user_id_list")
    participants = row.get("participants")
    if not isinstance(user_id_list, list):
        user_id_list = []
    if not isinstance(participants, list):
        participants = user_id_list
    return MemCell(
        memcell_id=memcell_id,
        type=str(row.get("type") or "Conversation"),
        user_id_list=[str(item) for item in user_id_list if item],
        original_data=row.get("original_data", []) if isinstance(row.get("original_data"), list) else [],
        timestamp=_normalize_time(row.get("timestamp")),
        summary=str(row.get("summary") or ""),
        participants=[str(item) for item in participants if item],
        group_id=row.get("group_id"),
        episode=str(row.get("episode") or ""),
        subject=str(row.get("subject") or ""),
        extend={
            "source": row.get("source", "evermemos_locomo_baseline"),
            "conversation_id": row.get("conversation_id"),
            "conversation_index": row.get("conversation_index"),
            "evidence_ids": row.get("evidence_ids", []),
        },
    )


def _extract_llm_profile_slices_for_memcell(memcell: MemCell, max_slices: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    client = OpenAIProtocolClient.from_env()
    profile_slices = extract_core_profile_slices([memcell], mode="llm", client=client, max_slices=max_slices)
    for item in profile_slices:
        item["_core_mode"] = "llm"
    return memcell.memcell_id, profile_slices


def build_evermemos_core_memory(
    memcells_path: str | Path,
    episodic_path: str | Path | None,
    *,
    core_output: str | Path,
    core_slices_output: str | Path | None = None,
    merged_core_output: str | Path | None = None,
    combined_output: str | Path | None = None,
    summary_output: str | Path | None = None,
    mode: str = "llm",
    max_slices: int | None = None,
    parallel_workers: int = 1,
) -> dict[str, Any]:
    memcell_rows = read_jsonl(memcells_path)
    memcells = [_memcell_from_row(row) for row in memcell_rows]
    resumed_profile_slices = 0
    if mode == "llm" and core_slices_output:
        profile_slices = []
        checkpoint = Path(core_slices_output)
        if checkpoint.exists():
            loaded = json.loads(checkpoint.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError(f"{checkpoint} must contain a JSON array")
            incompatible = [item for item in loaded if isinstance(item, dict) and item.get("_core_mode") != mode]
            if incompatible:
                raise ValueError(f"{checkpoint} contains non-{mode} core slices; use a different output path or remove it")
            profile_slices = loaded
            resumed_profile_slices = len(profile_slices)
        completed_memcell_ids = {
            str(memcell_id)
            for item in profile_slices
            for memcell_id in item.get("_source_memcell_ids", [item.get("memcell_id", "")])
            if memcell_id
        }
        pending_memcells = [memcell for memcell in memcells if memcell.memcell_id not in completed_memcell_ids]
        effective_workers = max(1, int(parallel_workers or 1))
        if max_slices is not None:
            effective_workers = 1
        if effective_workers == 1:
            for memcell in pending_memcells:
                if max_slices is not None and len(profile_slices) >= max_slices:
                    break
                remaining = None if max_slices is None else max_slices - len(profile_slices)
                memcell_id, new_slices = _extract_llm_profile_slices_for_memcell(memcell, max_slices=remaining)
                profile_slices.extend(new_slices)
                completed_memcell_ids.add(memcell_id)
                write_json(checkpoint, profile_slices)
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(_extract_llm_profile_slices_for_memcell, memcell, None): memcell.memcell_id
                    for memcell in pending_memcells
                }
                for future in as_completed(futures):
                    memcell_id, new_slices = future.result()
                    profile_slices.extend(new_slices)
                    completed_memcell_ids.add(memcell_id)
                    write_json(checkpoint, profile_slices)
        merged_core = merge_core_profile_slices(profile_slices)
        core_profiles = [convert_to_core_memory(item) for item in merged_core["not_agent"]]
        core_rows = core_memory_rows(core_profiles)
    else:
        core_rows, profile_slices, merged_core = build_core_memory_from_memcells(memcells, mode=mode, max_slices=max_slices)
    write_jsonl(core_output, core_rows)
    if core_slices_output:
        write_json(core_slices_output, profile_slices)
    if merged_core_output:
        write_json(merged_core_output, merged_core)
    combined_count = None
    if combined_output:
        combined_rows: list[dict[str, Any]] = []
        if episodic_path:
            combined_rows.extend(read_jsonl(episodic_path))
        combined_rows.extend(core_rows)
        write_jsonl(combined_output, combined_rows)
        combined_count = len(combined_rows)
    summary = {
        "memcells": len(memcells),
        "core_memories": len(core_rows),
        "profile_slices": len(profile_slices),
        "resumed_profile_slices": resumed_profile_slices,
        "merged_core_users": len(merged_core.get("not_agent", [])),
        "mode": mode,
        "max_slices": max_slices,
        "parallel_workers": parallel_workers,
        "core_output": str(core_output),
        "core_slices_output": str(core_slices_output) if core_slices_output else None,
        "merged_core_output": str(merged_core_output) if merged_core_output else None,
        "combined_output": str(combined_output) if combined_output else None,
        "combined_memories": combined_count,
    }
    if summary_output:
        write_json(summary_output, summary)
    return summary


def import_evermemos_locomo(
    memcells_dir: str | Path,
    locomo_path: str | Path,
    *,
    memcells_output: str | Path,
    episodic_output: str | Path,
    summary_output: str | Path | None = None,
    limit_conversations: int | None = None,
    keep_fact_embeddings: bool = False,
) -> dict[str, Any]:
    locomo = _locomo_indexes(locomo_path)
    paths = sorted(Path(memcells_dir).glob("memcell_list_conv_*.json"), key=_conversation_index)
    normalized_memcells: list[dict[str, Any]] = []
    episodic_rows: list[dict[str, Any]] = []
    missing_dia_ids = 0
    message_count = 0

    for path in paths:
        conv_index = _conversation_index(path)
        if limit_conversations is not None and conv_index >= limit_conversations:
            continue
        if conv_index not in locomo:
            raise ValueError(f"LoCoMo conversation {conv_index} is not available in {locomo_path}")
        cursor = 0
        memcells = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(memcells, list):
            raise ValueError(f"{path} must contain a JSON array")
        for raw_memcell in memcells:
            memcell = copy.deepcopy(raw_memcell)
            if not keep_fact_embeddings:
                _strip_large_embeddings(memcell)
            event_id = str(memcell.get("event_id") or memcell.get("memcell_id") or "")
            memcell["memcell_id"] = event_id
            memcell["source"] = "evermemos_locomo_baseline"
            memcell["conversation_index"] = conv_index
            memcell["conversation_id"] = f"locomo_{conv_index}"
            original_data = memcell.get("original_data") if isinstance(memcell.get("original_data"), list) else []
            enriched_messages: list[dict[str, Any]] = []
            for original in original_data:
                message_count += 1
                enriched = dict(original)
                matched, cursor = _match_locomo_message(enriched, locomo[conv_index], cursor)
                if matched:
                    for field in ["dia_id", "session", "original_timestamp", "img_url", "blip_caption", "query"]:
                        if field in matched:
                            enriched[field] = matched[field]
                else:
                    missing_dia_ids += 1
                enriched_messages.append(enriched)
            evidence_ids = scoped_locomo_evidence_ids([item.get("dia_id") for item in enriched_messages if item.get("dia_id")], conv_index)
            memcell["original_data"] = enriched_messages
            memcell["evidence_ids"] = evidence_ids
            normalized_memcells.append(memcell)
            episodic_rows.append(_episode_row(memcell))

    summary = {
        "source_memcells_dir": str(memcells_dir),
        "locomo": str(locomo_path),
        "memcells_output": str(memcells_output),
        "episodic_output": str(episodic_output),
        "conversations": len({row["conversation_index"] for row in normalized_memcells}),
        "memcells": len(normalized_memcells),
        "episodic_memories": len(episodic_rows),
        "messages": message_count,
        "messages_missing_dia_id": missing_dia_ids,
        "keep_fact_embeddings": keep_fact_embeddings,
    }
    write_jsonl(memcells_output, normalized_memcells)
    write_jsonl(episodic_output, episodic_rows)
    if summary_output:
        write_json(summary_output, summary)
    return summary
