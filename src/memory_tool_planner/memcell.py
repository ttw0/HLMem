from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .evidence import infer_locomo_conversation_index, scoped_locomo_evidence_ids, scoped_locomo_evidence_id
from .openai_protocol import OpenAIProtocolClient


CONV_BOUNDARY_DETECTION_PROMPT = """
You are an episodic memory boundary detection expert. Decide whether the newly
added dialogue should end the current episode and start a new one.

Current conversation history:
{conversation_history}

Time gap information:
{time_gap_info}

Newly added messages:
{new_messages}

Analyze substantive topic change, intent transition, meaningful content,
structural or temporal signals, and whether the new content is independent from
the current episode. Ignore greetings, brief acknowledgements, social closures,
and small talk unless they introduce a major new topic. Long time gaps strongly
suggest a new episode. Aim for episodes with 3-20 meaningful exchanges.

Return JSON only:
{{
  "reasoning": "one sentence",
  "should_end": true/false,
  "confidence": 0.0-1.0,
  "topic_summary": "summary of current episode if should_end=true, else blank"
}}
"""


EPISODE_GENERATION_PROMPT = """
You extract an episodic memory from a conversation memcell.

Conversation start time:
{conversation_start_time}

Conversation JSON lines:
{conversation}

Return JSON only with:
- title: short subject of the episode
- content: a self-contained factual episode description
- summary: concise summary
"""


LOCOMO_TIMESTAMP_FORMAT = "%I:%M %p on %d %B, %Y"


@dataclass
class BoundaryDetectionResult:
    should_end: bool
    should_wait: bool
    reasoning: str
    confidence: float
    topic_summary: str = ""


@dataclass
class MemCell:
    memcell_id: str
    type: str
    user_id_list: list[str]
    original_data: list[dict[str, Any]]
    timestamp: str
    summary: str
    participants: list[str] = field(default_factory=list)
    group_id: str | None = None
    episode: str = ""
    subject: str = ""
    extend: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memcell_id": self.memcell_id,
            "event_id": self.memcell_id,
            "type": self.type,
            "user_id_list": self.user_id_list,
            "original_data": self.original_data,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "participants": self.participants,
            "group_id": self.group_id,
            "episode": self.episode,
            "subject": self.subject,
            "extend": self.extend,
        }


def parse_locomo_timestamp(timestamp_str: str) -> datetime:
    """Parse LoCoMo timestamps the same way EverMemos stage1 does."""
    normalized = re.sub(r"\s+", " ", timestamp_str.strip()).upper()
    return datetime.strptime(normalized, LOCOMO_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def isoformat(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def datetime_to_epoch_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(parsed.timestamp() * 1000)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_locomo_conversations(input_path: str | Path, limit_conversations: int | None = None) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    conversations: dict[str, list[dict[str, Any]]] = {}
    for conv_index, item in enumerate(data):
        if limit_conversations is not None and conv_index >= limit_conversations:
            break
        conversation = item.get("conversation", {})
        if not isinstance(conversation, dict):
            continue

        speaker_name_to_id: dict[str, str] = {}
        messages: list[dict[str, Any]] = []
        session_keys = sorted(
            [
                key
                for key in conversation
                if key.startswith("session_") and not key.endswith("_date_time") and isinstance(conversation.get(key), list)
            ],
            key=lambda key: int(key.replace("session_", "")),
        )
        for session_key in session_keys:
            session_time_key = f"{session_key}_date_time"
            if session_time_key not in conversation:
                continue
            session_start = parse_locomo_timestamp(str(conversation[session_time_key]))
            for turn_index, msg in enumerate(conversation.get(session_key, [])):
                speaker_name = str(msg.get("speaker", "Unknown")).strip() or "Unknown"
                if speaker_name not in speaker_name_to_id:
                    speaker_name_to_id[speaker_name] = f"{speaker_name.lower().replace(' ', '_')}_{conv_index}"
                content = str(msg.get("text", "")).strip()
                if msg.get("img_url"):
                    caption = msg.get("blip_caption") or "an image"
                    content = f"[{speaker_name} shared an image: {caption}] {content}".strip()
                timestamp = isoformat(session_start + timedelta(seconds=turn_index * 30))
                raw_dia_id = str(msg.get("dia_id", f"{session_key}:{turn_index + 1}"))
                message = {
                    "speaker_id": speaker_name_to_id[speaker_name],
                    "user_name": speaker_name,
                    "speaker_name": speaker_name,
                    "content": content,
                    "timestamp": timestamp,
                    "original_timestamp": conversation[session_time_key],
                    "dia_id": scoped_locomo_evidence_id(raw_dia_id, conv_index),
                    "raw_dia_id": raw_dia_id,
                    "session": session_key,
                }
                for optional in ["img_url", "blip_caption", "query"]:
                    if optional in msg:
                        message[optional] = msg[optional]
                if content:
                    messages.append(message)
        conversations[str(conv_index)] = messages
    return conversations


def build_temporal_context(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    timestamps = [msg.get("timestamp") for msg in messages if msg.get("timestamp")]
    if not timestamps:
        now = isoformat(datetime.now(timezone.utc))
        return now, {"start_time": now, "end_time": now, "message_count": len(messages)}
    return timestamps[0], {"start_time": timestamps[0], "end_time": timestamps[-1], "message_count": len(messages)}


def participant_ids(messages: list[dict[str, Any]]) -> list[str]:
    return sorted({str(msg.get("speaker_id")) for msg in messages if msg.get("speaker_id")})


def format_conversation(messages: list[dict[str, Any]], include_timestamps: bool = True) -> str:
    lines = []
    for msg in messages:
        speaker = msg.get("speaker_name") or msg.get("speaker_id") or "Unknown"
        content = msg.get("content", "")
        if include_timestamps and msg.get("timestamp"):
            lines.append(f"[{msg['timestamp']}] {speaker}: {content}")
        else:
            lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def time_gap_info(history: list[dict[str, Any]], new_messages: list[dict[str, Any]]) -> str:
    if not history or not new_messages:
        return "No time gap information available"
    try:
        last_time = datetime.fromisoformat(str(history[-1]["timestamp"]).replace("Z", "+00:00"))
        next_time = datetime.fromisoformat(str(new_messages[0]["timestamp"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return "No timestamp information available"
    seconds = (next_time - last_time).total_seconds()
    if seconds < 0:
        return "Time gap: Messages appear to be out of order"
    if seconds < 60:
        return f"Time gap: {int(seconds)} seconds (immediate response)"
    if seconds < 3600:
        return f"Time gap: {int(seconds // 60)} minutes (recent conversation)"
    if seconds < 86400:
        return f"Time gap: {int(seconds // 3600)} hours (same day, but significant pause)"
    return f"Time gap: {int(seconds // 86400)} days (long gap, likely new conversation)"


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def detect_boundary_llm(
    history: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    client: OpenAIProtocolClient,
) -> BoundaryDetectionResult:
    if not history:
        return BoundaryDetectionResult(False, False, "First messages in conversation", 1.0, "")
    prompt = CONV_BOUNDARY_DETECTION_PROMPT.format(
        conversation_history=format_conversation(history, include_timestamps=True),
        time_gap_info=time_gap_info(history, new_messages),
        new_messages=format_conversation(new_messages, include_timestamps=True),
    )
    content = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    data = parse_json_object(content)
    return BoundaryDetectionResult(
        should_end=bool(data.get("should_end", False)),
        should_wait=not bool(data.get("should_end", False)),
        reasoning=str(data.get("reasoning", "")),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        topic_summary=str(data.get("topic_summary", "")),
    )


def detect_boundary_heuristic(history: list[dict[str, Any]], new_messages: list[dict[str, Any]]) -> BoundaryDetectionResult:
    if not history:
        return BoundaryDetectionResult(False, False, "First messages in conversation", 1.0, "")
    gap = time_gap_info(history, new_messages)
    if "days" in gap or "hours" in gap:
        return BoundaryDetectionResult(True, False, gap, 0.8, summarize_messages(history))
    if len(history) >= 20:
        return BoundaryDetectionResult(True, False, "Message count reached heuristic episode limit", 0.7, summarize_messages(history))
    prev_session = history[-1].get("session")
    next_session = new_messages[0].get("session")
    if prev_session and next_session and prev_session != next_session and len(history) >= 3:
        return BoundaryDetectionResult(True, False, "LoCoMo session boundary", 0.85, summarize_messages(history))
    return BoundaryDetectionResult(False, True, "Continue current episode", 0.6, "")


def summarize_messages(messages: list[dict[str, Any]], max_chars: int = 160) -> str:
    text = " ".join(str(msg.get("content", "")) for msg in messages if msg.get("content"))
    return text[:max_chars].strip() or "Conversation segment"


def generate_episode(memcell: MemCell, mode: str = "heuristic", client: OpenAIProtocolClient | None = None) -> tuple[str, str, str]:
    if mode == "llm":
        if client is None:
            client = OpenAIProtocolClient.from_env()
        prompt = EPISODE_GENERATION_PROMPT.format(
            conversation_start_time=memcell.timestamp,
            conversation="\n".join(json.dumps(item, ensure_ascii=False) for item in memcell.original_data),
        )
        data = parse_json_object(client.chat([{"role": "user", "content": prompt}], temperature=0.0))
        title = str(data.get("title") or "Conversation Episode")
        content = str(data.get("content") or summarize_messages(memcell.original_data, 500))
        summary = str(data.get("summary") or content[:200])
        return title, content, summary
    title = memcell.summary if memcell.summary and memcell.summary != "Final segment" else summarize_messages(memcell.original_data, 80)
    content = format_conversation(memcell.original_data, include_timestamps=True)
    summary = summarize_messages(memcell.original_data, 200)
    return title, content, summary


def make_memcell(
    messages: list[dict[str, Any]],
    conv_id: str,
    index: int,
    summary: str,
) -> MemCell:
    timestamp, temporal_context = build_temporal_context(messages)
    ids = participant_ids(messages)
    return MemCell(
        memcell_id=f"memcell_{conv_id}_{index:04d}_{uuid.uuid4().hex[:8]}",
        type="conversation",
        user_id_list=ids,
        original_data=list(messages),
        timestamp=timestamp,
        summary=summary,
        participants=ids,
        extend={
            "temporal_context": temporal_context,
            "source": "locomo",
            "conversation_index": int(conv_id),
            "conversation_id": f"locomo_{conv_id}",
        },
    )


def extract_memcells_from_messages(
    messages: list[dict[str, Any]],
    conv_id: str,
    boundary_mode: str = "heuristic",
    episode_mode: str = "heuristic",
    client: OpenAIProtocolClient | None = None,
    smart_mask: bool = True,
    hard_message_limit: int = 50,
    hard_token_limit: int = 8192,
) -> list[MemCell]:
    if boundary_mode == "llm" or episode_mode == "llm":
        client = client or OpenAIProtocolClient.from_env()

    memcells: list[MemCell] = []
    history: list[dict[str, Any]] = []
    for raw_message in messages:
        if len(history) < 2:
            history.append(raw_message)
            continue

        approx_tokens = len(format_conversation(history + [raw_message], include_timestamps=True)) // 4
        needs_force_split = len(history) + 1 >= hard_message_limit or approx_tokens >= hard_token_limit
        if needs_force_split and len(history) >= 2:
            boundary = BoundaryDetectionResult(True, False, "Force split limit", 1.0, summarize_messages(history))
        else:
            boundary_history = history[:-1] if smart_mask and len(history) > 5 else history
            if boundary_mode == "llm":
                boundary = detect_boundary_llm(boundary_history, [raw_message], client)  # type: ignore[arg-type]
            else:
                boundary = detect_boundary_heuristic(boundary_history, [raw_message])

        if boundary.should_end:
            memcell = make_memcell(history, conv_id, len(memcells) + 1, boundary.topic_summary or summarize_messages(history))
            subject, episode, summary = generate_episode(memcell, mode=episode_mode, client=client)
            memcell.subject = subject
            memcell.episode = episode
            memcell.summary = summary
            memcells.append(memcell)
            history = [raw_message]
        else:
            history.append(raw_message)

    if history:
        memcell = make_memcell(history, conv_id, len(memcells) + 1, "Final segment")
        subject, episode, summary = generate_episode(memcell, mode=episode_mode, client=client)
        memcell.subject = subject
        memcell.episode = episode
        memcell.summary = summary
        memcells.append(memcell)
    return memcells


def memcells_to_episodic_rows(memcells: list[MemCell]) -> list[dict[str, Any]]:
    rows = []
    for memcell in memcells:
        temporal = memcell.extend.get("temporal_context", {})
        conversation_index = infer_locomo_conversation_index(memcell.extend) or infer_locomo_conversation_index(
            memcell.memcell_id
        )
        evidence_ids = scoped_locomo_evidence_ids(
            [msg.get("dia_id", "") for msg in memcell.original_data if msg.get("dia_id")],
            conversation_index,
        )
        rows.append(
            {
                "store": "episodic",
                "memory_id": f"epi_{memcell.memcell_id}",
                "memcell_id": memcell.memcell_id,
                "conversation_id": f"locomo_{conversation_index}" if conversation_index is not None else "",
                "conversation_index": conversation_index,
                "event": memcell.episode or memcell.summary,
                "subject": memcell.subject,
                "summary": memcell.summary,
                "time": temporal.get("start_time", memcell.timestamp),
                "time_range": temporal,
                "location": "",
                "participants": memcell.participants,
                "status": "completed",
                "evidence_ids": evidence_ids,
            }
        )
    return rows
