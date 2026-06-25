#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/locomo/locomo10.json"
DEFAULT_OUTPUT = "data/locomo/locomo10_precess.json"


def load_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: float,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        normalize_endpoint(base_url),
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            is_retryable = exc.code >= 500
            if attempt < 2 and is_retryable:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAI-compatible chat HTTP {exc.code}: {error_body}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAI-compatible chat request failed: {exc}") from exc
    raise RuntimeError("OpenAI-compatible chat request failed after retries")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(cleaned[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError(f"chat response is not a JSON object: {text[:300]}")


def conversation_speakers(item: dict[str, Any]) -> list[str]:
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


def evidence_lookup(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    conversation = item.get("conversation", {})
    if not isinstance(conversation, dict):
        return lookup
    for session_name, messages in conversation.items():
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            dia_id = message.get("dia_id")
            if dia_id:
                row = dict(message)
                row["session"] = session_name
                lookup[str(dia_id)] = row
    return lookup


def compact_evidence(qa: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    evidence = qa.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [evidence]
    rows = []
    for evidence_id in evidence:
        message = lookup.get(str(evidence_id))
        if not message:
            rows.append({"dia_id": str(evidence_id), "speaker": "", "text": ""})
            continue
        rows.append(
            {
                "dia_id": str(evidence_id),
                "speaker": str(message.get("speaker", "")),
                "text": str(message.get("text", ""))[:800],
            }
        )
    return rows


def qa_rows_for_conversation(item: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = evidence_lookup(item)
    qa_rows = []
    for qa_index, qa in enumerate(item.get("qa", [])):
        qa_rows.append(
            {
                "qa_index": qa_index,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "evidence": compact_evidence(qa, lookup),
            }
        )
    return qa_rows


def prompt_for_qa_rows(speakers: list[str], qa_rows: list[dict[str, Any]], conv_index: int) -> list[dict[str, str]]:
    user_payload = {
        "conversation_index": conv_index,
        "candidate_people": speakers,
        "qa": qa_rows,
    }
    system = (
        "You annotate LoCoMo QA data with the primary person being asked about. "
        "For each QA, choose exactly one person from candidate_people. "
        "Use the question first, then answer and evidence if needed. "
        "If multiple candidate people are mentioned, choose the main subject of the answer; "
        "if still tied, choose the first named person in the question. "
        "Do not invent names outside candidate_people. "
        "Return JSON only."
    )
    user = (
        "Return this exact JSON shape:\n"
        '{"items":[{"qa_index":0,"person":"Name"}]}\n\n'
        f"Input:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def is_token_limit_error(exc: Exception) -> bool:
    text = str(exc)
    return "TOKEN_LIMIT_EXCEEDED" in text or "token limit" in text.lower()


def annotate_qa_batch(
    args: argparse.Namespace,
    conv_index: int,
    speakers: list[str],
    qa_rows: list[dict[str, Any]],
    api_key: str,
) -> dict[str, Any]:
    messages = prompt_for_qa_rows(speakers, qa_rows, conv_index)
    raw = chat_completion(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    return parse_json_object(raw)


def annotate_qa_batch_with_fallback(
    args: argparse.Namespace,
    conv_index: int,
    speakers: list[str],
    qa_rows: list[dict[str, Any]],
    api_key: str,
) -> dict[str, Any]:
    try:
        return annotate_qa_batch(args, conv_index, speakers, qa_rows, api_key)
    except RuntimeError as exc:
        if len(qa_rows) <= 1 or not is_token_limit_error(exc):
            raise
        midpoint = max(1, len(qa_rows) // 2)
        left = annotate_qa_batch_with_fallback(args, conv_index, speakers, qa_rows[:midpoint], api_key)
        right = annotate_qa_batch_with_fallback(args, conv_index, speakers, qa_rows[midpoint:], api_key)
        return {
            "items": list(left.get("items", [])) + list(right.get("items", [])),
        }


def apply_annotations(item: dict[str, Any], annotations: dict[str, Any], speakers: list[str]) -> int:
    by_index: dict[int, str] = {}
    for row in annotations.get("items", []):
        if not isinstance(row, dict):
            continue
        try:
            qa_index = int(row.get("qa_index"))
        except (TypeError, ValueError):
            continue
        person = str(row.get("person", "")).strip()
        if person in speakers:
            by_index[qa_index] = person
    count = 0
    for qa_index, qa in enumerate(item.get("qa", [])):
        person = by_index.get(qa_index, "")
        qa["person"] = person
        if person:
            count += 1
    return count


def env_default(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback).strip()


def annotate_conversation(args: argparse.Namespace, conv_index: int, item: dict[str, Any], api_key: str) -> tuple[int, dict[str, Any], int]:
    annotated_item = copy.deepcopy(item)
    speakers = conversation_speakers(annotated_item)
    if not speakers:
        return conv_index, annotated_item, 0
    qa_rows = qa_rows_for_conversation(annotated_item)
    annotations = annotate_qa_batch_with_fallback(args, conv_index, speakers, qa_rows, api_key)
    annotated = apply_annotations(annotated_item, annotations, speakers)
    return conv_index, annotated_item, annotated


def annotate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    load_env(args.env_file)
    api_key = args.api_key or env_default("OPENAI_CHAT_API_KEY") or env_default("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_CHAT_API_KEY or OPENAI_API_KEY, or pass --api-key.")
    if not args.base_url:
        raise RuntimeError("Set OPENAI_CHAT_BASE_URL or OPENAI_BASE_URL, or pass --base-url.")
    if not args.model:
        raise RuntimeError("Set OPENAI_CHAT_MODEL, or rely on the default xopqwen35v35b, or pass --model.")
    data = read_json(args.input)
    if not isinstance(data, list):
        raise ValueError("LoCoMo input must be a JSON array.")

    jobs: list[tuple[int, dict[str, Any]]] = []
    for conv_index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if args.skip_existing and all(isinstance(qa, dict) and qa.get("person") for qa in item.get("qa", [])):
            continue
        jobs.append((conv_index, item))
        if args.limit_conversations is not None and len(jobs) >= args.limit_conversations:
            break

    annotated = 0
    processed = 0
    workers = max(1, args.parallel_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(annotate_conversation, args, conv_index, item, api_key) for conv_index, item in jobs]
        for future in as_completed(futures):
            conv_index, annotated_item, count = future.result()
            data[conv_index] = annotated_item
            annotated += count
            processed += 1
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    write_json(args.output, data)
    return {
        "input": str(args.input),
        "output": str(args.output),
        "conversations": processed,
        "qa_annotated": annotated,
        "parallel_workers": workers,
        "base_url": args.base_url,
        "model": args.model,
    }


def build_parser() -> argparse.ArgumentParser:
    load_env()
    parser = argparse.ArgumentParser(description="Annotate LoCoMo QA rows with a person field using an OpenAI-compatible chat model.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default=env_default("OPENAI_CHAT_BASE_URL") or env_default("OPENAI_BASE_URL"))
    parser.add_argument("--model", default="xopqwen35v35b")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit-conversations", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = annotate_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
