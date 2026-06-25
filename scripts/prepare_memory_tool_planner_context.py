#!/usr/bin/env python3
"""Prepare a self-contained context pack for the memory tool planner experiment.

This script is intentionally non-interactive. It discovers what it can from
environment variables and conventional paths, writes defaults for dry-run, and
records missing training requirements as blockers instead of failing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOC_NAMES = [
    "记忆工具规划强化学习实验规格.md",
    "记忆工具规划强化学习实验实现说明.md",
    "记忆工具规划强化学习实验环境清单.md",
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def existing_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path) if path.exists() else None


def discover_dataset(name: str, env_var: str, candidates: list[Path]) -> dict[str, Any]:
    env_path = existing_path(os.environ.get(env_var))
    checked = []
    if env_path:
        return {"enabled": True, "source": "env", "path": env_path, "checked": [env_var]}

    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return {"enabled": True, "source": "convention", "path": str(candidate), "checked": checked}

    return {"enabled": False, "source": None, "path": None, "checked": checked}


def discover_data(root: Path) -> dict[str, Any]:
    data_root = existing_path(os.environ.get("MEMORY_TOOL_PLANNER_DATA_ROOT"))
    candidate_root = Path(data_root) if data_root else root / "data"
    parent = root.parent
    return {
        "locomo": discover_dataset(
            "locomo",
            "LOCOMO_DATA_ROOT",
            [
                candidate_root / "locomo",
                root / "data" / "LoCoMo",
                parent / "data" / "locomo",
                parent / "memoryeval" / "data" / "locomo",
            ],
        ),
        "personamem": discover_dataset(
            "personamem",
            "PERSONAMEM_DATA_ROOT",
            [
                candidate_root / "personamem",
                root / "data" / "PersonaMem",
                parent / "data" / "personamem",
                parent / "memoryeval" / "data" / "personamem",
            ],
        ),
    }


def discover_models() -> dict[str, Any]:
    api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    openai_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    chat_api_key_present = bool(os.environ.get("OPENAI_CHAT_API_KEY")) or api_key_present
    embedding_api_key_present = bool(os.environ.get("OPENAI_EMBEDDING_API_KEY")) or api_key_present
    chat_base = os.environ.get("OPENAI_CHAT_BASE_URL") or openai_base
    embedding_base = os.environ.get("OPENAI_EMBEDDING_BASE_URL") or openai_base
    vllm_base = os.environ.get("VLLM_BASE_URL")
    planner_path = existing_path(os.environ.get("PLANNER_MODEL_PATH"))
    answer_path = existing_path(os.environ.get("ANSWER_MODEL_PATH"))
    judge_path = existing_path(os.environ.get("JUDGE_MODEL_PATH"))

    if chat_api_key_present or embedding_api_key_present or chat_base or embedding_base:
        serving_mode = "openai_compatible_api"
    elif vllm_base:
        serving_mode = "vllm"
    elif planner_path or answer_path:
        serving_mode = "local_model_path"
    else:
        serving_mode = "unconfigured"

    return {
        "serving_mode": serving_mode,
        "openai_api_key_present": api_key_present,
        "api_base": openai_base,
        "chat_api_key_present": chat_api_key_present,
        "chat_base_url": chat_base,
        "chat_model": os.environ.get("OPENAI_CHAT_MODEL"),
        "embedding_api_key_present": embedding_api_key_present,
        "embedding_base_url": embedding_base,
        "embedding_model": os.environ.get("OPENAI_EMBEDDING_MODEL"),
        "vllm_base_url": vllm_base,
        "planner_model_path": planner_path,
        "answer_model_path": answer_path,
        "judge_model_path": judge_path,
        "tokenizer_name_or_path": os.environ.get("TOKENIZER_NAME_OR_PATH") or planner_path,
    }


def discover_training_frameworks() -> dict[str, Any]:
    frameworks = {
        "trl": module_available("trl"),
        "verl": module_available("verl"),
        "openrlhf": module_available("openrlhf"),
    }
    if frameworks["verl"]:
        grpo = "verl"
    elif frameworks["trl"]:
        grpo = "trl_or_custom"
    elif frameworks["openrlhf"]:
        grpo = "openrlhf_or_custom"
    else:
        grpo = "offline_reward_runner_only"
    return {"available": frameworks, "sft_framework": "trl" if frameworks["trl"] else "custom_or_unconfigured", "grpo_framework": grpo}


def prompt_templates() -> dict[str, str]:
    return {
        "planner.md": """You are a Memory Tool Planner.

Return only valid JSON with exactly one top-level key: tool_calls.
Do not answer the user question.
Do not invent memory content.
Every tool call must include name and arguments. Do not output call_id, purpose, or top_k.
The arguments must match the selected memory tool schema.
If no personal memory is needed, return {"tool_calls": []}.
""",
        "answer.md": """You answer the user question using the provided memory context only as personalization evidence.

If memory_context is empty, do not claim to know user preferences or history.
You may use general world knowledge for non-personal facts, but must not fabricate user history.
""",
        "judge.md": """Judge whether the predicted answer correctly answers the question given the gold answer.

Return only JSON: {"correct": true} or {"correct": false}.
Use a binary judgment. Do not return a continuous score.
""",
    }


def smoke_cases() -> list[dict[str, Any]]:
    return [
        {"case_id": "smoke_001_no_memory", "question": "北京有哪些传统小吃？", "current_date": "2026-05-08", "expected": {"tool_calls": []}},
        {"case_id": "smoke_002_core_memory", "question": "我和妈妈出去吃饭，有什么需要避开的？", "current_date": "2026-05-08", "expected_tools": ["search_core_memory"]},
        {"case_id": "smoke_003_relative_time", "question": "我明天和妈妈去北京，有什么好吃的？", "current_date": "2026-05-08", "expected_query_terms": ["明天", "北京"]},
        {"case_id": "smoke_004_repeat_tool_allowed", "question": "我明天和妈妈去北京，有什么好吃的？", "current_date": "2026-05-08", "expected": "two distinct search_episodic_memory queries allowed"},
        {"case_id": "smoke_005_empty_result", "tool_result": {"memories": []}, "expected": "do not inject pseudo memory"},
        {"case_id": "smoke_006_invalid_queries", "planner_output": {"tool_calls": [{"name": "search_core_memory", "arguments": {"queries": [], "fields": ["preference"]}}]}, "expected_error": "INVALID_ARGUMENTS"},
        {"case_id": "smoke_007_unknown_tool", "planner_output": {"tool_calls": [{"name": "search_social_memory", "arguments": {}}]}, "expected_error": "UNKNOWN_TOOL"},
        {"case_id": "smoke_008_duplicate_query", "expected_error": "DUPLICATE_CALL"},
        {"case_id": "smoke_009_evidence_reward", "retrieved_evidence_ids": ["e1", "e2"], "gold_evidence_ids": ["e2", "e3"], "expected_evidence_reward": 0.5},
        {"case_id": "smoke_010_context_budget", "expected": "dedupe by memory_id and keep memory_context <= 2000 tokens"},
    ]


def sample_memory_stores() -> list[dict[str, Any]]:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_blockers(data: dict[str, Any], models: dict[str, Any], frameworks: dict[str, Any], docs_ok: bool) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not docs_ok:
        blockers.append({"id": "missing_core_docs", "severity": "dry_run", "message": "Required experiment documents are missing."})
    if not data["locomo"]["enabled"]:
        blockers.append({"id": "missing_locomo_data", "severity": "training", "message": "LOCOMO_DATA_ROOT not found; LoCoMo training disabled."})
    if not data["personamem"]["enabled"]:
        blockers.append({"id": "missing_personamem_data", "severity": "training", "message": "PERSONAMEM_DATA_ROOT not found; PersonaMem training disabled."})
    chat_ready = bool(models.get("chat_base_url") and models.get("chat_model") and models.get("chat_api_key_present"))
    embedding_ready = bool(models.get("embedding_base_url") and models.get("embedding_model") and models.get("embedding_api_key_present"))
    local_ready = bool(models.get("planner_model_path") or models.get("answer_model_path"))
    if not (chat_ready or local_ready):
        blockers.append({"id": "missing_chat_model_serving", "severity": "training", "message": "Chat/LLM endpoint, API key, or model name is missing."})
    if not embedding_ready:
        blockers.append({"id": "missing_embedding_model_serving", "severity": "training", "message": "Embedding endpoint, API key, or model name is missing."})
    if frameworks["grpo_framework"] == "offline_reward_runner_only":
        blockers.append({"id": "missing_grpo_framework", "severity": "training", "message": "No GRPO-capable framework discovered; offline reward runner only."})
    return blockers


def status_from(blockers: list[dict[str, str]], data: dict[str, Any], models: dict[str, Any], frameworks: dict[str, Any]) -> str:
    if any(item["severity"] == "dry_run" for item in blockers):
        return "blocked"
    has_data = data["locomo"]["enabled"] or data["personamem"]["enabled"]
    has_model = not any(item["id"] in {"missing_chat_model_serving", "missing_embedding_model_serving"} for item in blockers)
    has_sft = frameworks["sft_framework"] != "custom_or_unconfigured"
    if has_data and has_model and has_sft:
        return "training_ready"
    return "dry_run_ready"


def copy_docs(root: Path, out: Path) -> bool:
    docs_dir = out / "docs"
    ensure_dir(docs_dir)
    ok = True
    for name in DOC_NAMES:
        src = root / "paper" / name
        if src.exists():
            shutil.copy2(src, docs_dir / name)
        else:
            ok = False
    return ok


def write_config(out: Path, discovered: dict[str, Any]) -> None:
    config = {
        "experiment_name": "memory_tool_planner_v0",
        "status": discovered["status"],
        "max_tool_calls": 4,
        "max_calls_per_tool": 2,
        "memory_token_budget": 0,
        "default_top_k": {"core": 5, "episodic": 5},
        "data": discovered["data"],
        "models": discovered["models"],
        "training_frameworks": discovered["training_frameworks"],
        "output_root": str(out / "outputs"),
    }
    write_json(out / "config" / "v0.auto.json", config)


def write_readme(out: Path, discovered: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    blocker_lines = "\n".join(f"- [{b['severity']}] {b['id']}: {b['message']}" for b in blockers) or "- None"
    content = f"""# Memory Tool Planner Context Pack

Generated at: {datetime.now(timezone.utc).isoformat()}

Status: `{discovered['status']}`

## What This Pack Contains

- Core experiment docs under `docs/`
- Auto-discovered environment under `env.discovered.json`
- Blocker report under `blockers.json`
- Default prompts under `prompts/`
- Smoke cases and sample memory stores under `data/`
- Auto config under `config/v0.auto.json`

## Blockers

{blocker_lines}

## Next Step

Read `agent_handoff.md`.
"""
    write_text(out / "README_CONTEXT_PACK.md", content)


def write_handoff(out: Path, discovered: dict[str, Any]) -> None:
    if discovered["status"] == "training_ready":
        next_steps = "Start with schema validation and smoke tests, then connect the discovered dataset and model serving configuration."
    else:
        next_steps = "Start with schema validation, local JSON memory executor, context builder, reward calculator, and smoke tests. Do not start real SFT/GRPO training until training blockers are resolved."
    content = f"""# Agent Handoff

You are receiving a self-contained context pack for the Memory Tool Planner experiment.

Current status: `{discovered['status']}`

## Required Reading

1. `docs/记忆工具规划强化学习实验规格.md`
2. `docs/记忆工具规划强化学习实验实现说明.md`
3. `docs/记忆工具规划强化学习实验环境清单.md`

## Start Here

{next_steps}

## Use These Files

- Config: `config/v0.auto.json`
- Prompts: `prompts/`
- Smoke cases: `data/smoke_cases.jsonl`
- Sample memory stores: `data/sample_memory_stores.jsonl`
- Blockers: `blockers.json`

## Rules

- Do not ask interactive questions during context preparation.
- Missing training inputs are blockers, not fatal errors for dry-run.
- Empty memory results must remain `{{"memories":[]}}`.
"""
    write_text(out / "agent_handoff.md", content)


def prepare(output: Path) -> int:
    root = repo_root()
    load_env_file(root / ".env")
    out = output.resolve()
    ensure_dir(out)
    ensure_dir(out / "outputs")

    docs_ok = copy_docs(root, out)
    data = discover_data(root)
    models = discover_models()
    frameworks = discover_training_frameworks()
    blockers = build_blockers(data, models, frameworks, docs_ok)
    discovered = {
        "status": status_from(blockers, data, models, frameworks),
        "data": data,
        "models": models,
        "training_frameworks": frameworks,
        "prompts": {"generated": True, "path": str(out / "prompts")},
        "outputs": {"root": str(out / "outputs")},
    }

    for filename, content in prompt_templates().items():
        write_text(out / "prompts" / filename, content)
    write_jsonl(out / "data" / "smoke_cases.jsonl", smoke_cases())
    write_jsonl(out / "data" / "sample_memory_stores.jsonl", sample_memory_stores())
    write_json(out / "env.discovered.json", discovered)
    write_json(out / "blockers.json", {"blockers": blockers})
    write_config(out, discovered)
    write_readme(out, discovered, blockers)
    write_handoff(out, discovered)

    print(json.dumps({"context_pack": str(out), "status": discovered["status"], "blockers": len(blockers)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Memory Tool Planner context pack.")
    parser.add_argument("--output", default="outputs/context_pack/latest", help="Output context pack directory.")
    args = parser.parse_args()
    return prepare(Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
