from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env(path: str | Path = ".env", override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OpenAISettings:
    chat_api_key: str
    embedding_api_key: str
    chat_base_url: str
    embedding_base_url: str
    chat_model: str
    embedding_model: str
    embedding_input_format: str
    embedding_max_chars: int
    timeout_seconds: float
    offline: bool

    @classmethod
    def from_env(cls, role: str | None = None) -> "OpenAISettings":
        load_env()
        shared_api_key = os.environ.get("OPENAI_API_KEY", "")
        shared_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        role_prefix = f"OPENAI_{role.upper()}_" if role else ""
        role_api_key = os.environ.get(f"{role_prefix}API_KEY", "") if role_prefix else ""
        role_base_url = os.environ.get(f"{role_prefix}BASE_URL", "") if role_prefix else ""
        role_model = os.environ.get(f"{role_prefix}MODEL", "") if role_prefix else ""
        return cls(
            chat_api_key=role_api_key or os.environ.get("OPENAI_CHAT_API_KEY") or shared_api_key,
            embedding_api_key=os.environ.get("OPENAI_EMBEDDING_API_KEY") or shared_api_key,
            chat_base_url=(role_base_url or os.environ.get("OPENAI_CHAT_BASE_URL") or shared_base_url).rstrip("/"),
            embedding_base_url=(os.environ.get("OPENAI_EMBEDDING_BASE_URL") or shared_base_url).rstrip("/"),
            chat_model=role_model or os.environ.get("OPENAI_CHAT_MODEL", ""),
            embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", ""),
            embedding_input_format=os.environ.get("OPENAI_EMBEDDING_INPUT_FORMAT", "array").strip().lower() or "array",
            embedding_max_chars=int(os.environ.get("OPENAI_EMBEDDING_MAX_CHARS", "0") or "0"),
            timeout_seconds=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60") or "60"),
            offline=env_bool("MEMORY_TOOL_PLANNER_OFFLINE", default=False),
        )

    def require_chat(self) -> None:
        if self.offline:
            return
        if not self.chat_api_key:
            raise RuntimeError("OPENAI_CHAT_API_KEY or OPENAI_API_KEY is required unless MEMORY_TOOL_PLANNER_OFFLINE=1")
        if not self.chat_model:
            raise RuntimeError("OPENAI_CHAT_MODEL is required unless MEMORY_TOOL_PLANNER_OFFLINE=1")

    def require_embedding(self) -> None:
        if self.offline:
            return
        if not self.embedding_api_key:
            raise RuntimeError("OPENAI_EMBEDDING_API_KEY or OPENAI_API_KEY is required unless MEMORY_TOOL_PLANNER_OFFLINE=1")
        if not self.embedding_model:
            raise RuntimeError("OPENAI_EMBEDDING_MODEL is required unless MEMORY_TOOL_PLANNER_OFFLINE=1")
