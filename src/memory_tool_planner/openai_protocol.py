from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import OpenAISettings


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _endpoint_url(base_url: str, path: str) -> str:
    """Support both API roots and full endpoint URLs.

    Examples:
    - base_url=https://host/v1, path=/embeddings -> https://host/v1/embeddings
    - base_url=https://host/v1/embeddings, path=/embeddings -> https://host/v1/embeddings
    """
    normalized_base = base_url.rstrip("/")
    normalized_path = "/" + path.strip("/")
    if normalized_base.endswith(normalized_path):
        return normalized_base
    return _url(normalized_base, normalized_path)


def _redact_error_body(body: str) -> str:
    body = re.sub(r"authorization=.*?(?=(?:\\\\u0026|&|\\\"|\"|\\s))", "authorization=<redacted>", body)
    body = re.sub(r"api[_-]?key[=:\\\\\" ]+[^,}\\s]+", "api_key=<redacted>", body, flags=re.IGNORECASE)
    body = re.sub(r"signature=.*?(?=(?:\\\\u0026|&|\\\"|\"|\\s))", "signature=<redacted>", body, flags=re.IGNORECASE)
    return body


@dataclass
class OpenAIProtocolClient:
    settings: OpenAISettings

    @classmethod
    def from_env(cls, role: str | None = None) -> "OpenAIProtocolClient":
        return cls(OpenAISettings.from_env(role))

    def _post(self, base_url: str, api_key: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(_endpoint_url(base_url, path), data=data, headers=headers, method="POST")
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"OpenAI-compatible API HTTP {exc.code}: {_redact_error_body(body)}")
                if exc.code < 500 or attempt == 3:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"OpenAI-compatible API request failed: {exc}") 
                if attempt == 3:
                    raise last_error from exc
            time.sleep(2**attempt)
        raise RuntimeError(str(last_error))

    def _embedding_payload(self, input_value: str | list[str]) -> dict[str, Any]:
        return {"model": self.settings.embedding_model, "input": input_value}

    def _embedding_rows(self, response: dict[str, Any], expected_count: int) -> list[list[float]]:
        rows = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [row["embedding"] for row in rows]
        if len(embeddings) != expected_count:
            raise RuntimeError(f"embedding response count mismatch: expected {expected_count}, got {len(embeddings)}")
        return embeddings

    def _truncate_embedding_inputs(self, texts: list[str]) -> list[str]:
        max_chars = self.settings.embedding_max_chars
        if max_chars <= 0:
            return texts
        return [text[:max_chars] for text in texts]

    def _embed_array(self, texts: list[str]) -> list[list[float]]:
        response = self._post(
            self.settings.embedding_base_url,
            self.settings.embedding_api_key,
            "/embeddings",
            self._embedding_payload(texts),
        )
        return self._embedding_rows(response, len(texts))

    def _embed_strings(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = self._post(
                self.settings.embedding_base_url,
                self.settings.embedding_api_key,
                "/embeddings",
                self._embedding_payload(text),
            )
            embeddings.extend(self._embedding_rows(response, 1))
        return embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.settings.require_embedding()
        if not texts:
            return []
        texts = self._truncate_embedding_inputs(texts)
        input_format = self.settings.embedding_input_format
        if input_format == "array":
            return self._embed_array(texts)
        if input_format == "string":
            return self._embed_strings(texts)
        if input_format == "auto":
            try:
                return self._embed_array(texts)
            except Exception as array_exc:
                try:
                    return self._embed_strings(texts)
                except Exception as string_exc:
                    raise RuntimeError(f"embedding array request failed: {array_exc}; string fallback failed: {string_exc}") from string_exc
        raise RuntimeError("OPENAI_EMBEDDING_INPUT_FORMAT must be one of: array, string, auto")

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        self.settings.require_chat()
        payload = {"model": self.settings.chat_model, "messages": messages, "temperature": temperature}
        response = self._post(self.settings.chat_base_url, self.settings.chat_api_key, "/chat/completions", payload)
        return response["choices"][0]["message"]["content"]
