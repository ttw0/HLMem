from __future__ import annotations

import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .builder import read_jsonl
from .embeddings import EmbeddingProvider, cosine
from .evidence import row_conversation_index, with_scoped_row_evidence
from .planner import validate_planner_output
from .schema import STORE_BY_TOOL, search_text


def keyword_score(query: str, text: str) -> float:
    q_tokens = {token for token in query.lower().split() if token}
    if not q_tokens:
        return 0.0
    lowered = text.lower()
    return sum(1.0 for token in q_tokens if token in lowered) / len(q_tokens)


def format_memory_for_tool(tool_name: str, memory: dict[str, Any]) -> dict[str, Any]:
    evidence_ids = memory.get("evidence_ids", [])
    if tool_name == "search_core_memory":
        return {
            "memory_id": memory.get("memory_id", ""),
            "memcell_ids": memory.get("memcell_ids", []),
            "content": memory.get("content", search_text(memory)),
            "entity": memory.get("entity", ""),
            "field": memory.get("field", ""),
            "core_path": memory.get("core_path", ""),
            "aspect": memory.get("aspect", ""),
            "value": memory.get("value", ""),
            "confidence": memory.get("confidence", 0.0),
            "updated_at": memory.get("updated_at", ""),
            "evidence_ids": evidence_ids,
            "score": memory.get("_score", 0.0),
        }
    return {
        "memory_id": memory.get("memory_id", ""),
        "memcell_id": memory.get("memcell_id", ""),
        "event": memory.get("event", search_text(memory)),
        "subject": memory.get("subject", ""),
        "summary": memory.get("summary", ""),
        "time": memory.get("time", ""),
        "time_range": memory.get("time_range", {}),
        "location": memory.get("location", ""),
        "participants": memory.get("participants", []),
        "status": memory.get("status", ""),
        "evidence_ids": evidence_ids,
        "score": memory.get("_score", 0.0),
    }


def _filter_candidates(
    tool_name: str,
    memories: list[dict[str, Any]],
    arguments: dict[str, Any],
    conversation_index: int | None = None,
) -> list[dict[str, Any]]:
    if conversation_index is not None:
        memories = [memory for memory in memories if row_conversation_index(memory) == conversation_index]
    if tool_name != "search_core_memory":
        return memories
    fields = arguments.get("fields", [])
    field_set = {field for field in fields if isinstance(field, str)}
    return [memory for memory in memories if not field_set or memory.get("field") in field_set]


def _effective_top_k(arguments: dict[str, Any]) -> int:
    fixed = os.environ.get("MTP_RETRIEVAL_TOP_K")
    if fixed:
        return max(1, int(fixed))
    return int(arguments["top_k"])


def _argument_queries(tool_name: str, arguments: dict[str, Any]) -> list[str]:
    queries = arguments.get("queries", [])
    if not isinstance(queries, list):
        return []
    return [query.strip() for query in queries if isinstance(query, str) and query.strip()]


@dataclass
class MemoryToolExecutor:
    memories: list[dict[str, Any]]
    embedding_provider: EmbeddingProvider

    @classmethod
    def from_jsonl(cls, path: str | Path, embedding_provider: EmbeddingProvider | None = None) -> "MemoryToolExecutor":
        return cls(
            [with_scoped_row_evidence(row) for row in read_jsonl(path)],
            embedding_provider or EmbeddingProvider.from_env(),
        )

    def execute_plan(self, planner_output: dict[str, Any], *, conversation_index: int | None = None) -> dict[str, Any]:
        validation = validate_planner_output(planner_output)
        if not validation.valid:
            return {
                "tool_results": [
                    {
                        "call_id": "",
                        "name": "",
                        "status": "error",
                        "latency_ms": 0,
                        "error": {"code": "INVALID_PLAN", "message": "planner output is invalid", "details": {"errors": validation.errors}},
                    }
                ]
            }
        executable_plan = validation.parsed or {"tool_calls": []}
        results = []
        for call in executable_plan.get("tool_calls", []):
            results.append(self.execute_call(call, conversation_index=conversation_index))
        return {"tool_results": results}

    def execute_call(self, call: dict[str, Any], *, conversation_index: int | None = None) -> dict[str, Any]:
        started = time.time()
        name = call["name"]
        args = call["arguments"]
        try:
            result = self.search(name, args, conversation_index=conversation_index)
            return {
                "call_id": call["call_id"],
                "name": name,
                "status": "ok",
                "latency_ms": int((time.time() - started) * 1000),
                "result": result,
            }
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            return {
                "call_id": call.get("call_id", ""),
                "name": name,
                "status": "error",
                "latency_ms": int((time.time() - started) * 1000),
                "error": {"code": "EXECUTION_ERROR", "message": str(exc), "details": {}},
            }

    def search(self, tool_name: str, arguments: dict[str, Any], *, conversation_index: int | None = None) -> dict[str, Any]:
        store = STORE_BY_TOOL[tool_name]
        candidates = [memory for memory in self.memories if memory.get("store") == store]
        candidates = _filter_candidates(tool_name, candidates, arguments, conversation_index=conversation_index)
        queries = _argument_queries(tool_name, arguments)
        query_embeddings = self.embedding_provider.embed(queries) if queries else []
        scored = []
        for memory in candidates:
            text = memory.get("search_text") or search_text(memory)
            memory_embedding = memory.get("embedding", [])
            best_score = 0.0
            best_vector_score = 0.0
            best_lexical_score = 0.0
            best_query = ""
            for query, query_embedding in zip(queries, query_embeddings, strict=False):
                lexical_score = keyword_score(query, text)
                if memory_embedding and len(query_embedding) == len(memory_embedding):
                    vector_score = cosine(query_embedding, memory_embedding)
                    score = vector_score + lexical_score
                else:
                    vector_score = 0.0
                    score = lexical_score
                if score > best_score:
                    best_score = score
                    best_vector_score = vector_score
                    best_lexical_score = lexical_score
                    best_query = query
            item = dict(memory)
            item["_score"] = best_score
            item["_vector_score"] = best_vector_score
            item["_lexical_score"] = best_lexical_score
            item["_matched_query"] = best_query
            scored.append(item)
        top_k = _effective_top_k(arguments)
        ranked = sorted(scored, key=lambda item: item.get("_score", 0.0), reverse=True)[:top_k]
        return {
            "queries": queries,
            "memories": [format_memory_for_tool(tool_name, memory) for memory in ranked if memory.get("_score", 0.0) > 0.0],
        }
