from __future__ import annotations

CORE_FIELDS = {
    "identity",
    "preference",
    "allergy",
    "goal",
    "constraint",
    "relationship",
}
TOOL_NAMES = {"search_core_memory", "search_episodic_memory"}

STORE_BY_TOOL = {
    "search_core_memory": "core",
    "search_episodic_memory": "episodic",
}


def memory_tool_schema() -> list[dict]:
    return [
        {
            "name": "search_core_memory",
            "description": "检索长期稳定的用户画像和关联人画像。",
            "input_schema": {
                "required": ["queries", "fields"],
                "properties": {
                    "queries": "自然语言检索词数组。每条 query 应聚焦一个画像主体或约束，例如：妈妈 饮食偏好 忌口。",
                    "fields": "需要检索的核心记忆字段列表，取值必须来自 fields 枚举。",
                },
                "fields": sorted(CORE_FIELDS),
            },
        },
        {
            "name": "search_episodic_memory",
            "description": "检索具体事件、近期计划、时间线、地点、同行人和阶段性状态。",
            "input_schema": {
                "required": ["queries"],
                "properties": {
                    "queries": "自然语言检索词数组。每条 query 应包含事件、人物、地点、时间或状态关键词。",
                },
            },
        },
    ]


def search_text(memory: dict) -> str:
    parts = [
        memory.get("content"),
        memory.get("event"),
        memory.get("entity"),
        memory.get("field"),
        memory.get("core_path"),
        memory.get("subject"),
        memory.get("summary"),
        memory.get("time"),
        memory.get("location"),
        memory.get("value"),
        " ".join(memory.get("participants", []) if isinstance(memory.get("participants"), list) else []),
        " ".join(memory.get("memcell_ids", []) if isinstance(memory.get("memcell_ids"), list) else []),
    ]
    return " ".join(str(part) for part in parts if part)
