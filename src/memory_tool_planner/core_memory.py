from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .evidence import infer_locomo_conversation_index, scoped_locomo_evidence_ids
from .memcell import MemCell, datetime_to_epoch_ms, parse_json_object
from .openai_protocol import OpenAIProtocolClient


KEEP_DIMENSIONS = {
    "认知分析": {
        "抽象程度": {"具体", "抽象"},
        "复杂程度": {"详细", "简洁"},
        "探索程度": {"多样", "收敛"},
    },
    "行为分析": {
        "决策周期": {"长期", "即时"},
        "风险偏好": {"保守", "冒险"},
    },
}


CORE_FACT_FIELDS = {"identity", "preference", "allergy", "goal", "constraint", "relationship"}


CORE_PROFILE_PROMPT = """
你是 LoCoMo 长对话记忆抽取专家。核心任务是从一个对话切片中抽取可用于回答 LoCoMo 问答的长期稳定核心事实。

输入是一个 JSON：
- uid: 用户唯一标识
- user_name: 用户显示名称
- memcell_id: 对话切片标识
- original_data: 按时间排序的多轮数据，每条包含 time、sid、query、answer

输出纯 JSON，必须包含 uid、memcell_id、核心事实、对话分析。字段结构：
{
  "uid": "字符串",
  "memcell_id": "字符串",
  "核心事实": [
    {
      "entity": "事实主体，优先使用人名；未知时使用 user_name 或 uid",
      "field": "identity/preference/allergy/goal/constraint/relationship",
      "aspect": "细分维度，例如 gender_identity/origin/location/relationship_status/career_goal/education_goal/family/friend",
      "value": "稳定事实值",
      "content": "一句自包含英文事实，必须能脱离上下文理解",
      "evidence_ids": ["来自 original_data.sid 的证据 id"],
      "confidence": 0.0
    }
  ],
  "对话分析": {
    "人格分析": {
      "认知分析": {
        "抽象程度": {"name": "抽象/具体", "value": "0.00~1.00", "证据说明": "字符串"},
        "复杂程度": {"name": "详细/简洁", "value": "0.00~1.00", "证据说明": "字符串"},
        "探索程度": {"name": "多样/收敛", "value": "0.00~1.00", "证据说明": "字符串"}
      },
      "行为分析": {
        "风险偏好": {"name": "冒险/保守", "value": "0.00~1.00", "证据说明": "字符串"},
        "决策周期": {"name": "长期/即时", "value": "0.00~1.00", "证据说明": "字符串"}
      }
    },
    "基础画像": {
      "基础资料": {},
      "教育背景": {},
      "专业技能": [],
      "职业背景": {},
      "消费分析": [],
      "查询实体": [],
      "生活习惯": []
    }
  }
}

核心事实字段定义：
- identity: 身份、稳定状态、来源地、居住地、职业/教育身份、关系状态等。
- preference: 长期偏好、兴趣、喜好、常见倾向。
- allergy: 过敏、忌口、健康禁忌。
- goal: 长期目标、职业目标、教育目标、计划追求的领域。
- constraint: 稳定限制、必须遵守的条件、长期健康或生活约束。
- relationship: 与家人、朋友、同事、伴侣、导师等人的稳定关系。

只抽取长期稳定事实，不抽取一次性事件、具体时间问题、临时计划、单次活动进展。
只从该用户本人 query 中抽取该用户核心事实；他人、假设、虚拟角色不能写入该用户事实。
若 query 明确描述该用户与他人的关系，可以抽取 relationship。
每条核心事实必须带 evidence_ids，且 id 必须来自输入 original_data.sid。
无依据字段删除或输出空对象/空数组。不要输出解释性文本。

对话切片：
{chat_list}
"""


CORE_PROFILE_REPAIR_PROMPT = """
你是严格 JSON 修复器。下面是一段 LoCoMo core profile 抽取结果，但它不是合法 JSON。

要求：
1. 只输出一个合法 JSON object，不输出 Markdown 或解释。
2. 保留原始语义，不新增没有依据的事实。
3. 所有字符串必须使用 JSON 双引号，并正确转义字符串内部双引号。
4. 输出字段必须包含 uid、memcell_id、核心事实、对话分析。

待修复文本：
{invalid_json}
"""


CORE_KEYWORDS = {
    "考研",
    "升学",
    "就业",
    "考公",
    "留学",
    "高考",
    "分数线",
    "院校",
    "专业",
    "毕业",
    "实习",
    "招聘",
    "面试",
    "考证",
    "学历",
    "本科",
    "研究生",
    "录取",
    "报考",
    "职业",
    "工作",
    "求职",
    "薪资",
    "课程",
    "考试",
    "学校",
    "培训",
    "学习",
    "school",
    "college",
    "university",
    "exam",
    "job",
    "work",
    "career",
    "study",
    "class",
    "course",
    "python",
    "coding",
    "travel",
    "food",
    "health",
    "kids",
    "family",
}


QUANT_PATTERNS = {
    "年份": re.compile(r"(19|20)\d{2}\s*[年届级]|近\s*\d+\s*年|\b(19|20)\d{2}\b"),
    "分数": re.compile(r"\d+\.?\d*\s*分(?!析|享|钟|工|配|类)|分数线|录取分|score|points"),
    "人数": re.compile(r"\d+\.?\d*\s*(?:万|千)?\s*(?:人|名|位)|招生\s*\d+|\d+\s+people"),
    "比率": re.compile(r"\d+\.?\d*\s*%|百分之\s*\d+|录取率|通过率|rate|ratio"),
    "薪资": re.compile(r"月薪|年薪|薪[资酬待]|\d+\.?\d*\s*(?:万|千)?\s*元|salary|\$"),
    "排名": re.compile(r"(?:前|第)\s*\d+\s*(?:名|位|档)|[Tt]op\s*\d+|985|211|\d+强|rank"),
    "时长": re.compile(r"\d+\s*个月|\d+\s*(?:周|天|小时)|\d+\s*(?:weeks?|days?|hours?|years?)"),
}

PUNCT_ONLY_PATTERN = re.compile(r"^[\s\W]+$")


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def value_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def date_from_epoch_ms(timestamp: int) -> str:
    if not timestamp:
        return "未知日期"
    try:
        return datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
    except (OSError, ValueError):
        return "未知日期"


def time_text_from_epoch_ms(timestamp: int) -> str:
    if not timestamp:
        return ""
    try:
        return datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError):
        return ""


def day_part(timestamp_iso: str) -> str:
    try:
        hour = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00")).hour
    except (ValueError, TypeError):
        return ""
    if 5 <= hour < 12:
        return "上午"
    if 12 <= hour < 18:
        return "下午"
    if 18 <= hour < 23:
        return "晚上"
    return "凌晨"


def memcell_profile_inputs(memcell: MemCell, appid: str = "locomo", agent_id: str = "") -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    messages = memcell.original_data
    for index, msg in enumerate(messages):
        speaker_id = str(msg.get("speaker_id", "")).strip()
        if not speaker_id:
            continue
        next_msg = messages[index + 1] if index + 1 < len(messages) else {}
        answer = ""
        if next_msg.get("speaker_id") != speaker_id:
            answer = str(next_msg.get("content", ""))
        by_speaker[speaker_id].append(
            {
                "time": msg.get("timestamp", ""),
                "sid": msg.get("dia_id", ""),
                "query": str(msg.get("content", "")),
                "answer": answer,
                "speaker_name": msg.get("speaker_name", speaker_id),
            }
        )

    records = []
    for uid, original_data in by_speaker.items():
        if not original_data:
            continue
        records.append(
            {
                "uid": uid,
                "user_name": original_data[0].get("speaker_name", uid),
                "memcell_id": memcell.memcell_id,
                "appid": appid,
                "agent_id": agent_id,
                "timestamp": datetime_to_epoch_ms(memcell.timestamp),
                "original_data": original_data,
            }
        )
    return records


def first_evidence(queries: list[str]) -> str:
    for query in queries:
        query = query.strip()
        if query:
            return query[:120]
    return "该切片包含用户对话。"


def classify_personality(queries: list[str]) -> dict[str, Any]:
    text = " ".join(queries).lower()
    evidence = first_evidence(queries)
    concrete = any(token in text for token in ["where", "when", "which", "how much", "how many", "specific", "tomorrow"])
    detailed = any(token in text for token in ["why", "how", "compare", "plan", "steps", "because", "detail"])
    diverse = any(token in text for token in ["other", "also", "besides", "different", "options", "recommend"])
    risk_safe = any(token in text for token in ["safe", "stable", "risk", "careful", "worried"])
    long_term = any(token in text for token in ["future", "plan", "career", "long", "next year", "years"])
    return {
        "认知分析": {
            "抽象程度": {"name": "具体" if concrete else "抽象", "value": "0.70" if concrete else "0.55", "证据说明": f"依据用户提问：{evidence}"},
            "复杂程度": {"name": "详细" if detailed else "简洁", "value": "0.65" if detailed else "0.55", "证据说明": f"依据用户提问：{evidence}"},
            "探索程度": {"name": "多样" if diverse else "收敛", "value": "0.65" if diverse else "0.55", "证据说明": f"依据用户提问：{evidence}"},
        },
        "行为分析": {
            "风险偏好": {"name": "保守" if risk_safe else "冒险", "value": "0.60" if risk_safe else "0.45", "证据说明": f"依据用户提问：{evidence}"},
            "决策周期": {"name": "长期" if long_term else "即时", "value": "0.65" if long_term else "0.55", "证据说明": f"依据用户提问：{evidence}"},
        },
    }


def extract_entities(queries: list[str]) -> list[str]:
    text = " ".join(queries)
    entities: list[str] = []
    for token in re.findall(r"[A-Z][A-Za-z0-9_+-]{2,}|[A-Za-z][A-Za-z0-9_+-]{3,}", text):
        lowered = token.lower()
        if lowered in CORE_KEYWORDS and token not in entities:
            entities.append(token)
    for keyword in sorted(CORE_KEYWORDS):
        if keyword in text.lower() and keyword not in entities:
            entities.append(keyword)
    return entities[:8]


def append_core_fact(
    facts: list[dict[str, Any]],
    *,
    entity: str,
    field: str,
    aspect: str,
    value: str,
    content: str,
    evidence_ids: list[str],
    confidence: float = 0.7,
) -> None:
    value = value.strip(" .")
    content = content.strip()
    evidence_ids = [item for item in evidence_ids if item]
    if field not in CORE_FACT_FIELDS or not entity or not value or not content or not evidence_ids:
        return
    key = (entity.lower(), field, aspect.lower(), value.lower())
    existing = {
        (
            str(item.get("entity", "")).lower(),
            str(item.get("field", "")),
            str(item.get("aspect", "")).lower(),
            str(item.get("value", "")).lower(),
        )
        for item in facts
    }
    if key in existing:
        return
    facts.append(
        {
            "entity": entity,
            "field": field,
            "aspect": aspect,
            "value": value,
            "content": content,
            "evidence_ids": evidence_ids,
            "confidence": confidence,
        }
    )


def normalize_core_fact(raw: Any, default_entity: str, default_evidence_ids: list[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    field = str(raw.get("field", "")).strip()
    if field not in CORE_FACT_FIELDS:
        return None
    entity = str(raw.get("entity") or default_entity).strip()
    value = str(raw.get("value", "")).strip(" .")
    content = str(raw.get("content") or f"{entity}: {value}").strip()
    aspect = str(raw.get("aspect") or field).strip()
    evidence_ids = raw.get("evidence_ids")
    if not isinstance(evidence_ids, list):
        evidence_ids = default_evidence_ids
    evidence_ids = [str(item) for item in evidence_ids if item]
    try:
        confidence = float(raw.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7
    if not entity or not value or not content or not evidence_ids:
        return None
    return {
        "entity": entity,
        "field": field,
        "aspect": aspect,
        "value": value,
        "content": content,
        "evidence_ids": evidence_ids,
        "confidence": round(min(max(confidence, 0.0), 1.0), 4),
    }


def extract_core_facts_heuristic(record: dict[str, Any]) -> list[dict[str, Any]]:
    entity = str(record.get("user_name") or record.get("uid") or "").strip()
    facts: list[dict[str, Any]] = []
    for item in record.get("original_data", []):
        text = str(item.get("query", "")).strip()
        if not text:
            continue
        lower = text.lower()
        evidence_ids = [str(item.get("sid", ""))]

        identity_match = re.search(r"\bI(?:'m| am)\s+(?:a|an)\s+([^.!?]+)", text, flags=re.IGNORECASE)
        if identity_match:
            value = identity_match.group(1).strip()
            if not re.match(r"(planning|going|looking|trying|thinking|considering)\b", value, flags=re.IGNORECASE):
                append_core_fact(
                    facts,
                    entity=entity,
                    field="identity",
                    aspect="self_description",
                    value=value,
                    content=f"{entity} is {value}.",
                    evidence_ids=evidence_ids,
                    confidence=0.8,
                )

        if re.search(r"\bI(?:'m| am)\s+single\b", text, flags=re.IGNORECASE):
            append_core_fact(
                facts,
                entity=entity,
                field="identity",
                aspect="relationship_status",
                value="single",
                content=f"{entity} is single.",
                evidence_ids=evidence_ids,
                confidence=0.8,
            )

        origin_match = re.search(r"\b(?:moved from|from)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)", text)
        if origin_match:
            value = origin_match.group(1)
            append_core_fact(
                facts,
                entity=entity,
                field="identity",
                aspect="origin",
                value=value,
                content=f"{entity} is from {value}.",
                evidence_ids=evidence_ids,
                confidence=0.75,
            )

        location_match = re.search(r"\b(?:live in|living in)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)", text)
        if location_match:
            value = location_match.group(1)
            append_core_fact(
                facts,
                entity=entity,
                field="identity",
                aspect="current_location",
                value=value,
                content=f"{entity} lives in {value}.",
                evidence_ids=evidence_ids,
                confidence=0.75,
            )

        relationship_terms = {
            "mother": ["my mother", "my mom"],
            "father": ["my father", "my dad"],
            "children": ["my children", "my kids"],
            "family": ["my family"],
            "friends": ["my friends"],
            "mentors": ["my mentors"],
            "partner": ["my partner", "my wife", "my husband"],
        }
        for value, patterns in relationship_terms.items():
            if any(pattern in lower for pattern in patterns):
                append_core_fact(
                    facts,
                    entity=entity,
                    field="relationship",
                    aspect=value,
                    value=value,
                    content=f"{entity} has a stable relationship reference to {value}.",
                    evidence_ids=evidence_ids,
                    confidence=0.65,
                )

        if any(token in lower for token in ["allergic", "allergy", "can't eat", "do not eat", "don't eat"]):
            append_core_fact(
                facts,
                entity=entity,
                field="allergy",
                aspect="health_or_food_restriction",
                value=text,
                content=f"{entity} has this allergy or food restriction: {text}",
                evidence_ids=evidence_ids,
                confidence=0.75,
            )
        elif any(token in lower for token in ["can't", "cannot", "doctor advised", "need to avoid", "have to avoid"]):
            append_core_fact(
                facts,
                entity=entity,
                field="constraint",
                aspect="stable_constraint",
                value=text,
                content=f"{entity} has this stable constraint: {text}",
                evidence_ids=evidence_ids,
                confidence=0.65,
            )

        if any(token in lower for token in ["i like", "i love", "i prefer", "favorite", "passionate about", "enjoy"]):
            append_core_fact(
                facts,
                entity=entity,
                field="preference",
                aspect="stated_preference",
                value=text,
                content=f"{entity} has this stated preference: {text}",
                evidence_ids=evidence_ids,
                confidence=0.7,
            )

        long_term_hint = any(token in lower for token in ["career", "education", "college", "university", "certification", "degree", "pursue", "psychology", "counseling"])
        if long_term_hint and any(token in lower for token in ["want to", "hope to", "plan to", "would like to", "pursu", "considering", "interested in"]):
            append_core_fact(
                facts,
                entity=entity,
                field="goal",
                aspect="long_term_goal",
                value=text,
                content=f"{entity} has this long-term goal or pursuit: {text}",
                evidence_ids=evidence_ids,
                confidence=0.7,
            )
    return facts


def extract_basic_profile(record: dict[str, Any], queries: list[str]) -> dict[str, Any]:
    text = " ".join(queries)
    lower = text.lower()
    basic: dict[str, str] = {"姓名": str(record.get("user_name") or record["uid"])}
    if any(token in lower for token in ["my mom", "my mother", "mother"]):
        basic["家庭关系"] = "提到母亲"
    elif any(token in lower for token in ["my dad", "my father", "father"]):
        basic["家庭关系"] = "提到父亲"
    elif any(token in lower for token in ["my kids", "my children", "kids"]):
        basic["家庭关系"] = "提到孩子"
    location_match = re.search(r"\b(?:live in|from|in)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)", text)
    if location_match:
        basic["所在区域"] = location_match.group(1)

    skills = []
    for skill in ["Python", "Java", "Excel", "coding", "writing", "teaching", "photography"]:
        if skill.lower() in lower:
            skills.append(skill)

    habits = []
    habit_patterns = [
        ("做饭", ["cook", "cooking", "meal"]),
        ("运动健身", ["gym", "run", "running", "workout"]),
        ("照顾家庭", ["kids", "children", "family"]),
        ("旅行规划", ["travel", "trip", "flight", "hotel"]),
        ("寻找餐饮", ["food", "restaurant", "eat", "dinner", "lunch"]),
    ]
    time_part = day_part(record["original_data"][0].get("time", ""))
    def has_pattern(pattern: str) -> bool:
        if pattern.isalpha() and len(pattern) <= 5:
            return re.search(rf"\b{re.escape(pattern)}\b", lower) is not None
        return pattern in lower

    for behavior, patterns in habit_patterns:
        if any(has_pattern(pattern) for pattern in patterns):
            habits.append({"行为": behavior, "时间": time_part, "时间类型": "对话时间", "证据": first_evidence(queries)})
            break

    return {
        "基础资料": basic,
        "教育背景": {},
        "专业技能": skills,
        "职业背景": {},
        "消费分析": [],
        "查询实体": extract_entities(queries),
        "生活习惯": habits,
    }


def quantitative_score(queries: list[str]) -> tuple[float, dict[str, Any]]:
    if not queries:
        return 0.0, {}
    per_query = []
    hit_counter = {name: 0 for name in QUANT_PATTERNS}
    for query in queries:
        matched = {name for name, pattern in QUANT_PATTERNS.items() if pattern.search(query)}
        richness = min(len(matched) / 3, 1.0)
        per_query.append(richness)
        for name in matched:
            hit_counter[name] += 1
    return round(sum(per_query) / len(per_query), 4), {
        "各类型命中query数": {name: count for name, count in hit_counter.items() if count > 0},
        "各query丰富度": [round(item, 2) for item in per_query],
    }


def content_richness(queries: list[str]) -> tuple[float, dict[str, Any]]:
    if not queries:
        return 0.0, {"平均长度得分": 0.0, "定量数据得分": 0.0, "长度均衡得分": 0.0, "avg_len": 0, "total_queries": 0}
    lengths = [len(query) for query in queries]
    total_len = sum(lengths)
    avg_len = total_len / len(queries)
    avg_len_score = round(min(avg_len / 200, 1.0), 4)
    quant_score, quant_detail = quantitative_score(queries)
    if total_len == 0:
        balance_score = 0.0
    else:
        variance_score = max(0.0, 1.0 - (max(lengths) - min(lengths)) / total_len)
        count_weight = min(len(queries) / 3, 1.0)
        balance_score = round(variance_score * count_weight, 4)
    richness = round((avg_len_score + quant_score + balance_score) / 3, 2)
    return richness, {
        "平均长度得分": round(avg_len_score, 2),
        "定量数据得分": round(quant_score, 2),
        "长度均衡得分": round(balance_score, 2),
        "avg_len": round(avg_len, 1),
        "quant_detail": quant_detail,
        "total_queries": len(queries),
    }


def core_relevance(queries: list[str], entities: list[Any]) -> tuple[float, dict[str, Any]]:
    entity_score = round(min(len(entities) / 3, 1.0), 2)
    combined = " ".join(queries).lower()
    keyword_hits = sum(1 for keyword in CORE_KEYWORDS if keyword.lower() in combined)
    keyword_score = round(min(keyword_hits / 5, 1.0), 2)
    relevance = round((entity_score + keyword_score) / 2, 2)
    return relevance, {"实体得分": entity_score, "关键词得分": keyword_score, "实体数量": len(entities), "关键词命中数": keyword_hits}


def dialogue_completeness(original_data: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    valid_count = sum(1 for item in original_data if str(item.get("query", "")).strip() and str(item.get("answer", "")).strip())
    count_score = round(min(valid_count / 3, 1.0), 2)
    loop_score = 1.0 if valid_count >= 2 else 0.0
    completeness = round((count_score + loop_score) / 2, 2)
    return completeness, {"条数得分": count_score, "闭环得分": loop_score, "有效对话数": valid_count}


def parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def interaction_activity(original_data: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    time_span = None
    if len(original_data) >= 2:
        first_dt = parse_time(str(original_data[0].get("time", "")))
        last_dt = parse_time(str(original_data[-1].get("time", "")))
        if first_dt and last_dt:
            time_span = abs((last_dt - first_dt).total_seconds())
    time_score = 1.0 if time_span is None or time_span <= 3600 else 0.5
    issues = []
    seen = set()
    for index, item in enumerate(original_data):
        query = str(item.get("query", "")).strip()
        if not query or len(query) <= 2:
            issues.append(f"第{index + 1}条query过短或为空")
        elif PUNCT_ONLY_PATTERN.match(query):
            issues.append(f"第{index + 1}条query为纯标点/空白")
        elif query in seen:
            issues.append(f"第{index + 1}条query重复")
        seen.add(query)
    active_score = 1.0 if not issues else 0.5
    activity = round((time_score + active_score) / 2, 2)
    return activity, {"时间得分": time_score, "主动得分": active_score, "时间跨度秒": time_span, "无效提问数": len(issues)}


def evaluate_slice(original_data: list[dict[str, Any]], profile_result: dict[str, Any]) -> dict[str, Any]:
    queries = [str(item.get("query", "")) for item in original_data]
    entities = profile_result.get("对话分析", {}).get("基础画像", {}).get("查询实体", [])
    richness, richness_detail = content_richness(queries)
    relevance, relevance_detail = core_relevance(queries, entities if isinstance(entities, list) else [])
    completeness, completeness_detail = dialogue_completeness(original_data)
    activity, activity_detail = interaction_activity(original_data)
    weight = round((richness + relevance + completeness + activity) / 4, 2)
    return {
        "切片权重": weight,
        "内容丰富度": richness,
        "核心关联度": relevance,
        "对话完整度": completeness,
        "交互活跃度": activity,
        "权重说明": (
            f"切片权重{weight}=(内容丰富度{richness}+核心关联度{relevance}+对话完整度{completeness}+交互活跃度{activity})/4；"
            f"内容丰富度=(平均长度得分{richness_detail['平均长度得分']}[均长{richness_detail['avg_len']}字]"
            f"+定量数据得分{richness_detail['定量数据得分']}[{richness_detail['total_queries']}条query均丰富度]"
            f"+长度均衡得分{richness_detail['长度均衡得分']})/3；"
            f"核心关联度=(实体得分{relevance_detail['实体得分']}[{relevance_detail['实体数量']}个实体]"
            f"+关键词得分{relevance_detail['关键词得分']}[命中{relevance_detail['关键词命中数']}个])/2；"
            f"对话完整度=(条数得分{completeness_detail['条数得分']}[{completeness_detail['有效对话数']}条有效]"
            f"+闭环得分{completeness_detail['闭环得分']})/2；"
            f"交互活跃度=(时间得分{activity_detail['时间得分']}+主动得分{activity_detail['主动得分']})/2"
        ),
    }


def heuristic_core_profile(record: dict[str, Any]) -> dict[str, Any]:
    queries = [str(item.get("query", "")) for item in record.get("original_data", []) if item.get("query")]
    result = {
        "uid": record["uid"],
        "memcell_id": record["memcell_id"],
        "核心事实": extract_core_facts_heuristic(record),
        "对话分析": {
            "人格分析": classify_personality(queries),
            "基础画像": extract_basic_profile(record, queries),
        },
    }
    result["切片定量评估"] = evaluate_slice(record.get("original_data", []), result)
    result.update(
        {
            "appid": record.get("appid", "locomo"),
            "agent_id": record.get("agent_id", ""),
            "timestamp": record.get("timestamp", 0),
            "user_name": record.get("user_name", record["uid"]),
        }
    )
    return result


def llm_core_profile(record: dict[str, Any], client: OpenAIProtocolClient) -> dict[str, Any]:
    payload = {
        "uid": record["uid"],
        "user_name": record.get("user_name", record["uid"]),
        "memcell_id": record["memcell_id"],
        "original_data": record.get("original_data", []),
    }
    prompt = CORE_PROFILE_PROMPT.replace("{chat_list}", json.dumps(payload, ensure_ascii=False))
    raw_response = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    try:
        parsed = parse_json_object(raw_response)
    except json.JSONDecodeError:
        repair_prompt = CORE_PROFILE_REPAIR_PROMPT.replace("{invalid_json}", raw_response)
        try:
            parsed = parse_json_object(client.chat([{"role": "user", "content": repair_prompt}], temperature=0.0))
        except json.JSONDecodeError:
            retry_prompt = prompt + "\n\n上一次输出不是合法 JSON。请重新抽取，并严格只返回合法 JSON object。"
            parsed = parse_json_object(client.chat([{"role": "user", "content": retry_prompt}], temperature=0.0))
    default_evidence_ids = [str(item.get("sid", "")) for item in record.get("original_data", []) if item.get("sid")]
    parsed["核心事实"] = [
        fact
        for fact in (
            normalize_core_fact(item, str(record.get("user_name") or record["uid"]), default_evidence_ids)
            for item in parsed.get("核心事实", [])
        )
        if fact
    ]
    parsed.setdefault("对话分析", {"人格分析": {"认知分析": {}, "行为分析": {}}, "基础画像": {}})
    parsed["切片定量评估"] = evaluate_slice(record.get("original_data", []), parsed)
    parsed.update(
        {
            "appid": record.get("appid", "locomo"),
            "agent_id": record.get("agent_id", ""),
            "timestamp": record.get("timestamp", 0),
            "user_name": record.get("user_name", parsed.get("uid", "")),
        }
    )
    return parsed


def extract_core_profile_slices(
    memcells: list[MemCell],
    mode: str = "llm",
    appid: str = "locomo",
    agent_id: str = "",
    client: OpenAIProtocolClient | None = None,
    max_slices: int | None = None,
) -> list[dict[str, Any]]:
    if mode == "llm":
        client = client or OpenAIProtocolClient.from_env()
        if client.settings.offline:
            raise RuntimeError(
                "core memory LLM extraction requires MEMORY_TOOL_PLANNER_OFFLINE=0 "
                "and OpenAI-compatible chat settings; pass mode='heuristic' only for offline smoke tests."
            )
    slices: list[dict[str, Any]] = []
    for memcell in memcells:
        for record in memcell_profile_inputs(memcell, appid=appid, agent_id=agent_id):
            if max_slices is not None and len(slices) >= max_slices:
                return slices
            profile = llm_core_profile(record, client) if mode == "llm" else heuristic_core_profile(record)  # type: ignore[arg-type]
            profile["_source_memcell_ids"] = [memcell.memcell_id]
            profile["_source_evidence_ids"] = [item.get("sid", "") for item in record.get("original_data", []) if item.get("sid")]
            slices.append(profile)
    return slices


def empty_merged_user() -> dict[str, Any]:
    return {
        "uid": "",
        "user_name": "",
        "appid": "",
        "agent_id": "",
        "date": "",
        "认知分析": defaultdict(lambda: defaultdict(float)),
        "行为分析": defaultdict(lambda: defaultdict(float)),
        "基础资料": {},
        "教育背景": {},
        "职业背景": {},
        "消费分析": [],
        "生活习惯": [],
        "查询实体": defaultdict(float),
        "专业技能": defaultdict(float),
        "核心事实": [],
        "总切片数": 0,
        "切片权重列表": [],
        "累计切片权重": 0.0,
        "timestamp_list": [],
        "source_memcell_ids": set(),
        "source_evidence_ids": set(),
        "_field_timestamps": {},
        "_list_timestamps": {},
        "_fact_timestamps": {},
    }


def process_analysis(analysis: dict[str, Any], target: dict[str, Any], slice_weight: float, analysis_type: str) -> None:
    keep = KEEP_DIMENSIONS[analysis_type]
    for dimension, payload in analysis.items():
        if dimension not in keep or not isinstance(payload, dict):
            continue
        label = payload.get("name")
        if label not in keep[dimension]:
            continue
        target[dimension][label] += value_float(payload.get("value")) * slice_weight


def latest_update(target: dict[str, Any], timestamps: dict[str, int], prefix: str, values: dict[str, Any], timestamp: int) -> None:
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        if value in ("", None, [], {}):
            continue
        path = f"{prefix}.{key}"
        if timestamp >= timestamps.get(path, -1):
            target[key] = value
            timestamps[path] = timestamp


def merge_list_items(target: list[Any], timestamps: dict[str, int], prefix: str, values: list[Any], timestamp: int) -> None:
    if not isinstance(values, list):
        return
    existing_keys = {json.dumps(item, ensure_ascii=False, sort_keys=True): i for i, item in enumerate(target)}
    for item in values:
        if item in ("", None, [], {}):
            continue
        item_key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        time_key = f"{prefix}.{item_key}"
        if item_key in existing_keys:
            if timestamp >= timestamps.get(time_key, -1):
                target[existing_keys[item_key]] = item
                timestamps[time_key] = timestamp
        else:
            target.append(item)
            timestamps[time_key] = timestamp
            existing_keys[item_key] = len(target) - 1


def merge_core_facts(target: list[dict[str, Any]], timestamps: dict[str, int], values: list[Any], timestamp: int, default_entity: str) -> None:
    if not isinstance(values, list):
        return
    existing_keys = {
        "|".join(
            [
                str(item.get("entity", "")).lower(),
                str(item.get("field", "")),
                str(item.get("aspect", "")).lower(),
                str(item.get("value", "")).lower(),
            ]
        ): i
        for i, item in enumerate(target)
    }
    for raw in values:
        fact = normalize_core_fact(raw, default_entity, [])
        if not fact:
            continue
        fact_key = "|".join(
            [
                str(fact.get("entity", "")).lower(),
                str(fact.get("field", "")),
                str(fact.get("aspect", "")).lower(),
                str(fact.get("value", "")).lower(),
            ]
        )
        if fact_key in existing_keys:
            if timestamp >= timestamps.get(fact_key, -1):
                target[existing_keys[fact_key]] = fact
                timestamps[fact_key] = timestamp
        else:
            target.append(fact)
            timestamps[fact_key] = timestamp
            existing_keys[fact_key] = len(target) - 1


def update_base_profile(base_profile: dict[str, Any], merged_user: dict[str, Any], timestamp: int, slice_weight: float) -> None:
    field_timestamps = merged_user["_field_timestamps"]
    latest_update(merged_user["基础资料"], field_timestamps, "基础资料", base_profile.get("基础资料", {}), timestamp)
    latest_update(merged_user["教育背景"], field_timestamps, "教育背景", base_profile.get("教育背景", {}), timestamp)
    latest_update(merged_user["职业背景"], field_timestamps, "职业背景", base_profile.get("职业背景", {}), timestamp)
    merge_list_items(merged_user["消费分析"], merged_user["_list_timestamps"], "消费分析", base_profile.get("消费分析", []), timestamp)
    merge_list_items(merged_user["生活习惯"], merged_user["_list_timestamps"], "生活习惯", base_profile.get("生活习惯", []), timestamp)

    for skill in base_profile.get("专业技能", []) if isinstance(base_profile.get("专业技能", []), list) else []:
        if skill:
            merged_user["专业技能"][str(skill)] += slice_weight
    for entity in base_profile.get("查询实体", []) if isinstance(base_profile.get("查询实体", []), list) else []:
        if entity:
            merged_user["查询实体"][str(entity)] += slice_weight


def update_merged_user(user: dict[str, Any], slice_item: dict[str, Any]) -> None:
    uid = str(slice_item.get("uid", ""))
    user_name = str(slice_item.get("user_name") or uid)
    appid = str(slice_item.get("appid", ""))
    agent_id = str(slice_item.get("agent_id", "")).strip()
    timestamp = int(slice_item.get("timestamp") or 0)
    slice_weight = value_float(slice_item.get("切片定量评估", {}).get("切片权重", 0.0))

    if not user["uid"]:
        user["uid"] = uid
        user["user_name"] = user_name
        user["appid"] = appid
        user["agent_id"] = agent_id
        user["date"] = date_from_epoch_ms(timestamp)

    user["timestamp_list"].append(timestamp)
    user["总切片数"] += 1
    user["切片权重列表"].append(slice_weight)
    user["累计切片权重"] += slice_weight
    user["source_memcell_ids"].update(slice_item.get("_source_memcell_ids", [slice_item.get("memcell_id", "")]))
    user["source_evidence_ids"].update(slice_item.get("_source_evidence_ids", []))

    dialog_analysis = slice_item.get("对话分析", {})
    personality = dialog_analysis.get("人格分析", {})
    if "认知分析" in personality:
        process_analysis(personality["认知分析"], user["认知分析"], slice_weight, "认知分析")
    if "行为分析" in personality:
        process_analysis(personality["行为分析"], user["行为分析"], slice_weight, "行为分析")
    update_base_profile(dialog_analysis.get("基础画像", {}), user, timestamp, slice_weight)
    merge_core_facts(user["核心事实"], user["_fact_timestamps"], slice_item.get("核心事实", []), timestamp, user_name)


def format_merged_user(user: dict[str, Any]) -> dict[str, Any]:
    timestamps = [ts for ts in user["timestamp_list"] if ts]
    earliest = min(timestamps) if timestamps else 0
    latest = max(timestamps) if timestamps else 0
    return {
        "uid": user["uid"],
        "user_name": user["user_name"],
        "appid": user["appid"],
        "agent_id": user["agent_id"],
        "date": user["date"],
        "时间范围": {
            "最早时间戳": earliest,
            "最晚时间戳": latest,
            "最早时间": time_text_from_epoch_ms(earliest),
            "最晚时间": time_text_from_epoch_ms(latest),
        },
        "合并后统计信息": {
            "总切片数量": user["总切片数"],
            "各切片权重": [round(weight, 4) for weight in user["切片权重列表"]],
            "切片权重累计值": round(user["累计切片权重"], 4),
        },
        "对话分析_合并后": {
            "人格分析": {
                "认知分析": {
                    dimension: {label: round(score, 4) for label, score in labels.items() if score > 0}
                    for dimension, labels in user["认知分析"].items()
                },
                "行为分析": {
                    dimension: {label: round(score, 4) for label, score in labels.items() if score > 0}
                    for dimension, labels in user["行为分析"].items()
                },
            },
            "基础画像": {
                "基础资料": user["基础资料"],
                "教育背景": user["教育背景"],
                "职业背景": user["职业背景"],
                "消费分析": user["消费分析"],
                "专业技能": {skill: round(score, 4) for skill, score in user["专业技能"].items() if score > 0},
                "查询实体": {entity: round(score, 4) for entity, score in user["查询实体"].items() if score > 0},
                "生活习惯": user["生活习惯"],
            },
        },
        "核心事实": user["核心事实"],
        "_source_memcell_ids": sorted(item for item in user["source_memcell_ids"] if item),
        "_source_evidence_ids": sorted(item for item in user["source_evidence_ids"] if item),
    }


def merge_core_profile_slices(profile_slices: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    not_agent: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(empty_merged_user)
    is_agent: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(empty_merged_user)
    for slice_item in profile_slices:
        uid = str(slice_item.get("uid", ""))
        appid = str(slice_item.get("appid", ""))
        agent_id = str(slice_item.get("agent_id", "")).strip()
        date = date_from_epoch_ms(int(slice_item.get("timestamp") or 0))
        if not uid or not appid:
            continue
        update_merged_user(not_agent[(uid, appid, date)], slice_item)
        if agent_id:
            update_merged_user(is_agent[(uid, appid, agent_id, date)], slice_item)
    return {
        "not_agent": [format_merged_user(user) for user in not_agent.values() if user["总切片数"] > 0],
        "is_agent": [format_merged_user(user) for user in is_agent.values() if user["总切片数"] > 0],
    }


def winner_label(scores: dict[str, float], positive_prefix: str) -> str:
    if not scores:
        return ""
    return f"{positive_prefix}{max(scores.items(), key=lambda item: item[1])[0]}"


def convert_to_core_memory(merged_user: dict[str, Any]) -> dict[str, Any]:
    dialog = merged_user.get("对话分析_合并后", {})
    personality = dialog.get("人格分析", {})
    base = dialog.get("基础画像", {})
    cognitive = personality.get("认知分析", {})
    behavior = personality.get("行为分析", {})

    personality_pref = {"认知维度": {}, "行为分析": {}}
    mappings = [
        ("抽象程度", cognitive.get("抽象程度", {}), "认知维度"),
        ("复杂程度", cognitive.get("复杂程度", {}), "认知维度"),
        ("探索程度", cognitive.get("探索程度", {}), "认知维度"),
        ("决策周期", behavior.get("决策周期", {}), "行为分析"),
        ("风险偏好", behavior.get("风险偏好", {}), "行为分析"),
    ]
    for key, scores, target in mappings:
        label = winner_label(scores, "偏")
        if label:
            personality_pref[target][key] = label

    query_entities = base.get("查询实体", {})
    sorted_entities = sorted(query_entities.items(), key=lambda item: item[1], reverse=True)[:5] if isinstance(query_entities, dict) else []
    habits = []
    for habit in base.get("生活习惯", [])[:3] if isinstance(base.get("生活习惯", []), list) else []:
        behavior_text = habit.get("行为", "") if isinstance(habit, dict) else str(habit)
        time_text = habit.get("时间", "") if isinstance(habit, dict) else ""
        if behavior_text:
            habits.append(f"{time_text}{behavior_text}" if time_text else behavior_text)

    base_portrait = {
        "基础资料": base.get("基础资料", {}),
        "教育背景": {k: v for k, v in base.get("教育背景", {}).items() if k in {"最高学历", "学科背景", "教育经历"}},
        "职业背景": {k: v for k, v in base.get("职业背景", {}).items() if k in {"职业身份", "所属行业"}},
        "消费偏好": base.get("消费分析", []),
        "专业技能": list(base.get("专业技能", {}).keys()) if isinstance(base.get("专业技能", {}), dict) else base.get("专业技能", []),
        "查询偏好": dict(sorted_entities),
        "生活习惯": habits,
    }
    return {
        "uid": merged_user.get("uid", ""),
        "user_name": merged_user.get("user_name", merged_user.get("uid", "")),
        "appid": merged_user.get("appid", ""),
        "agent_id": merged_user.get("agent_id", ""),
        "核心事实": merged_user.get("核心事实", []),
        "核心记忆": {
            "人格偏好": personality_pref,
            "基础画像": base_portrait,
            "兴趣偏好": {"学习兴趣主题": {}},
        },
        "_source_memcell_ids": merged_user.get("_source_memcell_ids", []),
        "_source_evidence_ids": merged_user.get("_source_evidence_ids", []),
        "updated_at": merged_user.get("时间范围", {}).get("最晚时间", ""),
    }


def core_memory_rows(core_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in core_profiles:
        uid = profile.get("uid", "")
        default_entity = str(profile.get("user_name") or uid)
        evidence_ids = profile.get("_source_evidence_ids", [])
        memcell_ids = profile.get("_source_memcell_ids", [])
        conversation_index = infer_locomo_conversation_index(uid) or infer_locomo_conversation_index(memcell_ids)
        updated_at = profile.get("updated_at", "")

        def add_fact(fact: dict[str, Any]) -> None:
            field = str(fact.get("field", ""))
            if field not in CORE_FACT_FIELDS:
                return
            entity = str(fact.get("entity") or default_entity)
            aspect = str(fact.get("aspect") or field)
            value = fact.get("value", "")
            content = str(fact.get("content") or "").strip()
            if value in ("", None, [], {}):
                return
            text_value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            if not content:
                content = f"{entity} 的{field}.{aspect}：{text_value}"
            fact_evidence_ids = fact.get("evidence_ids") if isinstance(fact.get("evidence_ids"), list) else evidence_ids
            rows.append(
                {
                    "store": "core",
                    "memory_id": f"core_{uid}_{len(rows) + 1:06d}",
                    "memcell_ids": memcell_ids,
                    "conversation_id": f"locomo_{conversation_index}" if conversation_index is not None else "",
                    "conversation_index": conversation_index,
                    "content": content,
                    "entity": entity,
                    "field": field,
                    "core_path": f"{field}.{aspect}",
                    "aspect": aspect,
                    "value": value,
                    "confidence": fact.get("confidence", 0.7),
                    "updated_at": updated_at,
                    "evidence_ids": scoped_locomo_evidence_ids(fact_evidence_ids, conversation_index),
                }
            )

        for raw_fact in profile.get("核心事实", []):
            fact = normalize_core_fact(raw_fact, default_entity, evidence_ids)
            if fact:
                add_fact(fact)
    return rows


def build_core_memory_from_memcells(
    memcells: list[MemCell],
    mode: str = "llm",
    appid: str = "locomo",
    agent_id: str = "",
    client: OpenAIProtocolClient | None = None,
    max_slices: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    profile_slices = extract_core_profile_slices(memcells, mode=mode, appid=appid, agent_id=agent_id, client=client, max_slices=max_slices)
    merged = merge_core_profile_slices(profile_slices)
    core_profiles = [convert_to_core_memory(item) for item in merged["not_agent"]]
    return core_memory_rows(core_profiles), profile_slices, merged
