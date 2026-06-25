from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memory_tool_planner.baselines import naive_rag_plan, oracle_planner_plan, run_planner_baselines
from memory_tool_planner.builder import build_locomo_memcell, build_sample_store, read_jsonl
from memory_tool_planner.context import build_memory_context
from memory_tool_planner.core_memory import llm_core_profile
from memory_tool_planner.embeddings import EmbeddingProvider
from memory_tool_planner.evermemos_import import build_evermemos_core_memory, import_evermemos_locomo
from memory_tool_planner.index import index_store
from memory_tool_planner.answer_eval import run_answer_eval
from memory_tool_planner.locomo_planner_data import prepare_planner_data
from memory_tool_planner.planner import validate_planner_output
from memory_tool_planner.retrieval import MemoryToolExecutor
from memory_tool_planner.reward import run_reward_runner
from memory_tool_planner.reward import answer_reward
from memory_tool_planner.reward import append_reward_trace
from memory_tool_planner.reward import estimate_tokens
from memory_tool_planner.reward import estimate_tokens_rough
from memory_tool_planner.reward import reward_breakdown
from memory_tool_planner.teacher_sft import generate_teacher_sft_data
from memory_tool_planner.teacher_sft import score_teacher_candidate
from memory_tool_planner.trace_analysis import assert_summary_matches_trace
from memory_tool_planner.trace_analysis import normalized_duplicate_query_count
from memory_tool_planner.trace_analysis import planner_metric_summary
from memory_tool_planner.trace_analysis import run_trace_analysis
from memory_tool_planner.training_data import export_rl_data, export_sft_data


def load_build_hard_rows():
    try:
        from scripts.build_hard_rl_data import build_hard_rows
    except ModuleNotFoundError as exc:
        if exc.name == "pyarrow":
            raise unittest.SkipTest("pyarrow is required for hard RL data mining tests") from exc
        raise
    return build_hard_rows


class MemoryToolPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "1"

    def valid_plan(self) -> dict:
        return {
            "tool_calls": [
                {
                    "name": "search_core_memory",
                    "arguments": {
                        "queries": ["妈妈 饮食偏好 忌口 过敏", "用户 饮食偏好 忌口 过敏"],
                        "fields": ["preference", "allergy", "constraint"],
                    },
                }
            ]
        }

    def test_validates_plan(self) -> None:
        result = validate_planner_output(self.valid_plan())
        self.assertTrue(result.valid, result.errors)

    def test_validates_episodic_plan_with_queries_only(self) -> None:
        plan = {
            "tool_calls": [
                {
                    "name": "search_episodic_memory",
                    "arguments": {
                        "queries": ["明天 和妈妈 去北京 行程 住宿 活动区域"],
                    },
                }
            ]
        }
        result = validate_planner_output(plan)
        self.assertTrue(result.valid, result.errors)
        args = result.parsed["tool_calls"][0]["arguments"]
        self.assertEqual(args, {"queries": ["明天 和妈妈 去北京 行程 住宿 活动区域"], "top_k": 5})

    def test_accepts_model_visible_plan_and_assigns_runtime_fields(self) -> None:
        plan = self.valid_plan()
        result = validate_planner_output(plan)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.parsed["tool_calls"][0]["call_id"], "call_1")
        self.assertEqual(result.parsed["tool_calls"][0]["arguments"]["top_k"], 5)
        self.assertNotIn("purpose", result.parsed["tool_calls"][0])

    def test_ignores_model_supplied_top_k(self) -> None:
        plan = self.valid_plan()
        plan["tool_calls"][0]["arguments"]["top_k"] = 20
        result = validate_planner_output(plan)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.parsed["tool_calls"][0]["arguments"]["top_k"], 5)

    def test_executor_accepts_model_visible_plan(self) -> None:
        plan = self.valid_plan()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = tmp_path / "stores.jsonl"
            indexed = tmp_path / "stores.embedded.jsonl"
            build_sample_store(store)
            index_store(store, indexed, EmbeddingProvider.from_env())
            executor = MemoryToolExecutor.from_jsonl(indexed)
            results = executor.execute_plan(plan)
            self.assertEqual(results["tool_results"][0]["status"], "ok")
            self.assertEqual(results["tool_results"][0]["call_id"], "call_1")

    def test_rejects_duplicate_query_same_tool(self) -> None:
        plan = self.valid_plan()
        second = json.loads(json.dumps(plan["tool_calls"][0], ensure_ascii=False))
        plan["tool_calls"].append(second)
        result = validate_planner_output(plan)
        self.assertFalse(result.valid)
        self.assertTrue(any(error["code"] == "DUPLICATE_CALL" for error in result.errors))

    def test_index_retrieve_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = tmp_path / "stores.jsonl"
            indexed = tmp_path / "stores.embedded.jsonl"
            build_sample_store(store)
            index_store(store, indexed, EmbeddingProvider.from_env())
            executor = MemoryToolExecutor.from_jsonl(indexed)
            results = executor.execute_plan(self.valid_plan())
            self.assertEqual(results["tool_results"][0]["status"], "ok")
            memories = results["tool_results"][0]["result"]["memories"]
            self.assertGreaterEqual(len(memories), 1)
            context = build_memory_context(results)
            self.assertIn("memory_context", context)
            self.assertIn("evidence=", context["memory_context"])

    def test_zero_token_budget_keeps_all_memories(self) -> None:
        results = {
            "tool_results": [
                {
                    "name": "search_episodic_memory",
                    "status": "ok",
                    "result": {
                        "memories": [
                            {"memory_id": "m1", "event": "one two three", "evidence_ids": ["e1"]},
                            {"memory_id": "m2", "event": "four five six", "evidence_ids": ["e2"]},
                        ]
                    },
                }
            ]
        }
        limited = build_memory_context(results, token_budget=1)
        unlimited = build_memory_context(results, token_budget=0)
        self.assertEqual(len(limited["memories"]), 0)
        self.assertEqual(len(unlimited["memories"]), 2)

    def test_estimate_tokens_uses_qwen_tokenizer_when_available(self) -> None:
        class FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                self.add_special_tokens = add_special_tokens
                return [1, 2, 3]

        with patch("memory_tool_planner.reward._load_qwen3_tokenizer", return_value=FakeTokenizer()):
            self.assertEqual(estimate_tokens("one two three four five"), 3)

    def test_estimate_tokens_falls_back_to_rough_count_without_tokenizer(self) -> None:
        text = "one two three four five"
        with patch("memory_tool_planner.reward._load_qwen3_tokenizer", return_value=None):
            self.assertEqual(estimate_tokens(text), estimate_tokens_rough(text))

    def test_append_reward_trace_can_enforce_unique_question_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            first = {"question_id": "q1", "reward": {"total": 0.25}}
            duplicate = {"question_id": "q1", "reward": {"total": 1.0}}
            second = {"question_id": "q2", "reward": {"total": 0.5}}

            self.assertTrue(append_reward_trace(first, path, unique_key="question_id"))
            self.assertFalse(append_reward_trace(duplicate, path, unique_key="question_id"))
            self.assertTrue(append_reward_trace(second, path, unique_key="question_id"))

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["question_id"] for row in rows], ["q1", "q2"])
            self.assertEqual(rows[0]["reward"]["total"], 0.25)

    def test_teacher_candidate_scoring_scopes_to_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "stores.jsonl"
            rows = [
                {
                    "store": "episodic",
                    "memory_id": "locomo_0_m1",
                    "event": "Caroline visited the greenhouse.",
                    "evidence_ids": ["D1:1"],
                    "embedding": [],
                },
                {
                    "store": "episodic",
                    "memory_id": "locomo_1_m1",
                    "event": "Caroline visited the greenhouse.",
                    "evidence_ids": ["D1:1"],
                    "embedding": [],
                },
            ]
            store.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            executor = MemoryToolExecutor.from_jsonl(store)
            row = {
                "question": "Where did Caroline visit?",
                "evidence": ["locomo_1:D1:1"],
                "conversation_index": 1,
            }
            raw_output = json.dumps({"tool_calls": [{"name": "search_episodic_memory", "arguments": {"queries": ["Caroline greenhouse"]}}]})

            candidate = score_teacher_candidate(row, raw_output, executor)

            self.assertEqual(candidate["score"], 1.0)
            memories = candidate["tool_results"]["tool_results"][0]["result"]["memories"]
            self.assertEqual([memory["evidence_ids"] for memory in memories], [["locomo_1:D1:1"]])

    def test_planner_metric_summary_can_verify_trace_consistency(self) -> None:
        rows = [
            {
                "question_id": "q1",
                "planner_valid": True,
                "planner_output": {"tool_calls": []},
                "reward": {"total": 1.0, "evidence": 1.0},
                "gold_evidence_ids": ["e1"],
                "retrieved_evidence_ids": ["e1"],
            }
        ]
        summary = planner_metric_summary(rows)
        assert_summary_matches_trace(summary, rows, label="unit")
        bad_summary = dict(summary)
        bad_summary["evidence_mean"] = 0.0
        with self.assertRaises(ValueError):
            assert_summary_matches_trace(bad_summary, rows, label="unit")

    def test_locomo_memcell_build_reuses_memcell_for_core_and_episodic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            locomo = tmp_path / "locomo.json"
            store = tmp_path / "stores.jsonl"
            memcells = tmp_path / "memcells.jsonl"
            locomo.write_text(
                json.dumps(
                    [
                        {
                            "conversation": {
                                "session_1_date_time": "1:56 pm on 8 May, 2023",
                                "session_1": [
                                    {"speaker": "Caroline", "dia_id": "D1:1", "text": "I am planning a trip with my mother tomorrow."},
                                    {"speaker": "Melanie", "dia_id": "D1:2", "text": "That sounds nice. Where are you going?"},
                                    {"speaker": "Caroline", "dia_id": "D1:3", "text": "We will go to Beijing and need food recommendations."},
                                    {"speaker": "Melanie", "dia_id": "D1:4", "text": "You should look for comfortable restaurants."},
                                ],
                            }
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            stats = build_locomo_memcell(locomo, store, memcells_output=memcells, core_mode="heuristic")
            rows = read_jsonl(store)
            episodic = [row for row in rows if row["store"] == "episodic"]
            core = [row for row in rows if row["store"] == "core"]
            self.assertGreaterEqual(stats["memcells"], 1)
            self.assertGreaterEqual(len(episodic), 1)
            self.assertGreaterEqual(len(core), 1)
            self.assertTrue(any(row["field"] == "relationship" for row in core))
            self.assertTrue({row["store"] for row in rows}.issubset({"core", "episodic"}))
            episodic_memcell_ids = {row["memcell_id"] for row in episodic}
            self.assertTrue(set(core[0]["memcell_ids"]).issubset(episodic_memcell_ids))

    def test_import_evermemos_locomo_generates_memcells_and_episodic_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            locomo = tmp_path / "locomo.json"
            memcells_dir = tmp_path / "memcells"
            memcells_dir.mkdir()
            memcells_output = tmp_path / "evermemos_memcells.jsonl"
            episodic_output = tmp_path / "evermemos_episodic.jsonl"
            locomo.write_text(
                json.dumps(
                    [
                        {
                            "conversation": {
                                "session_1_date_time": "1:56 pm on 8 May, 2023",
                                "session_1": [
                                    {"speaker": "Caroline", "dia_id": "D1:1", "text": "I went to a support group yesterday."},
                                    {"speaker": "Melanie", "dia_id": "D1:2", "text": "How did it feel?"},
                                ],
                            }
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (memcells_dir / "memcell_list_conv_0.json").write_text(
                json.dumps(
                    [
                        {
                            "event_id": "evt1",
                            "user_id_list": ["caroline_locomo_0", "melanie_locomo_0"],
                            "original_data": [
                                {
                                    "speaker_id": "caroline_locomo_0",
                                    "user_name": "Caroline",
                                    "speaker_name": "Caroline",
                                    "content": "I went to a support group yesterday.",
                                    "timestamp": "2023-05-08T13:56:00+00:00",
                                },
                                {
                                    "speaker_id": "melanie_locomo_0",
                                    "user_name": "Melanie",
                                    "speaker_name": "Melanie",
                                    "content": "How did it feel?",
                                    "timestamp": "2023-05-08T13:56:30+00:00",
                                },
                            ],
                            "timestamp": "2023-05-08T13:56:30+00:00",
                            "summary": "Caroline attended a support group.",
                            "participants": ["caroline_locomo_0", "melanie_locomo_0"],
                            "type": "Conversation",
                            "subject": "Caroline's support group",
                            "episode": "Caroline attended a support group on May 7, 2023.",
                            "event_log": {
                                "atomic_fact": ["Caroline went to a support group."],
                                "fact_embeddings": [[0.1, 0.2]],
                            },
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = import_evermemos_locomo(
                memcells_dir,
                locomo,
                memcells_output=memcells_output,
                episodic_output=episodic_output,
            )
            memcells = read_jsonl(memcells_output)
            episodic = read_jsonl(episodic_output)
            self.assertEqual(summary["memcells"], 1)
            self.assertEqual(summary["messages_missing_dia_id"], 0)
            self.assertEqual(memcells[0]["evidence_ids"], ["locomo_0:D1:1", "locomo_0:D1:2"])
            self.assertNotIn("fact_embeddings", memcells[0]["event_log"])
            self.assertEqual(episodic[0]["store"], "episodic")
            self.assertEqual(episodic[0]["evidence_ids"], ["locomo_0:D1:1", "locomo_0:D1:2"])
            self.assertEqual(episodic[0]["conversation_index"], 0)
            self.assertIn("Caroline went to a support group.", episodic[0]["search_text"])

    def test_build_evermemos_core_memory_combines_core_and_episodic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            memcells = tmp_path / "memcells.jsonl"
            episodic = tmp_path / "episodic.jsonl"
            core = tmp_path / "core.jsonl"
            combined = tmp_path / "combined.jsonl"
            memcells.write_text(
                json.dumps(
                    {
                        "memcell_id": "evt1",
                        "event_id": "evt1",
                        "user_id_list": ["caroline_locomo_0"],
                        "participants": ["caroline_locomo_0"],
                        "timestamp": "2023-05-08T13:56:30Z",
                        "summary": "Caroline loves painting.",
                        "subject": "Painting",
                        "episode": "Caroline said she loves painting.",
                        "type": "Conversation",
                        "original_data": [
                            {
                                "speaker_id": "caroline_locomo_0",
                                "speaker_name": "Caroline",
                                "content": "I love painting and photography.",
                                "timestamp": "2023-05-08T13:56:00Z",
                                "dia_id": "D1:1",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            episodic.write_text(json.dumps({"store": "episodic", "memory_id": "epi1"}, ensure_ascii=False) + "\n", encoding="utf-8")
            summary = build_evermemos_core_memory(
                memcells,
                episodic,
                core_output=core,
                combined_output=combined,
                mode="heuristic",
            )
            core_rows = read_jsonl(core)
            combined_rows = read_jsonl(combined)
            self.assertGreaterEqual(summary["profile_slices"], 1)
            self.assertGreaterEqual(summary["core_memories"], 1)
            self.assertEqual(combined_rows[0]["store"], "episodic")
            self.assertTrue(any(row["store"] == "core" for row in combined_rows))
            self.assertEqual(core_rows[0]["evidence_ids"], ["D1:1"])

    def test_build_evermemos_core_memory_parallel_llm_checkpoint(self) -> None:
        def fake_extract(memcell, max_slices=None):  # type: ignore[no-untyped-def]
            uid = memcell.user_id_list[0]
            return (
                memcell.memcell_id,
                [
                    {
                        "uid": uid,
                        "user_name": uid.split("_")[0].title(),
                        "memcell_id": memcell.memcell_id,
                        "appid": "locomo",
                        "agent_id": "",
                        "timestamp": 1683554190000,
                        "核心事实": [
                            {
                                "entity": uid.split("_")[0].title(),
                                "field": "preference",
                                "aspect": "stated_preference",
                                "value": "painting",
                                "content": f"{uid} likes painting.",
                                "evidence_ids": ["D1:1"],
                                "confidence": 0.8,
                            }
                        ],
                        "对话分析": {"人格分析": {"认知分析": {}, "行为分析": {}}, "基础画像": {}},
                        "切片定量评估": {"切片权重": 1.0},
                        "_source_memcell_ids": [memcell.memcell_id],
                        "_source_evidence_ids": ["D1:1"],
                        "_core_mode": "llm",
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            memcells = tmp_path / "memcells.jsonl"
            episodic = tmp_path / "episodic.jsonl"
            core = tmp_path / "core.jsonl"
            slices = tmp_path / "slices.json"
            combined = tmp_path / "combined.jsonl"
            rows = [
                {
                    "memcell_id": "evt1",
                    "event_id": "evt1",
                    "user_id_list": ["caroline_locomo_0"],
                    "timestamp": "2023-05-08T13:56:30Z",
                    "summary": "Caroline loves painting.",
                    "original_data": [],
                },
                {
                    "memcell_id": "evt2",
                    "event_id": "evt2",
                    "user_id_list": ["melanie_locomo_0"],
                    "timestamp": "2023-05-08T13:57:30Z",
                    "summary": "Melanie loves painting.",
                    "original_data": [],
                },
            ]
            memcells.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            episodic.write_text(json.dumps({"store": "episodic", "memory_id": "epi1"}, ensure_ascii=False) + "\n", encoding="utf-8")
            with patch("memory_tool_planner.evermemos_import._extract_llm_profile_slices_for_memcell", side_effect=fake_extract):
                summary = build_evermemos_core_memory(
                    memcells,
                    episodic,
                    core_output=core,
                    core_slices_output=slices,
                    combined_output=combined,
                    mode="llm",
                    parallel_workers=2,
                )
            self.assertEqual(summary["parallel_workers"], 2)
            self.assertEqual(summary["profile_slices"], 2)
            self.assertEqual(len(json.loads(slices.read_text(encoding="utf-8"))), 2)
            self.assertEqual(len(read_jsonl(core)), 2)
            self.assertEqual(len(read_jsonl(combined)), 3)

    def test_llm_core_profile_repairs_invalid_json(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, temperature=0.0):  # type: ignore[no-untyped-def]
                self.calls += 1
                if self.calls == 1:
                    return '{ "uid": "u1", "memcell_id": "m1", "核心事实": [ { "entity": "u1", "field": "identity", "aspect": "status", "value": "bad " quote", "content": "bad", "evidence_ids": ["s1"], "confidence": 0.8 } ], "对话分析": {"人格分析": {"认知分析": {}, "行为分析": {}}, "基础画像": {}} }'
                return json.dumps(
                    {
                        "uid": "u1",
                        "memcell_id": "m1",
                        "核心事实": [
                            {
                                "entity": "u1",
                                "field": "identity",
                                "aspect": "status",
                                "value": "student",
                                "content": "u1 is a student.",
                                "evidence_ids": ["s1"],
                                "confidence": 0.8,
                            }
                        ],
                        "对话分析": {"人格分析": {"认知分析": {}, "行为分析": {}}, "基础画像": {}},
                    },
                    ensure_ascii=False,
                )

        record = {
            "uid": "u1",
            "user_name": "u1",
            "memcell_id": "m1",
            "original_data": [{"sid": "s1", "query": "I am a student.", "answer": ""}],
        }
        profile = llm_core_profile(record, FakeClient())  # type: ignore[arg-type]
        self.assertEqual(profile["核心事实"][0]["value"], "student")
        self.assertIn("切片定量评估", profile)

    def test_prepares_unlabeled_planner_data_and_teacher_labels_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            locomo = tmp_path / "locomo.json"
            store = tmp_path / "stores.jsonl"
            output = tmp_path / "planner.jsonl"
            teacher_output = tmp_path / "planner.teacher.jsonl"
            locomo.write_text(
                json.dumps(
                    [
                        {
                            "conversation": {
                                "session_1_date_time": "1:56 pm on 8 May, 2023",
                                "session_1": [
                                    {"speaker": "Caroline", "dia_id": "D1:1", "text": "I am a transgender woman."},
                                    {"speaker": "Melanie", "dia_id": "D1:2", "text": "Thank you for sharing."},
                                ],
                            },
                            "qa": [{"question": "What is Caroline's identity?", "answer": "Transgender woman", "evidence": ["D1:1"], "category": 1}],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            build_locomo_memcell(locomo, store, core_mode="heuristic")
            store_rows = read_jsonl(store)
            self.assertTrue(any(row.get("store") == "core" and row.get("field") == "identity" and row.get("entity") == "Caroline" for row in store_rows))
            summary = prepare_planner_data(locomo, output)
            self.assertEqual(summary["samples"], 1)
            rows = read_jsonl(output)
            self.assertNotIn("planner_target", rows[0])
            self.assertEqual(rows[0]["person"], "Caroline")
            self.assertEqual(rows[0]["persons"], ["Caroline"])
            self.assertEqual(rows[0]["planner_input"]["person"], "Caroline")

            class FakeTeacher:
                settings = SimpleNamespace(chat_model="fake-teacher")

                def chat(self, messages, temperature=0.0):  # type: ignore[no-untyped-def]
                    del messages, temperature
                    return json.dumps(
                        {
                            "tool_calls": [
                                {
                                    "name": "search_core_memory",
                                    "arguments": {
                                        "queries": ["Caroline transgender woman identity"],
                                        "fields": ["identity"],
                                    },
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )

            teacher_summary = generate_teacher_sft_data(output, teacher_output, store, FakeTeacher(), candidates_per_sample=1)  # type: ignore[arg-type]
            self.assertEqual(teacher_summary["samples"], 1)
            teacher_rows = read_jsonl(teacher_output)
            self.assertEqual(teacher_rows[0]["label_source"], "teacher_distilled")
            self.assertIn("planner_target", teacher_rows[0])
            self.assertIn("tool_results", teacher_rows[0])
            core_memories = teacher_rows[0]["tool_results"]["tool_results"][0]["result"]["memories"]
            self.assertTrue(any(memory["field"] == "identity" and memory["entity"] == "Caroline" for memory in core_memories))

    def test_locomo_planner_data_extracts_question_person_from_speakers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            locomo = tmp_path / "locomo.json"
            output = tmp_path / "planner.jsonl"
            locomo.write_text(
                json.dumps(
                    [
                        {
                            "speaker_a": "Caroline",
                            "speaker_b": "Melanie",
                            "conversation": {},
                            "qa": [
                                {
                                    "question": "When did Caroline go to the LGBTQ support group?",
                                    "answer": "7 May 2023",
                                    "evidence": ["D1:3"],
                                    "category": 2,
                                },
                                {
                                    "question": "What subject have Caroline and Melanie both painted?",
                                    "answer": "horses",
                                    "evidence": ["D2:1"],
                                    "category": 3,
                                },
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prepare_planner_data(locomo, output, include_schema=False)
            rows = read_jsonl(output)
            self.assertEqual(rows[0]["person"], "Caroline")
            self.assertEqual(rows[0]["persons"], ["Caroline"])
            self.assertEqual(rows[0]["planner_input"]["person"], "Caroline")
            self.assertEqual(rows[1]["person"], "Caroline")
            self.assertEqual(rows[1]["persons"], ["Caroline", "Melanie"])

    def test_locomo_planner_data_uses_annotated_person_and_scoped_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            locomo = tmp_path / "locomo.json"
            output = tmp_path / "planner.jsonl"
            locomo.write_text(
                json.dumps(
                    [
                        {
                            "speaker_a": "Caroline",
                            "speaker_b": "Melanie",
                            "conversation": {},
                            "qa": [
                                {
                                    "question": "What did the charity race raise awareness for?",
                                    "answer": "mental health",
                                    "evidence": ["D2:2"],
                                    "category": 4,
                                    "person": "Melanie",
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prepare_planner_data(locomo, output, include_schema=False)
            rows = read_jsonl(output)
            self.assertEqual(rows[0]["person"], "Melanie")
            self.assertEqual(rows[0]["persons"], ["Melanie"])
            self.assertEqual(rows[0]["evidence"], ["locomo_0:D2:2"])
            self.assertEqual(rows[0]["planner_input"]["person"], "Melanie")

    def test_locomo_split_uses_first_second_and_remaining_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            locomo = tmp_path / "locomo.json"
            locomo.write_text(
                json.dumps(
                    [
                        {
                            "conversation": {},
                            "qa": [{"question": f"Question {index}", "answer": f"Answer {index}", "evidence": [f"D{index}:1"], "category": 1}],
                        }
                        for index in range(10)
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            outputs = {}
            for split in ["train", "valid", "test"]:
                output = tmp_path / f"{split}.jsonl"
                summary = prepare_planner_data(locomo, output, split=split)
                rows = read_jsonl(output)
                outputs[split] = (summary, rows)
            self.assertEqual([row["conversation_index"] for row in outputs["train"][1]], [0])
            self.assertEqual([row["conversation_index"] for row in outputs["valid"][1]], [1])
            self.assertEqual([row["conversation_index"] for row in outputs["test"][1]], list(range(2, 10)))

    def test_answer_eval_skip_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            planner_data = tmp_path / "planner.jsonl"
            answers = tmp_path / "answers.jsonl"
            summary = tmp_path / "summary.json"
            planner_data.write_text(
                json.dumps(
                    {
                        "conversation_index": 0,
                        "qa_index": 0,
                        "question": "What is Caroline's identity?",
                        "answer": "Transgender woman",
                        "generated_answer": "Caroline is a transgender woman.",
                        "category": 1,
                        "evidence": ["D1:1"],
                        "memory_context": {"memory_context": "Caroline said she is a transgender woman."},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            result = run_answer_eval(planner_data, answers, summary_path=summary, skip_generation=True)
            self.assertEqual(result["samples"], 1)
            self.assertEqual(result["contains_accuracy"], 1.0)
            rows = read_jsonl(answers)
            self.assertEqual(rows[0]["eval"]["contains_match"], True)

    def test_exports_sft_and_rl_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            planner_data = tmp_path / "planner.jsonl"
            sft_output = tmp_path / "sft.jsonl"
            rl_output = tmp_path / "rl.jsonl"
            planner_data.write_text(
                json.dumps(
                    {
                        "conversation_index": 0,
                        "qa_index": 0,
                        "split": "train",
                        "question": "What is Caroline's identity?",
                        "answer": "Transgender woman",
                        "evidence": ["D1:1"],
                        "category": 1,
                        "planner_input": {"question": "What is Caroline's identity?", "available_memory_tools": []},
                        "label_source": "teacher_distilled",
                        "planner_target": {
                            "tool_calls": [
                                {
                                    "name": "search_core_memory",
                                    "arguments": {
                                        "queries": ["Caroline transgender woman identity"],
                                        "fields": ["identity"],
                                    },
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            sft_summary = export_sft_data(planner_data, sft_output)
            rl_summary = export_rl_data(planner_data, rl_output, stores_path="stores.jsonl")
            self.assertEqual(sft_summary["samples"], 1)
            self.assertEqual(rl_summary["samples"], 1)
            sft_rows = read_jsonl(sft_output)
            rl_rows = read_jsonl(rl_output)
            self.assertIn("Memory Tool Planner", sft_rows[0]["prompt"])
            self.assertIn('"tool_calls"', sft_rows[0]["response"])
            self.assertNotIn("question_id", sft_rows[0]["input"])
            self.assertNotIn("timezone", sft_rows[0]["input"])
            self.assertNotIn("memory_tool_schema", sft_rows[0]["input"])
            self.assertNotIn("call_id", sft_rows[0]["target"]["tool_calls"][0])
            self.assertNotIn("purpose", sft_rows[0]["target"]["tool_calls"][0])
            self.assertNotIn("top_k", sft_rows[0]["target"]["tool_calls"][0]["arguments"])
            self.assertNotIn('"call_id"', sft_rows[0]["response"])
            self.assertNotIn('"purpose"', sft_rows[0]["response"])
            self.assertNotIn('"top_k"', sft_rows[0]["response"])
            self.assertEqual(sft_rows[0]["messages"][0]["role"], "user")
            self.assertEqual(sft_rows[0]["messages"][1]["role"], "assistant")
            self.assertFalse(sft_rows[0]["enable_thinking"])
            self.assertEqual(rl_rows[0]["prompt"][0]["role"], "user")
            self.assertEqual(rl_rows[0]["extra_info"]["stores_path"], "stores.jsonl")

    def test_export_sft_rejects_unlabeled_planner_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            planner_data = tmp_path / "planner.jsonl"
            sft_output = tmp_path / "sft.jsonl"
            planner_data.write_text(
                json.dumps(
                    {
                        "conversation_index": 0,
                        "qa_index": 0,
                        "question": "What is Caroline's identity?",
                        "planner_input": {"question": "What is Caroline's identity?", "available_memory_tools": []},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                export_sft_data(planner_data, sft_output)

    def test_reward_runner_scores_valid_teacher_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            locomo = tmp_path / "locomo.json"
            store = tmp_path / "stores.jsonl"
            planner_data = tmp_path / "planner.jsonl"
            rewards = tmp_path / "rewards.jsonl"
            locomo.write_text(
                json.dumps(
                    [
                        {
                            "conversation": {
                                "session_1_date_time": "1:56 pm on 8 May, 2023",
                                "session_1": [
                                    {"speaker": "Caroline", "dia_id": "D1:1", "text": "I am a transgender woman."},
                                    {"speaker": "Melanie", "dia_id": "D1:2", "text": "Thank you for sharing."},
                                ],
                            },
                            "qa": [{"question": "What is Caroline's identity?", "answer": "Transgender woman", "evidence": ["D1:1"], "category": 1}],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            build_locomo_memcell(locomo, store, core_mode="heuristic")
            prepare_planner_data(locomo, planner_data)

            class FakeTeacher:
                settings = SimpleNamespace(chat_model="fake-teacher")

                def chat(self, messages, temperature=0.0):  # type: ignore[no-untyped-def]
                    del messages, temperature
                    return json.dumps(
                        {
                            "tool_calls": [
                                {
                                    "name": "search_core_memory",
                                    "arguments": {
                                        "queries": ["Caroline transgender woman identity"],
                                        "fields": ["identity"],
                                    },
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )

            teacher_planner_data = tmp_path / "planner.teacher.jsonl"
            generate_teacher_sft_data(planner_data, teacher_planner_data, store, FakeTeacher(), candidates_per_sample=1)  # type: ignore[arg-type]
            summary = run_reward_runner(teacher_planner_data, store, rewards, use_gold_plan=True, skip_generation=True)
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["planner_valid_rate"], 1.0)
            reward_rows = read_jsonl(rewards)
            self.assertTrue(reward_rows[0]["planner_valid"])
            self.assertGreaterEqual(reward_rows[0]["reward"]["evidence"], 0.0)

    def test_reward_runner_scopes_memory_store_by_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = tmp_path / "stores.jsonl"
            planner_data = tmp_path / "planner.jsonl"
            rewards = tmp_path / "rewards.jsonl"
            store.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "store": "core",
                                "memory_id": "core_alex_locomo_0_000001",
                                "conversation_index": 0,
                                "content": "shared identity clue",
                                "entity": "Alex",
                                "field": "identity",
                                "evidence_ids": ["D1:1"],
                                "search_text": "shared identity clue",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "store": "core",
                                "memory_id": "core_blair_locomo_1_000001",
                                "conversation_index": 1,
                                "content": "shared identity clue",
                                "entity": "Blair",
                                "field": "identity",
                                "evidence_ids": ["D1:1"],
                                "search_text": "shared identity clue",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            planner_data.write_text(
                json.dumps(
                    {
                        "conversation_index": 1,
                        "qa_index": 0,
                        "question": "Who is Blair?",
                        "answer": "Blair",
                        "evidence": ["locomo_1:D1:1"],
                        "category": 1,
                        "planner_output": {
                            "tool_calls": [
                                {
                                    "name": "search_core_memory",
                                    "arguments": {"queries": ["shared identity clue"], "fields": ["identity"]},
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            run_reward_runner(planner_data, store, rewards, skip_generation=True)
            reward_rows = read_jsonl(rewards)
            self.assertEqual(reward_rows[0]["retrieved_evidence_ids"], ["locomo_1:D1:1"])
            self.assertEqual(reward_rows[0]["reward"]["evidence"], 1.0)

    def test_naive_rag_plan_uses_question_and_person_for_both_tools(self) -> None:
        plan = naive_rag_plan({"question": "Where did the visit happen?", "person": "Caroline"})
        calls = plan["tool_calls"]
        self.assertEqual([call["name"] for call in calls], ["search_core_memory", "search_episodic_memory"])
        self.assertIn("Caroline", calls[0]["arguments"]["queries"][0])
        self.assertEqual(set(calls[0]["arguments"]["fields"]), {"identity", "preference", "allergy", "goal", "constraint", "relationship"})

    def test_oracle_planner_uses_gold_evidence_without_gold_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "stores.jsonl"
            store.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "store": "episodic",
                                "memory_id": "locomo_0_epi_1",
                                "conversation_index": 0,
                                "event": "Caroline visited the greenhouse.",
                                "evidence_ids": ["locomo_0:D1:1"],
                                "search_text": "Caroline visited the greenhouse.",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "store": "core",
                                "memory_id": "locomo_0_core_1",
                                "conversation_index": 0,
                                "content": "Caroline is a gardener.",
                                "entity": "Caroline",
                                "field": "identity",
                                "evidence_ids": ["locomo_0:D1:2"],
                                "search_text": "Caroline gardener identity.",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            executor = MemoryToolExecutor.from_jsonl(store)
            row = {"question": "What happened?", "evidence": ["locomo_0:D1:1"]}
            plan = oracle_planner_plan(row, executor)
            self.assertEqual(plan["tool_calls"][0]["name"], "search_episodic_memory")
            self.assertIn("greenhouse", plan["tool_calls"][0]["arguments"]["queries"][0])

    def test_run_planner_baselines_writes_reward_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = tmp_path / "stores.jsonl"
            planner_data = tmp_path / "planner.jsonl"
            output_dir = tmp_path / "baselines"
            summary_path = tmp_path / "summary.json"
            store.write_text(
                json.dumps(
                    {
                        "store": "episodic",
                        "memory_id": "locomo_0_epi_1",
                        "conversation_index": 0,
                        "event": "Caroline visited the greenhouse.",
                        "evidence_ids": ["locomo_0:D1:1"],
                        "search_text": "Caroline visited the greenhouse.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            planner_data.write_text(
                json.dumps(
                    {
                        "conversation_index": 0,
                        "qa_index": 0,
                        "question_id": "locomo_0_qa0",
                        "question": "Where did Caroline visit?",
                        "person": "Caroline",
                        "answer": "greenhouse",
                        "evidence": ["locomo_0:D1:1"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            summary = run_planner_baselines(
                planner_data,
                store,
                output_dir,
                baselines=["no_memory", "oracle_planner"],
                summary_path=summary_path,
            )
            self.assertEqual(set(summary["baselines"]), {"no_memory", "oracle_planner"})
            self.assertEqual(summary["baselines"]["oracle_planner"]["planner_valid_rate"], 1.0)
            self.assertTrue((output_dir / "oracle_planner.reward_traces.jsonl").exists())
            self.assertTrue(summary_path.exists())

    def test_reward_breakdown_accepts_custom_weights(self) -> None:
        reward = reward_breakdown(
            valid=True,
            answer_score=1.0,
            evidence_score=0.5,
            memory_tokens=1000,
            tool_count=2,
            memory_token_budget=2000,
            weights={"answer": 1.2, "evidence": 0.8, "token": 0.05, "tool": 0.025, "under_tool": 0.0},
        )
        self.assertEqual(reward["weights"]["answer"], 1.2)
        self.assertEqual(reward["total"], 1.5625)

    def test_answer_reward_uses_partial_token_f1_for_wrong_llm_judgment(self) -> None:
        old = os.environ.get("MTP_REWARD_PARTIAL_ANSWER_WEIGHT")
        os.environ["MTP_REWARD_PARTIAL_ANSWER_WEIGHT"] = "0.25"
        try:
            reward = answer_reward({"llm_correct": False, "token_f1": 0.8}, "llm")
        finally:
            if old is None:
                os.environ.pop("MTP_REWARD_PARTIAL_ANSWER_WEIGHT", None)
            else:
                os.environ["MTP_REWARD_PARTIAL_ANSWER_WEIGHT"] = old
        self.assertEqual(reward, 0.2)

    def test_reward_token_cost_is_per_thousand_tokens(self) -> None:
        reward = reward_breakdown(
            valid=True,
            answer_score=0.0,
            evidence_score=0.0,
            memory_tokens=2500,
            tool_count=3,
            memory_token_budget=1000,
            weights={"answer": 0.0, "evidence": 0.0, "token": 0.2, "tool": 0.0, "under_tool": 0.0},
        )
        self.assertEqual(reward["token_cost"], 2.5)
        self.assertEqual(reward["tool_cost"], 0.75)
        self.assertEqual(reward["total"], -0.5)

    def test_reward_breakdown_penalizes_under_tooling(self) -> None:
        reward = reward_breakdown(
            valid=True,
            answer_score=1.0,
            evidence_score=1.0,
            memory_tokens=1000,
            tool_count=1,
            weights={"answer": 1.0, "evidence": 1.0, "token": 0.0, "tool": 0.0, "under_tool": 0.3},
            min_tool_count=2,
        )
        self.assertEqual(reward["under_tool_penalty"], 1)
        self.assertEqual(reward["total"], 1.7)

    def test_trace_analysis_pairs_outputs_and_reward_ablations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sft_trace = tmp_path / "sft.jsonl"
            grpo_trace = tmp_path / "grpo.jsonl"
            paired = tmp_path / "paired.jsonl"
            summary_path = tmp_path / "summary.json"

            sft_trace.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question_id": "q1",
                                "question": "Who is Caroline?",
                                "category": 1,
                                "planner_valid": True,
                                "eval": {"llm_correct": True},
                                "reward": {"answer": 1.0, "evidence": 1.0, "total": 1.0},
                                "memory_tokens": 1000,
                                "tool_count": 2,
                                "planner_output": {
                                    "tool_calls": [
                                        {
                                            "name": "search_episodic_memory",
                                            "arguments": {"queries": ["Caroline identity"]},
                                        },
                                        {
                                            "name": "search_core_memory",
                                            "arguments": {"queries": ["Caroline identity"], "fields": ["identity"]},
                                        },
                                    ]
                                },
                                "gold_evidence_ids": ["e1"],
                                "retrieved_evidence_ids": ["e1"],
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "question_id": "q2",
                                "question": "When did it happen?",
                                "category": 2,
                                "planner_valid": True,
                                "eval": {"llm_correct": False},
                                "reward": {"answer": 0.0, "evidence": 0.5, "total": 0.1},
                                "memory_tokens": 800,
                                "tool_count": 1,
                                "planner_output": {
                                    "tool_calls": [
                                        {
                                            "name": "search_episodic_memory",
                                            "arguments": {"queries": ["When did it happen?"]},
                                        }
                                    ]
                                },
                                "gold_evidence_ids": ["e2", "e3"],
                                "retrieved_evidence_ids": ["e2"],
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            grpo_trace.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question_id": "q1",
                                "question": "Who is Caroline?",
                                "category": 1,
                                "planner_valid": True,
                                "eval": {"llm_correct": False},
                                "reward": {"answer": 0.0, "evidence": 0.0, "total": -0.01},
                                "memory_tokens": 200,
                                "tool_count": 1,
                                "planner_output": {
                                    "tool_calls": [
                                        {
                                            "name": "search_core_memory",
                                            "arguments": {"queries": ["Caroline identity"], "fields": ["identity"]},
                                        }
                                    ]
                                },
                                "gold_evidence_ids": ["e1"],
                                "retrieved_evidence_ids": [],
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "question_id": "q2",
                                "question": "When did it happen?",
                                "category": 2,
                                "planner_valid": True,
                                "eval": {"llm_correct": True},
                                "reward": {"answer": 1.0, "evidence": 0.5, "total": 1.1},
                                "memory_tokens": 700,
                                "tool_count": 1,
                                "planner_output": {
                                    "tool_calls": [
                                        {
                                            "name": "search_episodic_memory",
                                            "arguments": {"queries": ["When did it happen?"]},
                                        }
                                    ]
                                },
                                "gold_evidence_ids": ["e2", "e3"],
                                "retrieved_evidence_ids": ["e2"],
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_trace_analysis(sft_trace, grpo_trace, paired, summary_path=summary_path)
            self.assertEqual(summary["paired"]["paired_samples"], 2)
            self.assertEqual(summary["paired"]["outcomes"]["baseline_wins"], 1)
            self.assertEqual(summary["paired"]["outcomes"]["candidate_wins"], 1)
            self.assertEqual(summary["paired"]["planner_outcomes"]["baseline_wins"], 1)
            self.assertIn("by_conversation", summary["baseline"])
            self.assertIn("by_tool_set", summary["candidate"])
            self.assertIn("evidence_heavy", summary["paired"]["reward_ablation_deltas"])
            self.assertEqual(summary["baseline"]["evidence_hit_rate"], 1.0)
            self.assertEqual(summary["candidate"]["planner_failure_buckets"]["no_retrieved_evidence"], 1)
            self.assertEqual(summary["baseline"]["tool_name_counts"]["search_episodic_memory"], 2)
            rows = read_jsonl(paired)
            self.assertEqual(rows[0]["question_id"], "q1")
            self.assertEqual(rows[0]["outcome"], "baseline_wins")
            self.assertLess(rows[0]["memory_tokens_delta"], 0)
            self.assertTrue(summary_path.exists())

    def test_trace_analysis_detects_normalized_duplicate_queries(self) -> None:
        row = {
            "planner_output": {
                "tool_calls": [
                    {"name": "search_episodic_memory", "arguments": {"queries": ["Caroline, greenhouse!"]}},
                    {"name": "search_episodic_memory", "arguments": {"queries": ["caroline greenhouse"]}},
                ]
            }
        }
        self.assertEqual(normalized_duplicate_query_count(row), 1)

    def test_build_hard_rl_data_preserves_default_reasons(self) -> None:
        build_hard_rows = load_build_hard_rows()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace = tmp_path / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    [
                        json.dumps({"question_id": "q1", "reward": {"evidence": 0.5}}, ensure_ascii=False),
                        json.dumps({"question_id": "q2", "reward": {"evidence": 0.0}}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            base_rows = [
                {"extra_info": {"question_id": "q1", "evidence": ["e1"]}},
                {"extra_info": {"question_id": "q2", "evidence": ["e2", "e3"]}},
            ]
            args = SimpleNamespace(
                input="base.parquet",
                trace=[str(trace)],
                reference_trace=None,
                output="hard.parquet",
                hard_threshold=1.0,
                hard_repeat=3,
                multi_evidence_repeat=3,
                no_overlap_repeat=3,
                zero_threshold=0.0,
                disagreement_threshold=0.5,
                disagreement_repeat=0,
                reference_gap_threshold=0.25,
                reference_gap_repeat=0,
                missing_trace_repeat=0,
            )

            rows, summary, hard_questions = build_hard_rows(base_rows, args)

            self.assertEqual(summary["output_rows"], 14)
            self.assertEqual(summary["repeat_counts"], {4: 1, 10: 1})
            self.assertEqual(summary["reasons"]["evidence_lt_1"], 2)
            self.assertEqual(summary["reasons"]["evidence_eq_0"], 1)
            self.assertEqual(summary["reasons"]["multi_evidence"], 1)
            self.assertEqual(len(hard_questions), 2)
            q2_rows = [row for row in rows if row["extra_info"]["question_id"] == "q2"]
            self.assertEqual(len(q2_rows), 10)
            self.assertEqual(q2_rows[0]["extra_info"]["hard_reasons"], ["evidence_lt_1", "evidence_eq_0", "multi_evidence"])

    def test_build_hard_rl_data_mines_multi_trace_reference_gaps(self) -> None:
        build_hard_rows = load_build_hard_rows()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_a = tmp_path / "trace_a.jsonl"
            trace_b = tmp_path / "trace_b.jsonl"
            oracle = tmp_path / "oracle.jsonl"
            trace_a.write_text(
                "\n".join(
                    [
                        json.dumps({"question_id": "q1", "reward": {"evidence": 0.2}}, ensure_ascii=False),
                        json.dumps({"question_id": "q2", "reward": {"evidence": 1.0}}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            trace_b.write_text(
                "\n".join(
                    [
                        json.dumps({"question_id": "q1", "reward": {"evidence": 0.8}}, ensure_ascii=False),
                        json.dumps({"question_id": "q3", "reward": {"evidence": 0.7}}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            oracle.write_text(
                "\n".join(
                    [
                        json.dumps({"question_id": "q1", "reward": {"evidence": 1.0}}, ensure_ascii=False),
                        json.dumps({"question_id": "q2", "reward": {"evidence": 1.0}}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            base_rows = [
                {"extra_info": {"question_id": "q1", "evidence": ["e1"]}},
                {"extra_info": {"question_id": "q2", "evidence": ["e2"]}},
                {"extra_info": {"question_id": "q3", "evidence": ["e3"]}},
            ]
            args = SimpleNamespace(
                input="base.parquet",
                trace=[str(trace_a), str(trace_b)],
                reference_trace=str(oracle),
                output="hard.parquet",
                hard_threshold=1.0,
                hard_repeat=1,
                multi_evidence_repeat=0,
                no_overlap_repeat=0,
                zero_threshold=0.0,
                disagreement_threshold=0.5,
                disagreement_repeat=2,
                reference_gap_threshold=0.15,
                reference_gap_repeat=3,
                missing_trace_repeat=0,
            )

            rows, summary, hard_questions = build_hard_rows(base_rows, args)

            self.assertEqual(summary["output_rows"], 10)
            self.assertEqual(summary["repeat_counts"], {1: 1, 2: 1, 7: 1})
            self.assertEqual(summary["reasons"]["trace_disagreement"], 1)
            self.assertEqual(summary["reasons"]["reference_gap"], 1)
            q1_hard = next(row for row in hard_questions if row["question_id"] == "q1")
            self.assertEqual(q1_hard["trace_scores"], [0.2, 0.8])
            self.assertAlmostEqual(q1_hard["reference_gap"], 0.2)
            q1_rows = [row for row in rows if row["extra_info"]["question_id"] == "q1"]
            self.assertEqual(len(q1_rows), 7)
            self.assertEqual(q1_rows[0]["extra_info"]["hard_trace_score_min"], 0.2)
            self.assertEqual(q1_rows[0]["extra_info"]["hard_reference_evidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
