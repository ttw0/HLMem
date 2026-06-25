from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .answer import answer_question
from .answer_eval import run_answer_eval
from .baselines import run_planner_baselines
from .builder import build_locomo_memcell, build_sample_store
from .config import load_env
from .context import build_memory_context
from .embeddings import EmbeddingProvider
from .evermemos_import import build_evermemos_core_memory, import_evermemos_locomo
from .index import index_store
from .locomo_planner_data import prepare_planner_data
from .openai_protocol import OpenAIProtocolClient
from .planner import validate_planner_output
from .retrieval import MemoryToolExecutor
from .reward import run_reward_runner
from .teacher_sft import generate_teacher_sft_data
from .trace_analysis import run_trace_analysis
from .training_data import export_rl_data, export_sft_data


def cmd_build_sample(args: argparse.Namespace) -> int:
    build_sample_store(args.output)
    print(args.output)
    return 0


def _derived_path(path: str, suffix: str) -> str:
    return str(Path(path).with_suffix(suffix))


def cmd_build_locomo_memcell(args: argparse.Namespace) -> int:
    memcells_output = args.memcells_output or _derived_path(args.output, ".memcells.jsonl")
    core_slices_output = args.core_slices_output or _derived_path(args.output, ".core_slices.json")
    merged_core_output = args.merged_core_output or _derived_path(args.output, ".merged_core.json")
    stats = build_locomo_memcell(
        args.input,
        args.output,
        memcells_output=memcells_output,
        core_slices_output=core_slices_output,
        merged_core_output=merged_core_output,
        limit_conversations=args.limit_conversations,
        boundary_mode=args.boundary_mode,
        episode_mode=args.episode_mode,
        core_mode=args.core_mode,
        include_core=not args.no_core,
        max_core_slices=args.max_core_slices,
    )
    print(json.dumps({"output": args.output, "memcells_output": memcells_output, "stats": stats}, ensure_ascii=False))
    return 0


def cmd_import_evermemos_locomo(args: argparse.Namespace) -> int:
    summary = import_evermemos_locomo(
        args.memcells_dir,
        args.locomo,
        memcells_output=args.memcells_output,
        episodic_output=args.episodic_output,
        summary_output=args.summary,
        limit_conversations=args.limit_conversations,
        keep_fact_embeddings=args.keep_fact_embeddings,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_build_evermemos_core(args: argparse.Namespace) -> int:
    if args.online:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "0"
    if args.offline:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "1"
    summary = build_evermemos_core_memory(
        args.memcells,
        args.episodic,
        core_output=args.core_output,
        core_slices_output=args.core_slices_output,
        merged_core_output=args.merged_core_output,
        combined_output=args.combined_output,
        summary_output=args.summary,
        mode=args.core_mode,
        max_slices=args.max_slices,
        parallel_workers=args.parallel_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    if args.online:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "0"
    if args.offline:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "1"
    if args.embedding_input_format:
        os.environ["OPENAI_EMBEDDING_INPUT_FORMAT"] = args.embedding_input_format
    index_store(args.input, args.output, EmbeddingProvider.from_env(), batch_size=args.batch_size, max_text_chars=args.max_text_chars)
    print(args.output)
    return 0


def cmd_embedding_smoke(args: argparse.Namespace) -> int:
    if args.input_format:
        os.environ["OPENAI_EMBEDDING_INPUT_FORMAT"] = args.input_format
    if args.max_chars:
        os.environ["OPENAI_EMBEDDING_MAX_CHARS"] = str(args.max_chars)
    os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "0"
    vector = OpenAIProtocolClient.from_env().embed([args.text])[0]
    print(json.dumps({"ok": True, "dims": len(vector), "input_format": os.environ.get("OPENAI_EMBEDDING_INPUT_FORMAT", "array")}, ensure_ascii=False))
    return 0


def cmd_prepare_planner_data(args: argparse.Namespace) -> int:
    summary = prepare_planner_data(
        args.locomo,
        args.output,
        limit_conversations=args.limit_conversations,
        limit_qa=args.limit_qa,
        include_schema=not args.no_schema,
        split=args.split,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_generate_teacher_sft_data(args: argparse.Namespace) -> int:
    if args.online:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "0"
    if args.offline:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "1"
    if args.api_key:
        os.environ["OPENAI_TEACHER_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["OPENAI_TEACHER_BASE_URL"] = args.base_url
    if args.model:
        os.environ["OPENAI_TEACHER_MODEL"] = args.model
    client = OpenAIProtocolClient.from_env("TEACHER")
    summary = generate_teacher_sft_data(
        args.planner_data,
        args.output,
        args.stores,
        client,
        candidates_per_sample=args.candidates_per_sample,
        limit=args.limit,
        temperature=args.temperature,
        current_date=args.current_date,
        timezone=args.timezone,
        token_budget=args.token_budget,
        max_memory_chars=args.max_memory_chars,
    )
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    raw = Path(args.input).read_text(encoding="utf-8") if args.input else args.json
    result = validate_planner_output(raw)
    print(json.dumps({"valid": result.valid, "parsed": result.parsed, "errors": result.errors}, ensure_ascii=False, indent=2))
    return 0 if result.valid else 1


def cmd_execute(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    executor = MemoryToolExecutor.from_jsonl(args.stores)
    results = executor.execute_plan(plan)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    tool_results = json.loads(Path(args.tool_results).read_text(encoding="utf-8"))
    context = build_memory_context(tool_results, token_budget=args.token_budget)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    context_obj = json.loads(Path(args.context).read_text(encoding="utf-8"))
    answer = answer_question(args.question, context_obj.get("memory_context", ""), prompt_style=args.prompt_style)
    print(answer)
    return 0


def cmd_eval_answer(args: argparse.Namespace) -> int:
    if args.online:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "0"
    if args.offline:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "1"
    summary = run_answer_eval(
        args.planner_data,
        args.output,
        summary_path=args.summary,
        limit=args.limit,
        prompt_style=args.prompt_style,
        eval_mode=args.eval_mode,
        num_judge_runs=args.num_judge_runs,
        token_budget=args.token_budget,
        max_memory_chars=args.max_memory_chars,
        rebuild_context=args.rebuild_context,
        include_context=not args.no_context,
        skip_generation=args.skip_generation,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_export_sft_data(args: argparse.Namespace) -> int:
    summary = export_sft_data(
        args.planner_data,
        args.output,
        output_format=args.format,
        current_date=args.current_date,
        timezone=args.timezone,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_export_rl_data(args: argparse.Namespace) -> int:
    summary = export_rl_data(
        args.planner_data,
        args.output,
        stores_path=args.stores,
        output_format=args.format,
        current_date=args.current_date,
        timezone=args.timezone,
        drop_empty_evidence=args.drop_empty_evidence,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_reward_runner(args: argparse.Namespace) -> int:
    if args.online:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "0"
    if args.offline:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "1"
    summary = run_reward_runner(
        args.planner_data,
        args.stores,
        args.output,
        summary_path=args.summary,
        candidate_field=args.candidate_field,
        use_gold_plan=args.use_gold_plan,
        limit=args.limit,
        eval_mode=args.eval_mode,
        num_judge_runs=args.num_judge_runs,
        token_budget=args.token_budget,
        max_memory_chars=args.max_memory_chars,
        prompt_style=args.prompt_style,
        skip_generation=args.skip_generation,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_analyze_traces(args: argparse.Namespace) -> int:
    summary = run_trace_analysis(
        args.baseline_trace,
        args.candidate_trace,
        args.output,
        summary_path=args.summary,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        memory_token_budget=args.memory_token_budget,
        max_tool_calls=args.max_tool_calls,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_planner_baselines(args: argparse.Namespace) -> int:
    if args.online:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "0"
    if args.offline:
        os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "1"
    summary = run_planner_baselines(
        args.planner_data,
        args.stores,
        args.output_dir,
        baselines=args.baseline,
        summary_path=args.summary,
        limit=args.limit,
        eval_mode=args.eval_mode,
        token_budget=args.token_budget,
        skip_generation=args.skip_generation,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def default_plan() -> dict:
    return {
        "tool_calls": [
            {
                "call_id": "call_1",
                "name": "search_core_memory",
                "purpose": "检索用户和妈妈的饮食偏好、忌口和过敏约束",
                "arguments": {
                    "queries": ["妈妈 饮食偏好 忌口 过敏 清淡", "用户 饮食偏好 忌口 过敏 清淡"],
                    "fields": ["preference", "allergy", "constraint"],
                    "top_k": 5,
                },
            },
            {
                "call_id": "call_2",
                "name": "search_episodic_memory",
                "purpose": "检索北京行程和同行人上下文",
                "arguments": {
                    "queries": ["明天 和妈妈 去北京 行程 住宿 活动区域"],
                    "top_k": 5,
                },
            },
        ]
    }


def cmd_dry_run(args: argparse.Namespace) -> int:
    os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "1"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stores = out / "sample_memory_stores.jsonl"
    indexed = out / "sample_memory_stores.embedded.jsonl"
    plan_path = out / "planner_output.json"
    results_path = out / "tool_results.json"
    context_path = out / "memory_context.json"

    build_sample_store(stores)
    index_store(stores, indexed, EmbeddingProvider.from_env())
    plan_path.write_text(json.dumps(default_plan(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    executor = MemoryToolExecutor.from_jsonl(indexed)
    results = executor.execute_plan(default_plan())
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    context = build_memory_context(results)
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    answer = answer_question(args.question, context["memory_context"])
    (out / "answer.txt").write_text(answer + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "answer": answer}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtp")
    parser.add_argument("--env-file", default=".env")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-sample")
    p.add_argument("--output", default="outputs/v0/data/sample_memory_stores.jsonl")
    p.set_defaults(func=cmd_build_sample)

    p = sub.add_parser("build-locomo-memcell")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="outputs/v0/data/locomo_memcell_memory_stores.jsonl")
    p.add_argument("--memcells-output")
    p.add_argument("--core-slices-output")
    p.add_argument("--merged-core-output")
    p.add_argument("--limit-conversations", type=int)
    p.add_argument("--boundary-mode", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--episode-mode", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--core-mode", choices=["heuristic", "llm"], default="llm")
    p.add_argument("--max-core-slices", type=int)
    p.add_argument("--no-core", action="store_true")
    p.set_defaults(func=cmd_build_locomo_memcell)

    p = sub.add_parser("import-evermemos-locomo")
    p.add_argument("--memcells-dir", required=True)
    p.add_argument("--locomo", required=True)
    p.add_argument("--memcells-output", default="memory_store/locomo_evermemos_memcells.jsonl")
    p.add_argument("--episodic-output", default="memory_store/locomo_evermemos_episodic.jsonl")
    p.add_argument("--summary", default="memory_store/locomo_evermemos_import.summary.json")
    p.add_argument("--limit-conversations", type=int)
    p.add_argument("--keep-fact-embeddings", action="store_true")
    p.set_defaults(func=cmd_import_evermemos_locomo)

    p = sub.add_parser("build-evermemos-core")
    p.add_argument("--memcells", default="memory_store/locomo_evermemos_memcells.jsonl")
    p.add_argument("--episodic", default="memory_store/locomo_evermemos_episodic.jsonl")
    p.add_argument("--core-output", default="memory_store/locomo_evermemos_core.jsonl")
    p.add_argument("--core-slices-output", default="memory_store/locomo_evermemos_core_slices.json")
    p.add_argument("--merged-core-output", default="memory_store/locomo_evermemos_merged_core.json")
    p.add_argument("--combined-output", default="memory_store/locomo_evermemos_memory_stores.jsonl")
    p.add_argument("--summary", default="memory_store/locomo_evermemos_core.summary.json")
    p.add_argument("--core-mode", choices=["heuristic", "llm"], default="llm")
    p.add_argument("--max-slices", type=int)
    p.add_argument("--parallel-workers", type=int, default=1)
    p.add_argument("--online", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_build_evermemos_core)

    p = sub.add_parser("index")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--online", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-text-chars", type=int)
    p.add_argument("--embedding-input-format", choices=["array", "string", "auto"])
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("embedding-smoke")
    p.add_argument("--text", default="hello")
    p.add_argument("--input-format", choices=["array", "string", "auto"])
    p.add_argument("--max-chars", type=int)
    p.set_defaults(func=cmd_embedding_smoke)

    p = sub.add_parser("prepare-planner-data")
    p.add_argument("--locomo", required=True)
    p.add_argument("--output", default="outputs/v0/planner_data/locomo_planner_data.jsonl")
    p.add_argument("--limit-conversations", type=int)
    p.add_argument("--limit-qa", type=int)
    p.add_argument("--split", choices=["train", "valid", "eval", "test"])
    p.add_argument("--no-schema", action="store_true")
    p.set_defaults(func=cmd_prepare_planner_data)

    p = sub.add_parser("generate-teacher-sft-data")
    p.add_argument("--planner-data", required=True)
    p.add_argument("--stores", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key")
    p.add_argument("--online", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--candidates-per-sample", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--current-date", default="2026-05-08")
    p.add_argument("--timezone", default="Asia/Shanghai")
    p.add_argument("--token-budget", type=int, default=0)
    p.add_argument("--max-memory-chars", type=int)
    p.set_defaults(func=cmd_generate_teacher_sft_data)

    p = sub.add_parser("validate")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--json")
    group.add_argument("--input")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("execute")
    p.add_argument("--stores", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--output", default="outputs/v0/tool_results.json")
    p.set_defaults(func=cmd_execute)

    p = sub.add_parser("context")
    p.add_argument("--tool-results", required=True)
    p.add_argument("--output", default="outputs/v0/memory_context.json")
    p.add_argument("--token-budget", type=int, default=2000)
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("answer")
    p.add_argument("--question", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--prompt-style", choices=["concise", "cot"], default="concise")
    p.set_defaults(func=cmd_answer)

    p = sub.add_parser("eval-answer")
    p.add_argument("--planner-data", required=True)
    p.add_argument("--output", default="outputs/v0/answer_eval/answers.jsonl")
    p.add_argument("--summary", default="outputs/v0/answer_eval/summary.json")
    p.add_argument("--limit", type=int)
    p.add_argument("--online", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--prompt-style", choices=["concise", "cot"], default="concise")
    p.add_argument("--eval-mode", choices=["lexical", "llm", "hybrid"], default="lexical")
    p.add_argument("--num-judge-runs", type=int, default=1)
    p.add_argument("--token-budget", type=int, default=2000)
    p.add_argument("--max-memory-chars", type=int)
    p.add_argument("--rebuild-context", action="store_true")
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--skip-generation", action="store_true")
    p.set_defaults(func=cmd_eval_answer)

    p = sub.add_parser("export-sft-data")
    p.add_argument("--planner-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", choices=["auto", "jsonl", "parquet"], default="auto")
    p.add_argument("--current-date", default="2026-05-08")
    p.add_argument("--timezone", default="Asia/Shanghai")
    p.set_defaults(func=cmd_export_sft_data)

    p = sub.add_parser("export-rl-data")
    p.add_argument("--planner-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stores")
    p.add_argument("--format", choices=["auto", "jsonl", "parquet"], default="auto")
    p.add_argument("--current-date", default="2026-05-08")
    p.add_argument("--timezone", default="Asia/Shanghai")
    p.add_argument("--drop-empty-evidence", action="store_true")
    p.set_defaults(func=cmd_export_rl_data)

    p = sub.add_parser("reward-runner")
    p.add_argument("--planner-data", required=True)
    p.add_argument("--stores", required=True)
    p.add_argument("--output", default="outputs/v0/reward_debug/rewards.jsonl")
    p.add_argument("--summary", default="outputs/v0/reward_debug/summary.json")
    p.add_argument("--candidate-field", default="planner_output")
    p.add_argument("--use-gold-plan", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--online", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--prompt-style", choices=["concise", "cot"], default="concise")
    p.add_argument("--eval-mode", choices=["lexical", "llm", "hybrid"], default="lexical")
    p.add_argument("--num-judge-runs", type=int, default=1)
    p.add_argument("--token-budget", type=int, default=2000)
    p.add_argument("--max-memory-chars", type=int)
    p.add_argument("--skip-generation", action="store_true")
    p.set_defaults(func=cmd_reward_runner)

    p = sub.add_parser("analyze-traces")
    p.add_argument("--baseline-trace", required=True)
    p.add_argument("--candidate-trace", required=True)
    p.add_argument("--output", default="outputs/v0/diagnostics/sft_grpo_paired_errors.jsonl")
    p.add_argument("--summary", default="outputs/v0/diagnostics/sft_grpo_trace_analysis.summary.json")
    p.add_argument("--baseline-name", default="sft")
    p.add_argument("--candidate-name", default="grpo")
    p.add_argument("--memory-token-budget", type=int, default=0)
    p.add_argument("--max-tool-calls", type=int, default=4)
    p.set_defaults(func=cmd_analyze_traces)

    p = sub.add_parser("planner-baselines")
    p.add_argument("--planner-data", required=True)
    p.add_argument("--stores", required=True)
    p.add_argument("--output-dir", default="outputs/v0/baselines")
    p.add_argument("--summary", default="outputs/v0/baselines/summary.json")
    p.add_argument("--baseline", action="append", choices=["no_memory", "naive_rag", "oracle_planner"])
    p.add_argument("--limit", type=int)
    p.add_argument("--online", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--eval-mode", choices=["lexical", "llm", "hybrid"], default="lexical")
    p.add_argument("--token-budget", type=int, default=0)
    p.add_argument("--skip-generation", action="store_true", default=True)
    p.set_defaults(func=cmd_planner_baselines)

    p = sub.add_parser("dry-run")
    p.add_argument("--question", default="我明天和我妈妈去北京，有什么好吃的？")
    p.add_argument("--output", default="outputs/v0/dry_run")
    p.set_defaults(func=cmd_dry_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_env(args.env_file)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
