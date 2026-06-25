from __future__ import annotations

from typing import Any

from memory_tool_planner.reward import compute_verl_score


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> float:
    return compute_verl_score(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
