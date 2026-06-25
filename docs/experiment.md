# HLMem Experiment Overview

HLMem trains a planner that decides which memory tools to call and how to query
them. The planner does not answer the user directly. A frozen answer model may
be used later for end-to-end reporting, but the primary reinforcement learning
signal is retrieval evidence quality.

## Planner Protocol

Planner input contains:

- `question_id`
- `question`
- `current_date`
- `timezone`
- `memory_tool_schema`

Planner output must be valid JSON with a single top-level `tool_calls` field.
Each tool call contains a tool `name` and tool-specific `arguments`. The runtime
fills `call_id` and `top_k`.

## Tools

- `search_core_memory`: long-term profile, stable preference, relationship,
  allergy, goal, and constraint retrieval.
- `search_episodic_memory`: event, timeline, plan, place, participant, and
  temporary-state retrieval.

The executor returns real memory rows only. Empty retrievals are represented as
`{"memories": []}`.

## Reward

Planner-only GRPO uses:

- invalid planner output penalty
- evidence recall and evidence hit metrics
- tool-call and duplicate-query diagnostics

It disables answer generation, answer-judge reward, token cost, and tool cost by
default. This keeps the training target focused on cleaner retrieval behavior.

## Splits

The bundled LoCoMo example uses a fixed 1:1:8 conversation split:

- train: conversation index `0`
- validation: conversation index `1`
- test: conversation indices `2` through `9`

Evidence IDs are scoped as `locomo_{conversation_index}:{dia_id}` so evaluation
does not leak evidence across conversations.
