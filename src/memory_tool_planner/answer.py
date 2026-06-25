from __future__ import annotations

from .config import OpenAISettings
from .openai_protocol import OpenAIProtocolClient


ANSWER_PROMPT_CONCISE = """
You are a knowledgeable and helpful memory assistant.

# CONTEXT
You have access to retrieved memories from conversations. These memories may
include episodic events and core profile facts.

# INSTRUCTIONS
1. Answer only from the provided memories.
2. If the question asks about a date or relative time, calculate the specific date from memory timestamps when possible.
3. Preserve exact names, dates, quantities, places, and proper nouns.
4. If the memories do not contain enough evidence, say "I don't know".
5. Keep the final answer brief and direct.

# RELATIVE TIME RULE
When a memory line has a timestamp and its content says "yesterday", "last week",
"last year", "next month", or a similar relative reference, the final answer
must use the calculated absolute date or period, not the memory timestamp itself.
Example: if a memory timestamp is 2023-05-08 and the content says "I went
yesterday", answer 2023-05-07.
Never use unresolved relative time expressions such as "yesterday", "last year",
"last week", or "next month" in the final answer when a timestamp is available.

{context}

Question: {question}

Answer:
"""


ANSWER_PROMPT_COT = """
You are an intelligent memory assistant tasked with retrieving accurate information from memories.

# CONTEXT
You have access to timestamped memories that may be relevant to answering the question.

# INSTRUCTIONS
Synthesize information from all relevant memories. Preserve exact names, numbers,
dates, times, frequencies, places, and proper nouns. Convert relative time
references to specific dates when the memory timestamp supports it. If multiple
memories conflict, prefer the most recent reliable memory. If the memories do
not contain enough evidence, answer "I don't know".

# RESPONSE FORMAT
## Relevant Memories
- [timestamp] content

## Key Evidence
- Names:
- Dates/Times:
- Other details:

## Final Answer
[concise answer]

---

{context}

Question: {question}
"""


def parse_final_answer(text: str) -> str:
    markers = ["## Final Answer", "## FINAL ANSWER", "FINAL ANSWER:", "Final Answer:", "最终答案：", "最终答案:"]
    for marker in markers:
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return text.strip()


def answer_question(question: str, memory_context: str, temperature: float = 0.0, prompt_style: str = "concise") -> str:
    settings = OpenAISettings.from_env("ANSWER")
    if settings.offline:
        if memory_context:
            return f"[offline answer] 基于检索到的记忆回答：{question}"
        return f"[offline answer] 未检索到个性化记忆，按通用知识回答：{question}"
    client = OpenAIProtocolClient(settings)
    prompt_template = ANSWER_PROMPT_COT if prompt_style == "cot" else ANSWER_PROMPT_CONCISE
    prompt = prompt_template.format(context=memory_context or "(empty)", question=question)
    content = client.chat([{"role": "user", "content": prompt}], temperature=temperature)
    return parse_final_answer(content)
