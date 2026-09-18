"""Benchmark application: LLM-as-judge orchestration.

Scores a candidate answer against the cached gold answer on correctness +
completeness (1-5). The judge is blinded to which arm produced the candidate
to avoid arm bias, and is told to judge against the gold answer, not its own
knowledge.
"""
from __future__ import annotations

import asyncio

from treeloom.adapters.benchmark.agent_llm import AgentLLM
from treeloom.domain.benchmark.judge_schema import JudgeScore, parse_judge_json

_JUDGE_SYSTEM = (
    "You are a strict, fair evaluator for a code-search benchmark. Compare a "
    "candidate answer to a gold reference answer for a question about a codebase. "
    "Score two axes from 1 to 5. correctness = does the candidate identify the "
    "correct file(s)/entity and make accurate claims (5 = fully correct, 1 = "
    "wrong). completeness = does it cover the key points the gold answer covers "
    "(5 = covers all, 1 = misses everything). Judge ONLY against the gold answer "
    "and the question, not your own knowledge. Output ONLY a JSON object. /no_think"
)

_JUDGE_USER = (
    "Question: {query}\n\n"
    "Gold answer:\n{gold}\n\n"
    "Candidate answer:\n{candidate}\n\n"
    'Respond with JSON only: {{"correctness": <1-5>, "completeness": <1-5>, '
    '"rationale": "<one sentence>"}}'
)


async def judge_answer(
    query: str,
    gold_answer: str,
    candidate_answer: str,
    llm: AgentLLM,
) -> JudgeScore:
    if not gold_answer.strip():
        return JudgeScore(1, 1, "no gold answer available", parse_ok=False)
    if not candidate_answer.strip():
        return JudgeScore(1, 1, "empty candidate answer", parse_ok=True)
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": _JUDGE_USER.format(
                query=query, gold=gold_answer, candidate=candidate_answer
            ),
        },
    ]
    # One transient API failure (e.g. a ReadTimeout) must never kill the
    # benchmark run — retry once, then return a flagged 1/1 score, matching
    # the unparseable-output convention (visible, not silently dropped).
    last_err: Exception | None = None
    for attempt in (1, 2):
        try:
            content, _, _ = await llm.chat(messages, max_tokens=256)
            return parse_judge_json(content)
        except Exception as e:
            last_err = e
            if attempt == 1:
                await asyncio.sleep(2.0)
    return JudgeScore(1, 1, f"judge call failed: {last_err}", parse_ok=False)
