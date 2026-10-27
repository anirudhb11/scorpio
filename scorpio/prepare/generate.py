"""Stage 1: generate responses for each problem and grade them."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from scorpio.core.llm import LLM
from scorpio.core.types import MAX_CONCURRENT_REQUESTS, Problem


def _one_response(llm: LLM, problem: Problem, max_tokens: int):
    call = llm.chat(problem.messages, max_tokens=max_tokens)
    return call.texts[0], call.finish_reasons[0], call.input_tokens, call.output_tokens


def prepare_problem(llm: LLM, problem: Problem, grade, num_responses: int,
                    max_tokens: int) -> dict:
    """Generate `num_responses` independent responses and grade each.

    Each response is its own request, so sampling is independent and a failure
    costs one response rather than the whole set. Problems are handled one at
    a time; parallelising those too would put problems x responses in flight.
    """
    workers = min(MAX_CONCURRENT_REQUESTS, num_responses)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda _: _one_response(llm, problem, max_tokens), range(num_responses)))

    responses = [r[0] for r in results]
    rewards = [grade(text, problem.ground_truth) for text in responses]

    return {
        "problem_id": problem.problem_id,
        "responses": responses,
        "finish_reasons": [r[1] for r in results],
        "rewards": rewards,
        "pass_at_1": sum(rewards) / len(rewards),
        "pass_at_n": max(rewards),
        "tokens": {
            "generation_input": sum(r[2] for r in results),
            "generation_output": sum(r[3] for r in results),
        },
    }
