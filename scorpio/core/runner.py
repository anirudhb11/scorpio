"""Stage 2: run a scorer over a problem's candidates, for some rounds."""

from __future__ import annotations

import hashlib
import random


class RecordingLLM:
    """An LLM that keeps every call it makes.

    A scorer just calls `chat` and returns scores; the transcript is written
    for it, so a new scorer cannot forget to record one. `candidate_ids` says
    which candidates a prompt was about, which only the scorer knows.
    """

    def __init__(self, llm):
        self.llm = llm
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], max_tokens: int, candidate_ids=None):
        call = self.llm.chat(messages, max_tokens=max_tokens)
        self.calls.append({
            "candidate_ids": candidate_ids,
            "prompt": messages,
            "response": call.texts[0],
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
        })
        return call


def seed_for(experiment: str, scorer: str, problem_id: str, rep: int, rnd: int) -> int:
    """Reproducible per-round seed.

    A digest, not hash(): Python randomises string hashing per process, so
    hash() would draw a different partition on every run. The shard index is
    deliberately absent -- splitting the work differently must not change a
    result -- and the round is present, so a resumed round draws exactly what
    an uninterrupted run would have drawn.
    """
    key = f"{experiment}|{scorer}|{problem_id}|{rep}|{rnd}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def run_problem(scorer, problem_text: str, judge_inputs: list[str], llm,
                num_rounds: int, experiment: str, scorer_name: str,
                problem_id: str, rep: int, done: list[dict] | None = None):
    """Yield one record per round, so a round survives the job that follows it.

    `done` is the round records already on disk; those rounds are skipped and
    the scorer still sees them, as `scores_so_far` and as `calls_so_far`. A
    scorer whose next move depends on what it already asked -- V1 picks its
    pairs from who has met whom -- rebuilds that from the calls, so a resumed
    job continues exactly as an uninterrupted one would.
    """
    done = list(done or [])
    scores_so_far = [record["scores"] for record in done]
    calls_so_far = [call for record in done for call in record["calls"]]
    for rnd in range(len(scores_so_far), num_rounds):
        recorder = RecordingLLM(llm)
        rng = random.Random(seed_for(experiment, scorer_name, problem_id, rep, rnd))
        scores = scorer.run_round(problem_text, judge_inputs, scores_so_far, recorder, rng,
                                  calls_so_far)
        scores_so_far.append(scores)
        calls_so_far.extend(recorder.calls)
        yield {
            "problem_id": problem_id,
            "round": rnd,
            "scores": scores,
            "calls": recorder.calls,
        }
