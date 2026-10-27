"""Rate each candidate on its own, 1-10."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from scorpio.core.clip import clip, tokenizer_for
from scorpio.core.types import JUDGE_OUTPUT_TOKENS, MAX_CONCURRENT_REQUESTS
from scorpio.prompts import load

RATING = re.compile(r"<rating>\s*(\d{1,2})\s*</rating>", re.IGNORECASE)
RATING_LOOSE = re.compile(r"rating[^\d]*(\d{1,2})", re.IGNORECASE)

NEUTRAL = 5


def parse_rating(text: str) -> int:
    """The LAST rating in the reply, clamped to 1-10.

    Last, because a judge that reasons aloud quotes the tag format before
    using it. A reply with no rating is neutral rather than fatal.
    """
    matches = RATING.findall(text) or RATING_LOOSE.findall(text)
    if not matches:
        return NEUTRAL
    return max(1, min(10, int(matches[-1])))


class Pointwise:
    def __init__(self, task: str, model_path: str | None = None):
        self.prompt = load(f"pointwise_{task}")
        self.model_path = model_path

    def run_round(self, problem: str, judge_inputs: list[str], scores_so_far, llm, rng,
                  calls_so_far=None):
        # One candidate per call, so the whole budget is its own.
        judge_inputs = clip(judge_inputs, tokenizer_for(self.model_path), 1)

        def rate(i_candidate):
            i, candidate = i_candidate
            text = self.prompt.replace("{problem}", problem).replace("{candidate}", candidate)
            call = llm.chat([{"role": "user", "content": text}],
                            max_tokens=JUDGE_OUTPUT_TOKENS, candidate_ids=[i])
            return float(parse_rating(call.texts[0]))

        workers = min(MAX_CONCURRENT_REQUESTS, len(judge_inputs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(rate, enumerate(judge_inputs)))


def build(task: str, model_path: str | None = None, **params) -> Pointwise:
    return Pointwise(task, model_path)
