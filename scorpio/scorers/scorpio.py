"""Rate candidates in groups, so the judge compares rather than grades alone.

Each round draws a fresh partition of all candidates into groups of
`group_size`, and one call rates a whole group. A candidate's score for the
round is the mean of the ratings it received -- usually one, but a padded
tail group can rate it twice.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from scorpio.core.clip import clip, tokenizer_for
from scorpio.core.types import JUDGE_OUTPUT_TOKENS, MAX_CONCURRENT_REQUESTS
from scorpio.prompts import load

NEUTRAL = 5


def parse_ratings(text: str, k: int) -> list[int]:
    """The k ratings, clamped to 1-10. A missing tag is neutral."""
    ratings = [NEUTRAL] * k
    if not text:
        return ratings
    for i in range(k):
        match = re.search(rf"<rating_{i + 1}>\s*(\d+)\s*</rating_{i + 1}>", text)
        if match:
            ratings[i] = max(1, min(10, int(match.group(1))))
    return ratings


def partition(n: int, k: int, rng) -> list[list[int]]:
    """Shuffle all n candidates and cut into groups of k.

    A short tail group is padded from the rest, so every call rates exactly
    k candidates. Padded candidates are rated twice this round, which the
    mean absorbs.
    """
    order = list(range(n))
    rng.shuffle(order)
    groups = []
    for start in range(0, n, k):
        group = order[start:start + k]
        if 0 < len(group) < k:
            pool = [c for c in order if c not in group]
            group = group + rng.sample(pool, min(k - len(group), len(pool)))
        groups.append(group)
    return groups


class Scorpio:
    def __init__(self, task: str, group_size: int = 4, model_path: str | None = None):
        self.template = load(f"scorpio_{task}")
        self.group_size = group_size
        self.model_path = model_path

    def prompt(self, problem: str, candidates: list[str]) -> str:
        blocks = "\n\n".join(f"**Solution {i + 1}**\n{c}" for i, c in enumerate(candidates))
        tags = "\n".join(f"<rating_{i + 1}>INTEGER_1_TO_10</rating_{i + 1}>"
                         for i in range(len(candidates)))
        return (self.template.replace("{k}", str(len(candidates)))
                .replace("{problem}", problem)
                .replace("{solutions}", blocks)
                .replace("{rating_lines}", tags))

    def run_round(self, problem: str, judge_inputs: list[str], scores_so_far, llm, rng,
                  calls_so_far=None):
        k = min(self.group_size, len(judge_inputs))
        candidates = clip(judge_inputs, tokenizer_for(self.model_path), k)
        groups = partition(len(candidates), k, rng)

        def rate(group):
            text = self.prompt(problem, [candidates[i] for i in group])
            call = llm.chat([{"role": "user", "content": text}],
                            max_tokens=JUDGE_OUTPUT_TOKENS, candidate_ids=group)
            return group, parse_ratings(call.texts[0], len(group))

        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(groups))) as pool:
            results = list(pool.map(rate, groups))

        totals = [0.0] * len(candidates)
        counts = [0] * len(candidates)
        for group, ratings in results:
            for i, rating in zip(group, ratings):
                totals[i] += rating
                counts[i] += 1
        return [t / c if c else float(NEUTRAL) for t, c in zip(totals, counts)]


def build(task: str, group_size: int = 4, model_path: str | None = None, **params):
    return Scorpio(task, group_size, model_path)
