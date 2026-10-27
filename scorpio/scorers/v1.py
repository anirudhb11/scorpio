"""Rank candidates by pairwise comparison -- V1-Infer (Singh et al.).

One round is one batch of disjoint pairs judged in parallel, which is how the
original batches them, so `num_rounds` is the verification budget. Round 0 is
a random disjoint pairing so nobody starts unseen; rounds after it lift every
candidate to MIN_DEGREE distinct opponents; the rest are Swiss rounds pairing
near neighbours in the current ranking.

A round's score is mu, the weighted win rate over every comparison made so
far -- cumulative, not per-round, since one batch gives a candidate a single
comparison. So the score after k rounds is round k's mu, and a report over V1
must read the last round rather than average rounds.

The scorer keeps nothing between rounds. It rebuilds its tally by replaying
the judge calls already recorded for this problem, so a resumed job pairs
exactly as an uninterrupted one would.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from scorpio.core.clip import clip, tokenizer_for
from scorpio.core.types import JUDGE_OUTPUT_TOKENS, MAX_CONCURRENT_REQUESTS
from scorpio.prompts import load

MIN_DEGREE = 2   # d_min: distinct opponents every candidate must face
WINDOW = 8       # h: how far down the ranking a Swiss partner may be found
TAU = 0.1        # weight floor, so a tied comparison still carries weight
NEUTRAL = 5

RATING_A = re.compile(r"<rating_A>\s*(\d{1,2})\s*</rating_A>", re.IGNORECASE)
RATING_B = re.compile(r"<rating_B>\s*(\d{1,2})\s*</rating_B>", re.IGNORECASE)
LOOSE_A = re.compile(r"rating_A[^\d]*(\d{1,2})", re.IGNORECASE)
LOOSE_B = re.compile(r"rating_B[^\d]*(\d{1,2})", re.IGNORECASE)


def parse_pair(text: str) -> tuple[int, int]:
    """The LAST rating for each side, clamped to 1-10.

    Last, because a judge that reasons aloud quotes the tag format before
    using it. A missing rating is neutral, which makes the pair a tie and so
    contributes only the floor weight.
    """
    a = RATING_A.findall(text) or LOOSE_A.findall(text)
    b = RATING_B.findall(text) or LOOSE_B.findall(text)
    return (max(1, min(10, int(a[-1]))) if a else NEUTRAL,
            max(1, min(10, int(b[-1]))) if b else NEUTRAL)


class Tally:
    """Who beat whom, and by how much, over the pairs judged so far."""

    def __init__(self, n: int):
        self.n = n
        self.wins = [0.0] * n
        self.weight = [0.0] * n
        self.rating_sum = [0] * n
        self.rating_count = [0] * n
        self.met = [[0] * n for _ in range(n)]

    def add(self, i: int, j: int, ra: int, rb: int) -> None:
        w = max(abs(ra - rb) / 9.0, TAU)      # a confident verdict counts for more
        a_won = 0.5 if ra == rb else (1.0 if ra > rb else 0.0)
        self.wins[i] += w * a_won
        self.wins[j] += w * (1.0 - a_won)
        self.weight[i] += w
        self.weight[j] += w
        self.rating_sum[i] += ra
        self.rating_count[i] += 1
        self.rating_sum[j] += rb
        self.rating_count[j] += 1
        self.met[i][j] += 1
        self.met[j][i] += 1

    def mu(self, i: int) -> float:
        return self.wins[i] / self.weight[i] if self.weight[i] > 0 else 0.5

    def avg_rating(self, i: int) -> float:
        return self.rating_sum[i] / self.rating_count[i] if self.rating_count[i] else 0.0

    def degree(self, i: int) -> int:
        return sum(1 for j in range(self.n) if self.met[i][j])

    def scores(self) -> list[float]:
        return [self.mu(i) for i in range(self.n)]


def replay(calls, n: int) -> Tally:
    """The tally implied by the judge calls already recorded for this problem."""
    tally = Tally(n)
    for call in calls:
        i, j = call["candidate_ids"]
        ra, rb = parse_pair(call["response"])
        tally.add(i, j, ra, rb)
    return tally


def connectivity_pairs(n: int, rng) -> list[tuple[int, int]]:
    """A random disjoint pairing, so every candidate is compared at least once."""
    order = list(range(n))
    rng.shuffle(order)
    return [(order[k], order[k + 1]) for k in range(0, n - 1, 2)]


def coverage_pairs(tally: Tally) -> list[tuple[int, int]]:
    """Lift candidates under MIN_DEGREE, against the closest mu they have not met."""
    degree = [tally.degree(i) for i in range(tally.n)]
    needy = sorted((i for i in range(tally.n) if degree[i] < MIN_DEGREE),
                   key=lambda i: degree[i])
    taken, pairs = set(), []
    for i in needy:
        if i in taken:
            continue
        options = [j for j in range(tally.n)
                   if j != i and j not in taken and tally.met[i][j] == 0]
        if not options:
            continue
        j = min(options, key=lambda j: (abs(tally.mu(i) - tally.mu(j)), degree[j]))
        pairs.append((i, j))
        taken.update((i, j))
    return pairs


def swiss_pairs(tally: Tally) -> list[tuple[int, int]]:
    """Pair each candidate with a near neighbour in the current ranking.

    Unseen opponents first, then the smallest mu gap, then the fewest prior
    meetings -- so the budget goes on the comparisons still in doubt.
    """
    rank = sorted(range(tally.n), key=lambda i: (-tally.mu(i), -tally.avg_rating(i)))
    taken, pairs = set(), []
    for k, i in enumerate(rank):
        if i in taken:
            continue
        best = None
        for t in range(k + 1, min(k + 1 + WINDOW, tally.n)):
            j = rank[t]
            if j in taken:
                continue
            key = (0 if tally.met[i][j] == 0 else 1,
                   abs(tally.mu(i) - tally.mu(j)), tally.met[i][j])
            if best is None or key < best[0]:
                best = (key, j)
        if best:
            pairs.append((i, best[1]))
            taken.update((i, best[1]))
    return pairs


class V1:
    def __init__(self, task: str, model_path: str | None = None):
        self.template = load(f"v1_{task}")
        self.model_path = model_path

    def prompt(self, problem: str, a: str, b: str) -> str:
        return (self.template.replace("{problem}", problem)
                .replace("{solution_a}", a)
                .replace("{solution_b}", b))

    def run_round(self, problem: str, judge_inputs: list[str], scores_so_far, llm, rng,
                  calls_so_far=None):
        # Two candidates share the response budget of a pairwise call.
        candidates = clip(judge_inputs, tokenizer_for(self.model_path), 2)
        n = len(candidates)
        calls_so_far = calls_so_far or []
        tally = replay(calls_so_far, n)

        if not calls_so_far:
            pairs = connectivity_pairs(n, rng)
        elif n > 2:
            pairs = coverage_pairs(tally) or swiss_pairs(tally)
        else:
            pairs = []
        if not pairs:
            return tally.scores()

        def judge(pair):
            i, j = pair
            call = llm.chat(
                [{"role": "user", "content": self.prompt(problem, candidates[i], candidates[j])}],
                max_tokens=JUDGE_OUTPUT_TOKENS, candidate_ids=[i, j])
            return pair, parse_pair(call.texts[0])

        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(pairs))) as pool:
            for (i, j), (ra, rb) in pool.map(judge, pairs):
                tally.add(i, j, ra, rb)
        return tally.scores()


def build(task: str, model_path: str | None = None, **params) -> V1:
    return V1(task, model_path)
