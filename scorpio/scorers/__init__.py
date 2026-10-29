"""Scorers, one module per method.

Each exposes `build(task, **params)` returning an object with

    run_round(problem, judge_inputs, scores_so_far, llm, rng, calls_so_far) -> list[float]

one score per candidate. A new method is a new module here, selected with
`scorer=<module name>`; the runner owns rounds, seeding, transcripts and
token accounting.
"""

from __future__ import annotations

import importlib


SHORT = {"group_size": "k"}


def scorer_tag(scorer: str, params: dict) -> str:
    """Scorer plus the params that change its output, e.g. scorpio_k4."""
    used = params if scorer not in ("pointwise", "v1") else {}
    return scorer + "".join(f"_{SHORT.get(k, k)}{v}" for k, v in sorted(used.items()))


def get(name: str):
    """The scorer module `scorpio/scorers/<name>.py`."""
    return importlib.import_module(f"scorpio.scorers.{name}")
