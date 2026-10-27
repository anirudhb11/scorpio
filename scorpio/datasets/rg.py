"""Reasoning-gym puzzles, and how well an answer solves one."""

from __future__ import annotations

import json

import pyarrow.parquet as pq

from scorpio.core.types import Problem

ANSWER_OPEN, ANSWER_CLOSE, THINK_END = "<answer>", "</answer>", "</think>"


def extract_answer(response: str) -> str | None:
    """The text of the LAST <answer> block.

    Last, not first: a thinking model quotes the instruction "answer between
    <answer> and </answer>" back to itself mid-reasoning, so the first tag is
    usually the echo rather than the answer.
    """
    opened = response.rfind(ANSWER_OPEN)
    if opened >= 0:
        rest = response[opened + len(ANSWER_OPEN):]
        closed = rest.find(ANSWER_CLOSE)
        if closed >= 0:
            return rest[:closed].strip()

    parts = response.split(THINK_END, 1)
    if len(parts) == 1:
        return None
    tail = parts[1].strip().rsplit("Final Answer:", 1)
    return tail[1].strip() if len(tail) > 1 else None


def load(path: str, name: str) -> list[Problem]:
    """Read an RG parquet.

    Grading needs both the puzzle entry and the game it came from, so
    `ground_truth` carries the whole `extra_info` rather than an answer.
    """
    return [
        Problem(
            problem_id=f"{name}_{row['extra_info']['index']:04d}"
            if "index" in row["extra_info"] else f"{name}_{i:04d}",
            messages=[dict(m) for m in row["prompt"]],
            ground_truth={"entry": row["extra_info"]["entry"],
                          "dataset_name": row["extra_info"]["dataset_name"]},
        )
        for i, row in enumerate(pq.read_table(path).to_pylist())
    ]


def grade(response: str, ground_truth) -> float:
    """The game's own score in [0, 1]. Partial credit is game-specific."""
    from reasoning_gym.factory import get_score_answer_fn

    name = ground_truth["dataset_name"]
    entry = ground_truth["entry"]
    if isinstance(entry, str):
        entry = json.loads(entry)

    return float(get_score_answer_fn(name=name)(extract_answer(response) or "", entry))
