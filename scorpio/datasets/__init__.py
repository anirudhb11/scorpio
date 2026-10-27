"""Datasets, one module per task.

Each exposes the same pair: `load(path, name) -> list[Problem]` and
`grade(response, ground_truth) -> float`. A new task is a new module here
plus one entry below.
"""

from __future__ import annotations


def get(task: str):
    from scorpio.datasets import code, math, rg

    return {"math": math, "code": code, "rg": rg}[task]
