"""Math problems, and whether an answer is right."""

from __future__ import annotations

import pyarrow.parquet as pq
from math_verify import parse, verify

from scorpio.core.types import Problem

THINK_END = "</think>"


def last_boxed_string(text: str) -> str | None:
    """The last \\boxed{...} in the text, braces balanced."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    depth = 0
    for i in range(idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[idx : i + 1]
    return None


def extract_boxed_answer(text: str) -> str | None:
    """What sits inside the last \\boxed{}, or None."""
    boxed = last_boxed_string(text)
    if boxed is None:
        return None
    left = "\\boxed{"
    if not boxed.startswith(left) or not boxed.endswith("}"):
        return None
    return boxed[len(left) : -1]


def is_equiv(answer: str, truth: str) -> bool:
    """Mathematical equivalence, not string equality: 0.5 matches \\frac{1}{2}."""
    if "$" not in answer:
        answer = f"${answer}$"
    if "$" not in truth:
        truth = f"${truth}$"
    return bool(verify(parse(truth), parse(answer)))


def load(path: str, name: str) -> list[Problem]:
    """Read a math parquet. The prompt is already chat-formatted in the file."""
    return [
        Problem(
            problem_id=f"{name}_{row['extra_info']['index']:04d}",
            messages=[dict(m) for m in row["prompt"]],
            ground_truth=row["reward_model"]["ground_truth"],
        )
        for row in pq.read_table(path).to_pylist()
    ]


def grade(response: str, ground_truth) -> float:
    """1.0 if the response's boxed answer matches a ground truth, else 0.0.

    The reasoning is dropped first, so a \\boxed{} the model wrote while
    thinking cannot be mistaken for its answer. A response with no \\boxed{}
    scores 0 even when it states the answer in prose.
    """
    if not response:
        return 0.0

    solution = response.split(THINK_END)[1] if THINK_END in response else response
    answer = extract_boxed_answer(solution)
    if answer is None:
        return 0.0

    truths = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    truths = [extract_boxed_answer(str(t)) if "\\boxed" in str(t) else str(t) for t in truths]

    return 1.0 if any(t is not None and is_equiv(answer, t) for t in truths) else 0.0
