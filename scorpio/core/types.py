"""Shared types."""

from __future__ import annotations

from dataclasses import dataclass

# Requests in flight per problem.
MAX_CONCURRENT_REQUESTS = 16

# Output budget for any judge call, whatever the scorer.
JUDGE_OUTPUT_TOKENS = 32000


@dataclass
class Problem:
    """One problem to solve.

    `messages` is the chat-formatted prompt as the dataset ships it.
    `ground_truth` is whatever that task grades against: an answer (or several
    that all count) for math, the problem's test cases for code. It is a
    grading input only -- it is never written to a record.
    """

    problem_id: str
    messages: list[dict]
    ground_truth: str | list[str]
