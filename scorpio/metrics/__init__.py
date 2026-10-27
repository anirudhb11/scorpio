"""What a verifier's scores were worth, over the whole candidate set."""

from __future__ import annotations


def w_best(rewards: list[float], scores: list[float]) -> float:
    """Expected reward of the top-scored candidate, averaged over ties."""
    top = max(scores)
    tied = [r for r, s in zip(rewards, scores) if s == top]
    return sum(tied) / len(tied)


def w_best_keyed(rewards: list[float], keys: list[tuple]) -> float:
    """w_best over a tuple ranking key, compared left to right."""
    top = max(keys)
    tied = [r for r, k in zip(rewards, keys) if k == top]
    return sum(tied) / len(tied)
