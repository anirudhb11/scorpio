"""Fit the candidates of one judge call inside its share of the context."""

from __future__ import annotations

from functools import lru_cache

# Candidate tokens per judge call, split evenly across the candidates in it.
JUDGE_RESPONSE_BUDGET = 80000


@lru_cache(maxsize=4)
def tokenizer_for(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def clip(candidates: list[str], tokenizer, candidates_per_call: int,
         total_budget: int = JUDGE_RESPONSE_BUDGET) -> list[str]:
    """Keep the tail: the last budget//candidates_per_call tokens of each."""
    per = max(1, total_budget // max(1, candidates_per_call))
    out = []
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        out.append(tokenizer.decode(ids[-per:]) if len(ids) > per else candidate)
    return out
