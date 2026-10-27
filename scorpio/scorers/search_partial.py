"""Rate in-progress attempts, so the search can prune the ones going nowhere.

Grouping, parsing and clipping are Scorpio's; only the prompt differs. The
judge rates the probability each attempt reaches a correct answer if continued,
so it is told how many tokens the attempt has spent and has left.
"""

from __future__ import annotations

from scorpio.prompts import load
from scorpio.scorers.scorpio import Scorpio


class SearchPartial(Scorpio):
    def __init__(self, task: str, group_size: int = 4, model_path: str | None = None):
        super().__init__(task, group_size, model_path)
        self.template = load(f"search_partial_{task}")
        self.tokens_used = 0
        self.remaining = 0

    def prompt(self, problem: str, candidates: list[str]) -> str:
        blocks = "\n\n".join(
            f"**Candidate {i + 1}**\n{c.strip() or '(empty — no work yet)'}"
            for i, c in enumerate(candidates))
        tags = "\n".join(f"<rating_{i + 1}>INTEGER_1_TO_10</rating_{i + 1}>"
                         for i in range(len(candidates)))
        return (self.template.replace("{k}", str(len(candidates)))
                .replace("{problem}", problem)
                .replace("{candidates}", blocks)
                .replace("{tokens_used}", str(self.tokens_used))
                .replace("{remaining}", str(self.remaining))
                .replace("{rating_lines}", tags)) + "\n"


def build(task: str, group_size: int = 4, model_path: str | None = None, **params):
    return SearchPartial(task, group_size, model_path)
