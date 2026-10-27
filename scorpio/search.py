"""Grow a tree of partial solutions, dropping the branches a judge rates low.

Per problem, per level: continue every surviving attempt by one step, set the
finished ones aside, score the rest, and prune those below the `percentile`-th
percentile of this problem's own scores. Survivors of the last level finish in
whatever token budget is left.
"""

from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import requests

from scorpio.core.clip import tokenizer_for
from scorpio.core.llm import Call
from scorpio.core.runner import RecordingLLM, seed_for
from scorpio.core.types import MAX_CONCURRENT_REQUESTS

FINAL = "<|channel|>final<|message|>"
ANALYSIS = "<|channel|>analysis<|message|>"
TURN = "<|end|><|start|>assistant"


def render(model_path: str, messages: list[dict], sampling: dict) -> str:
    """The prompt as raw text. /generate does no templating of its own."""
    kwargs = {k: sampling[k] for k in ("enable_thinking", "reasoning_effort") if k in sampling}
    return tokenizer_for(model_path).apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs)


def normalise(text: str) -> str:
    """Raw output in the shape `LLM.chat` returns, which is what the post
    processors and the judge read. Inline reasoning passes through."""
    if FINAL not in text:
        return text
    head, _, content = text.partition(FINAL)
    return f"<think>{head.replace(ANALYSIS, '').replace(TURN, '')}</think>\n{content}"


def complete(llm, text: str, max_tokens: int, n: int = 1) -> Call:
    """Continue `text`, which the chat API cannot do. Sampling is read off
    `llm`, so it cannot drift from chat."""
    s = llm.sampling
    # skip_special_tokens defaults True, which would drop the channel markers.
    params = {"max_new_tokens": max_tokens, "temperature": s["temperature"],
              "top_p": s["top_p"], "n": n, "skip_special_tokens": False}
    params.update({k: s[k] for k in ("top_k", "presence_penalty") if k in s})
    root = llm.base_url.rsplit("/v1", 1)[0]
    result = requests.post(f"{root}/generate", timeout=6000,
                           json={"text": text, "sampling_params": params}).json()
    outputs = result if isinstance(result, list) else [result]
    meta = [out["meta_info"] for out in outputs]
    return Call(texts=[out["text"] for out in outputs],
                finish_reasons=[m["finish_reason"]["type"] for m in meta],
                input_tokens=meta[0]["prompt_tokens"],  # one shared prefill
                output_tokens=sum(m["completion_tokens"] for m in meta))


@dataclass
class Leaf:
    text: str
    done: bool
    level: int


def expand(llm, prefix: str, leaves: list[Leaf], n: int, max_tokens: int):
    """Continue each leaf by one step, `n` continuations each."""
    def step(leaf):
        return leaf, complete(llm, prefix + leaf.text, max_tokens, n)

    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(leaves))) as pool:
        results = list(pool.map(step, leaves))

    grown, tokens_in, tokens_out = [], 0, 0
    for leaf, call in results:
        tokens_in += call.input_tokens
        tokens_out += call.output_tokens
        for text, reason in zip(call.texts, call.finish_reasons):
            grown.append(Leaf(leaf.text + text, reason == "stop", leaf.level + 1))
    return grown, tokens_in, tokens_out


def keep(scores: list[float], percentile: float) -> list[int]:
    """Indices scoring at or above the percentile, best first. Ties break on
    the lower index, so the order a resumed run sees is the one it would have."""
    threshold = float(np.percentile(scores, percentile))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], -i), reverse=True)
    return [i for i in order if scores[i] >= threshold]


def score(scorer, problem: str, leaves: list[Leaf], llm, rounds: int, level: int,
          step_tokens: int, solution_tokens: int, experiment: str, scorer_name: str,
          problem_id: str, rep: int):
    """Mean score per leaf over `rounds` independent groupings."""
    scorer.tokens_used = level * step_tokens
    scorer.remaining = max(0, solution_tokens - scorer.tokens_used)
    recorder = RecordingLLM(llm)
    so_far: list[list[float]] = []
    for rnd in range(rounds):
        rng = random.Random(seed_for(experiment, scorer_name, f"{problem_id}|L{level}", rep, rnd))
        so_far.append(scorer.run_round(problem, [leaf.text for leaf in leaves], so_far,
                                       recorder, rng, recorder.calls))
    scores = [sum(col) / len(col) for col in zip(*so_far)]
    return scores, recorder.calls


def run(llm, scorer, problem: str, prefix: str, *, width: int, depth: int,
        step_tokens: int, solution_tokens: int, percentile: float, rounds: int,
        experiment: str, scorer_name: str, problem_id: str, rep: int,
        done: list[dict] | None = None):
    """Yield one record per level, so a level survives the job that follows it.

    `done` is the level records already on disk; those levels are skipped and
    the search resumes from the survivors the last one recorded.
    """
    done = list(done or [])
    finished = [Leaf(leaf["text"], True, record["level"])
                for record in done for leaf in record["leaves"] if leaf["fate"] == "done"]
    if done:
        active = [Leaf(leaf["text"], False, done[-1]["level"])
                  for leaf in done[-1]["leaves"] if leaf["fate"] == "kept"]
    else:
        active = [Leaf("", False, 0)]

    for level in range(len(done) + 1, depth + 1):
        if not active:
            break
        last = level == depth
        grown, tokens_in, tokens_out = expand(
            llm, prefix, active, width if level == 1 else 1, step_tokens)

        running = [leaf for leaf in grown if not leaf.done]
        finished += [leaf for leaf in grown if leaf.done]

        # The last level does not prune: its survivors go to the final
        # completion pass, and the final scorer re-scores every leaf anyway.
        scores, calls, kept = [], [], list(range(len(running)))
        if running and not last:
            scores, calls = score(scorer, problem, running, llm, rounds, level,
                                  step_tokens, solution_tokens, experiment,
                                  scorer_name, problem_id, rep)
            kept = keep(scores, percentile)

        fates = {id(leaf): "pruned" for leaf in running}
        for i in kept:
            fates[id(running[i])] = "kept"
        by_index = {id(leaf): i for i, leaf in enumerate(running)}

        yield {
            "problem_id": problem_id,
            "level": level,
            "leaves": [{"text": leaf.text,
                        "fate": "done" if leaf.done else fates[id(leaf)],
                        "score": None if leaf.done or not scores
                                 else scores[by_index[id(leaf)]]}
                       for leaf in grown],
            "calls": calls,
            "tokens": {"generation_input": tokens_in, "generation_output": tokens_out,
                       "judge_input": sum(c["input_tokens"] for c in calls),
                       "judge_output": sum(c["output_tokens"] for c in calls)},
        }
        active = [running[i] for i in kept]

    yield {"problem_id": problem_id, "level": None,
           "finished": [leaf.text for leaf in finished],
           "active": [leaf.text for leaf in active]}


def finish(llm, prefix: str, texts: list[str], max_tokens: int):
    """Let the survivors of the last level run to the end of their budget."""
    if not texts or max_tokens <= 0:
        return texts, 0, 0
    leaves, tokens_in, tokens_out = expand(
        llm, prefix, [Leaf(t, False, 0) for t in texts], 1, max_tokens)
    return [leaf.text for leaf in leaves], tokens_in, tokens_out


def solve(llm, scorer, problem: str, prefix: str, grade, ground_truth, *,
          width: int, depth: int, step_tokens: int, solution_tokens: int,
          percentile: float, rounds: int, experiment: str, scorer_name: str,
          problem_id: str, seed: int, done: list[dict] | None = None,
          on_level=None) -> dict:
    """Grow one problem's tree, finish the survivors, and grade them.

    `on_level` sees each level record as it is produced, which is how the
    caller persists it; levels passed back in as `done` are not redone, and
    their token counts carry into the total.
    """
    tokens = Counter()
    for record in done or []:
        tokens.update(record["tokens"])

    tail = None
    for record in run(llm, scorer, problem, prefix, width=width, depth=depth,
                      step_tokens=step_tokens, solution_tokens=solution_tokens,
                      percentile=percentile, rounds=rounds, experiment=experiment,
                      scorer_name=scorer_name, problem_id=problem_id, rep=seed,
                      done=done):
        if on_level:
            on_level(record)
        if record["level"] is None:
            tail = record
            break
        tokens.update(record["tokens"])

    texts, tokens_in, tokens_out = finish(
        llm, prefix, tail["active"], solution_tokens - depth * step_tokens)
    tokens.update(generation_input=tokens_in, generation_output=tokens_out)

    responses = [normalise(t) for t in tail["finished"] + texts]
    rewards = [grade(r, ground_truth) for r in responses]
    return {
        "problem_id": problem_id,
        "responses": responses,
        "rewards": rewards,
        "pass_at_1": sum(rewards) / len(rewards),
        "pass_at_n": max(rewards),
        "tokens": dict(tokens),
    }
