# CLAUDE.md

Guide for coding agents working in this repository.

## Pipeline

Each stage is a Hydra entry point under `scorpio/cli/`, run as
`python -m scorpio.cli.<stage> key=value ...`. Configs live in `configs/`
(`prepare.yaml`, `score.yaml`, `search.yaml`; model and dataset groups under
`configs/model/` and `configs/dataset/`). Every stage that needs a model starts
its own SGLang server (`scorpio/core/server.py`) and stops it on exit.

| stage | reads | writes (in `out_dir`) |
|---|---|---|
| `prepare` | dataset parquet | `prepared_<shard>.jsonl`: responses, finish reasons, rewards, pass@1/pass@n, generation tokens |
| `post_process` | `prepared_*` | `postprocessed_<processor>_<shard>.jsonl`: `judge_inputs` per problem, summarizer tokens |
| `score` | `postprocessed_*` | `scores_<tag>_rep<r>_<shard>.jsonl`: one line per (problem, round) with `scores` and every judge `call` |
| `report` | all of the above | `metrics_<tag>.json`: cost, w_best, bias², variance, ranking error per number of rounds |
| `plot` | `metrics_*.json` | bias², variance and ranking error against cost |
| `search` | dataset parquet | `search_<shard>.jsonl` (per-level tree), then the same `prepared` / `postprocessed` / `scores` files |
| `report_search` | search outputs | `metrics_search_<tag>.json` |

`<shard>` is `shard=i/n`: the i-th of n contiguous blocks of problems.
`<tag>` is the scorer plus its parameters, e.g. `scorpio_k4` (`scorers.scorer_tag`).

## Extending

- **Scorer**: add `scorpio/scorers/<name>.py` with `build(task, **params)` returning
  an object with `run_round(problem, judge_inputs, scores_so_far, llm, rng, calls_so_far) -> list[float]`,
  and select it with `scorer=<name>`; `scorers.get` imports it by module name. Call the model
  only through `llm.chat(..., candidate_ids=...)`; the runner records every call, seeds the
  round and handles resuming.
- **Prompt**: `scorpio/prompts/<scorer>_<task>.txt`, filled with `str.replace` on
  `{problem}`-style placeholders (not `str.format`: math prompts contain braces).
- **Dataset**: add `scorpio/datasets/<task>.py` with `load(path, name) -> list[Problem]`
  and `grade(response, ground_truth) -> float`, register it in `datasets.get`, and add
  `configs/dataset/<name>.yaml` (`name`, `task`, `path`, optional `post_processor`).
- **Model**: add `configs/model/<name>.yaml` (`name`, `path`, `serve_args`, `max_tokens`,
  `sampling`, optional `per_dataset`, `post_processor`, `prices` in USD per million tokens).
  A sampling key that is absent is not sent, so the server default applies.

## Invariants

- Round seeds come from `core.runner.seed_for(experiment, scorer, problem_id, rep, round)`.
  The shard is deliberately not part of the seed, so changing the sharding never
  changes a result.
- `score` and `search` append as they go and resume from what is on disk. Do not
  rewrite their output files in place.
- `report` refuses to report unless every rep has every round for every problem.
  Bias² and variance are unbiased estimates across reps, so they need at least two reps.
- Judge inputs are clipped to the last 80k tokens per call, split across the
  candidates in it (`core/clip.py`). Stored responses are never clipped.
- Cost charges the problem prompt once per problem, not once per candidate.
- Tree search supports gpt-oss models (`search.normalise` parses its channel format)
  and the math and code tasks (the only ones with `search_partial_*` prompts).

## Checking changes

There is no test suite. Before submitting a change:

```bash
uvx pyflakes scorpio
```

and run the affected stages end to end on a small slice, for example
`dataset=hmmt num_responses=4 shard=0/15 model.max_tokens=8192`.
