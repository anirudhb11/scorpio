"""Accuracy, ranking bias and variance against inference compute.

    python -m scorpio.cli.report scorer=pointwise

For each number of rounds: the cost per problem spent to get there, the w_best
it bought (averaged over reps, with a 95% CI), and the ranking error split
into bias^2 and variance across reps. Writes metrics_<scorer>.json.

Cost is policy input and output, the summariser when there is one, and
judging. The problem prompt is charged once per problem, not once per
candidate.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from scorpio.metrics import w_best_keyed
from scorpio.scorers import scorer_tag

logger = logging.getLogger(__name__)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def dense_rank(keys):
    """The ranking keys as positions, so a plain `>` reproduces their order and
    candidates tied in every component share a position. V1 ranks on
    (mu, avg_rating), so its key is a pair; the others rank on one score."""
    position = {key: i for i, key in enumerate(sorted(set(keys)))}
    return [position[key] for key in keys]


def bias_var(runs, rewards):
    """Ranking error of one problem, split into bias^2 and variance.

    For every pair where one candidate has a higher reward than the other, p is
    the fraction of reps in which the verifier ranks it strictly higher. The
    pair's error 1[ranked higher] - 1 has bias^2 (1-p)^2 and variance p(1-p);
    both are returned as unbiased estimates over the R reps, averaged over
    pairs, together with the ranking error 1 - p.
    """
    r = np.asarray(rewards, dtype=float)
    ii, jj = np.where(r[:, None] > r[None, :])
    if not len(ii):
        return None
    run = np.asarray(runs, dtype=float).T
    R = run.shape[1]
    p = (run[ii] > run[jj]).mean(axis=1)
    pq = p * (1 - p)
    return (float(((1 - p) ** 2 - pq / (R - 1)).mean()),
            float((pq * R / (R - 1)).mean()),
            float((1 - p).mean()))


def ci95(xs):
    """Half-width of the 95% interval on the mean, across reps."""
    if len(xs) < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


@hydra.main(version_base=None, config_path="../../configs", config_name="score")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.out_dir)
    price_in = cfg.model.prices.input / 1e6
    price_out = cfg.model.prices.output / 1e6

    stage1 = {}
    gen_in = gen_out = 0
    for f in sorted(glob.glob(str(out_dir / "prepared_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            stage1[r["problem_id"]] = r
            tokens = r["tokens"]
            gen_in += tokens["generation_input"] // len(r["responses"])
            gen_out += tokens["generation_output"]
    n_problems = len(stage1)

    processor = cfg.dataset.get("post_processor") or cfg.model.post_processor
    summ_in = summ_out = 0
    for f in sorted(glob.glob(str(out_dir / f"postprocessed_{processor}_*.jsonl"))):
        for line in open(f):
            t = json.loads(line)["tokens"]
            summ_in += t["summarizer_input"]
            summ_out += t["summarizer_output"]
    base_cost = ((gen_in + summ_in) * price_in + (gen_out + summ_out) * price_out) / n_problems

    tag = scorer_tag(cfg.scorer, OmegaConf.to_container(cfg.scorer_params))
    by_rep = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(str(out_dir / f"scores_{tag}_rep*_*.jsonl"))):
        rep = int(re.search(r"_rep(\d+)_", f).group(1))
        for line in open(f):
            r = json.loads(line)
            by_rep[rep][r["problem_id"]][r["round"]] = r
    if not by_rep:
        logger.error("no score files for %s in %s", tag, out_dir)
        return

    max_round = max(r for p in by_rep.values() for rounds in p.values() for r in rounds)
    if cfg.rounds:
        max_round = min(max_round, cfg.rounds - 1)

    # Every rep must cover every problem for every round, so all points are
    # measured on the same problem set.
    incomplete = sorted(rep for rep, problems in by_rep.items()
                        if not all(all(i in problems.get(pid, {}) for i in range(max_round + 1))
                                   for pid in stage1))
    if incomplete:
        logger.error("incomplete reps %s: each needs %d rounds for all %d problems",
                     incomplete, max_round + 1, n_problems)
        return

    # V1 scores are cumulative win rates (mu), so it ranks on the last round's mu,
    # breaking ties on the average rating each candidate received. The other
    # scorers rank on their score averaged over the rounds used.
    if cfg.scorer == "v1":
        from scorpio.scorers.v1 import replay

        def ranking_key(used, scores):
            tally = replay([c for r in used for c in r["calls"]], len(scores))
            return [(s, tally.avg_rating(i)) for i, s in enumerate(used[-1]["scores"])]
    else:
        def ranking_key(used, scores):
            return [(s,) for s in scores]

    pass_at_1 = mean([r["pass_at_1"] for r in stage1.values()])
    curve = [{
        "rounds": 0,
        "reps": 0,
        "problems": n_problems,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
        "judge_cost_usd": 0.0,
        "total_cost_usd": base_cost,
        "w_best": pass_at_1,
        "w_best_ci95": 0.0,
    }]
    for upto in range(max_round + 1):
        best_per_rep, in_per_rep, out_per_rep = [], [], []
        runs = defaultdict(list)
        for problems in by_rep.values():
            best, judge_in, judge_out = [], 0, 0
            for pid, rounds in problems.items():
                used = [rounds[i] for i in range(upto + 1)]
                scores = [mean(col) for col in zip(*(r["scores"] for r in used))]
                key = ranking_key(used, scores)
                best.append(w_best_keyed(stage1[pid]["rewards"], key))
                runs[pid].append(dense_rank(key))
                for r in used:
                    judge_in += sum(c["input_tokens"] for c in r["calls"])
                    judge_out += sum(c["output_tokens"] for c in r["calls"])
            best_per_rep.append(mean(best))
            in_per_rep.append(judge_in)
            out_per_rep.append(judge_out)

        bv = [x for x in (bias_var(runs[pid], stage1[pid]["rewards"]) for pid in runs) if x]
        judge_in = mean(in_per_rep) / n_problems
        judge_out = mean(out_per_rep) / n_problems
        judge_cost = judge_in * price_in + judge_out * price_out
        point = {
            "rounds": upto + 1,
            "reps": len(best_per_rep),
            "problems": n_problems,
            "judge_input_tokens": judge_in,
            "judge_output_tokens": judge_out,
            "judge_cost_usd": judge_cost,
            "total_cost_usd": base_cost + judge_cost,
            "w_best": mean(best_per_rep),
            "w_best_ci95": ci95(best_per_rep),
            "questions": len(bv),
            "bias2": mean([x[0] for x in bv]),
            "variance": mean([x[1] for x in bv]),
            "ranking_error": mean([x[2] for x in bv]),
        }
        curve.append(point)
        logger.info("rounds=%d reps=%d  total=$%.4f  w_best=%.4f +/- %.4f",
                    point["rounds"], point["reps"], point["total_cost_usd"],
                    point["w_best"], point["w_best_ci95"])

    metrics = {
        "scorer": tag,
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "problems": n_problems,
        "pass_at_1": pass_at_1,
        "pass_at_n": mean([r["pass_at_n"] for r in stage1.values()]),
        "prices_usd_per_mtok": {"input": cfg.model.prices.input,
                                "output": cfg.model.prices.output},
        "generation_input_tokens": gen_in,
        "generation_output_tokens": gen_out,
        "summarizer_input_tokens": summ_in,
        "summarizer_output_tokens": summ_out,
        "base_cost_usd": base_cost,
        "curve": curve,
    }
    path = out_dir / f"metrics_{tag}.json"
    json.dump(metrics, open(path, "w"), indent=2)
    logger.info("pass@1=%.4f pass@n=%.4f base=$%.4f | wrote %s",
                metrics["pass_at_1"], metrics["pass_at_n"], base_cost, path)


if __name__ == "__main__":
    main()
