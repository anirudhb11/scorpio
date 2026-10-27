"""What a search cost, and what its best survivor was worth.

    python -m scorpio.cli.report_search model=gpt_oss_20b dataset=hmmt
"""

from __future__ import annotations

import glob
import json
import logging
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig

from scorpio.cli.search import critic_params
from scorpio.metrics import w_best
from scorpio.scorers import scorer_tag

logger = logging.getLogger(__name__)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def ci95(xs):
    """Half-width of the 95% interval on the mean, across reps."""
    if len(xs) < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


@hydra.main(version_base=None, config_path="../../configs", config_name="search")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.out_dir)
    price_in = cfg.model.prices.input / 1e6
    price_out = cfg.model.prices.output / 1e6
    rounds = cfg.search.num_rounds

    stage1, tokens = {}, Counter()
    for f in sorted(glob.glob(str(out_dir / "prepared_*.jsonl"))):
        for line in open(f):
            record = json.loads(line)
            stage1[record["problem_id"]] = record
            tokens.update(record["tokens"])
    if not stage1:
        logger.error("no prepared_*.jsonl in %s", out_dir)
        return

    tag = scorer_tag(cfg.search.critic, critic_params(cfg))
    by_rep = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(str(out_dir / f"scores_{tag}_rep*_*.jsonl"))):
        rep = int(re.search(r"_rep(\d+)_", f).group(1))
        for line in open(f):
            record = json.loads(line)
            by_rep[rep][record["problem_id"]][record["round"]] = record
    if not by_rep:
        logger.error("no score files for %s in %s", tag, out_dir)
        return

    # Whole reps only, so a rep still being scored cannot shrink the problem
    # set for every other one.
    full = sorted(rep for rep, problems in by_rep.items()
                  if all(all(i in problems.get(pid, {}) for i in range(rounds))
                         for pid in stage1))
    dropped = sorted(set(by_rep) - set(full))
    if dropped:
        logger.warning("skipping incomplete reps %s", dropped)
    if not full:
        logger.error("no rep has all %d rounds for all %d problems yet",
                     rounds, len(stage1))
        return

    best_per_rep, judge_in, judge_out = [], [], []
    for rep in full:
        best, rep_in, rep_out = [], 0, 0
        for pid, record in stage1.items():
            used = [by_rep[rep][pid][i] for i in range(rounds)]
            scores = [mean(col) for col in zip(*(r["scores"] for r in used))]
            best.append(w_best(record["rewards"], scores))
            for r in used:
                rep_in += sum(c["input_tokens"] for c in r["calls"])
                rep_out += sum(c["output_tokens"] for c in r["calls"])
        best_per_rep.append(mean(best))
        judge_in.append(rep_in)
        judge_out.append(rep_out)

    # One rep is one run of the pipeline, so its judging is costed once and
    # averaged over the reps that measure it.
    final_in, final_out = mean(judge_in), mean(judge_out)
    cost = {
        "generation": (tokens["generation_input"] * price_in
                       + tokens["generation_output"] * price_out),
        "search_judge": (tokens["judge_input"] * price_in
                         + tokens["judge_output"] * price_out),
        "final_judge": final_in * price_in + final_out * price_out,
    }
    cost["total"] = sum(cost.values())

    survivors = [len(r["responses"]) for r in stage1.values()]
    metrics = {
        "critic": cfg.search.critic,
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "width": cfg.num_responses,
        "depth": cfg.search.max_depth,
        "step_tokens": cfg.search.num_tokens_in_each_step,
        "prune_percentile": cfg.search.prune_percentile,
        "group_size": cfg.search.group_size,
        "num_rounds": rounds,
        "seed": cfg.search.seed,
        "problems": len(stage1),
        "reps": len(full),
        "survivors_mean": mean(survivors),
        "survivors_min": min(survivors),
        "survivors_max": max(survivors),
        "pass_at_1": mean([r["pass_at_1"] for r in stage1.values()]),
        "pass_at_n": mean([r["pass_at_n"] for r in stage1.values()]),
        "w_best": mean(best_per_rep),
        "w_best_ci95": ci95(best_per_rep),
        "prices_usd_per_mtok": {"input": cfg.model.prices.input,
                                "output": cfg.model.prices.output},
        "tokens": {
            "generation_input": tokens["generation_input"],
            "generation_output": tokens["generation_output"],
            "search_judge_input": tokens["judge_input"],
            "search_judge_output": tokens["judge_output"],
            "final_judge_input": final_in,
            "final_judge_output": final_out,
        },
        "cost_usd": cost,
    }
    path = out_dir / f"metrics_search_{tag}.json"
    json.dump(metrics, open(path, "w"), indent=2)

    logger.info("%s %s x %s  n=%d d=%d  %d problems, %d rep(s), %.1f survivors",
                metrics["critic"], metrics["model"], metrics["dataset"],
                metrics["width"], metrics["depth"], metrics["problems"],
                metrics["reps"], metrics["survivors_mean"])
    logger.info("cost $%.4f = gen $%.4f + prune $%.4f + final $%.4f",
                cost["total"], cost["generation"], cost["search_judge"],
                cost["final_judge"])
    logger.info("pass@1=%.4f pass@n=%.4f w_best=%.4f +/- %.4f | wrote %s",
                metrics["pass_at_1"], metrics["pass_at_n"],
                metrics["w_best"], metrics["w_best_ci95"], path)


if __name__ == "__main__":
    main()
