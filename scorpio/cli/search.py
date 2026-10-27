"""Stage 1 as a pruned tree search, then the post-processing and the scoring.

    python -m scorpio.cli.search shard=3/4
    python -m scorpio.cli.search dataset=lcb

One job, because all three stages want the same model up. The files it writes
have the same format as those of prepare, post_process and score.
"""

from __future__ import annotations

import collections
import json
import logging
from contextlib import nullcontext
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scorpio.cli.post_process import write as write_post
from scorpio.cli.prepare import take_shard
from scorpio.cli.score import rounds_on_disk
from scorpio.core.llm import LLM
from scorpio.core.runner import run_problem
from scorpio.core.server import serve
from scorpio.datasets import get as get_dataset
from scorpio.prepare.post_process import PROCESSORS
from scorpio.scorers import get as get_scorer
from scorpio.scorers import scorer_tag
from scorpio.scorers.search_partial import build as build_partial
from scorpio.search import render, solve

logger = logging.getLogger(__name__)


def levels_on_disk(path: Path) -> dict[str, list[dict]]:
    """Level records already written, per problem, in level order."""
    if not path.exists():
        return {}
    by_problem = collections.defaultdict(dict)
    for line in open(path):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break  # a write cut short: resume from the last whole level
        if record["level"] is not None:
            by_problem[record["problem_id"]][record["level"]] = record
    done = {}
    for problem_id, levels in by_problem.items():
        records = []
        while len(records) + 1 in levels:
            records.append(levels[len(records) + 1])
        done[problem_id] = records
    return done


def search_shard(llm, scorer, dataset, problems, cfg, out_dir, index) -> list[dict]:
    """Stage 1: grow and prune each problem's tree, then grade what survives."""
    s = cfg.search
    trace = out_dir / f"search_{index:03d}.jsonl"
    on_disk = levels_on_disk(trace)
    resumed = sum(1 for v in on_disk.values() if v)
    if resumed:
        logger.info("resuming: %d problems already have levels on disk", resumed)

    records = []
    with open(trace, "a") as tf:
        def keep(record):
            tf.write(json.dumps(record) + "\n")
            tf.flush()
            if record["level"] is not None:
                fates = collections.Counter(leaf["fate"] for leaf in record["leaves"])
                logger.info("%s level %d: %d leaves %s", record["problem_id"],
                            record["level"], len(record["leaves"]), dict(fates))

        for i, problem in enumerate(problems, 1):
            record = solve(
                llm, scorer, problem.messages[-1]["content"],
                render(cfg.model.path, problem.messages, llm.sampling),
                dataset.grade, problem.ground_truth,
                width=cfg.num_responses, depth=s.max_depth,
                step_tokens=s.num_tokens_in_each_step,
                solution_tokens=s.max_tokens_in_solution,
                percentile=s.prune_percentile, rounds=s.num_rounds,
                experiment=cfg.experiment, scorer_name="search_partial",
                problem_id=problem.problem_id, seed=s.seed,
                done=on_disk.get(problem.problem_id), on_level=keep)
            records.append(record)
            logger.info("[%d/%d] %s %d survivors pass@1=%.3f pass@n=%d",
                        i, len(problems), problem.problem_id,
                        len(record["responses"]), record["pass_at_1"],
                        record["pass_at_n"])

    dst = out_dir / f"prepared_{index:03d}.jsonl"
    with open(dst, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info("wrote %s", dst)
    return records


def post_process_shard(cfg, out_dir, index, records) -> dict:
    """Stage 2: turn the survivors into what the judge is shown."""
    processor = cfg.dataset.get("post_processor") or cfg.model.post_processor
    apply = PROCESSORS[processor]
    dst = out_dir / f"postprocessed_{processor}_{index:03d}.jsonl"
    judge_inputs = {}
    with open(dst, "w") as f:
        for record in records:
            inputs = [apply(r) for r in record["responses"]]
            write_post(f, record, processor, inputs)
            judge_inputs[record["problem_id"]] = inputs
    logger.info("wrote %s", dst)
    return judge_inputs


def score_shard(llm, scorer, problems, cfg, out_dir, index, judge_inputs) -> None:
    """Stage 3: score the survivors with the critic that pruned them."""
    s = cfg.search
    tag = scorer_tag(s.critic, critic_params(cfg))
    dst = out_dir / f"scores_{tag}_rep{cfg.rep}_{index:03d}.jsonl"
    done = rounds_on_disk(dst)
    problem_text = {p.problem_id: p.messages[-1]["content"] for p in problems}

    with open(dst, "a") as f:
        for problem_id, candidates in judge_inputs.items():
            for record in run_problem(
                    scorer, problem_text[problem_id], candidates, llm,
                    s.num_rounds, cfg.experiment, s.critic, problem_id, cfg.rep,
                    done.get(problem_id)):
                f.write(json.dumps(record) + "\n")
                f.flush()
                logger.info("%s round %d: %d calls, mean score %.2f",
                            problem_id, record["round"], len(record["calls"]),
                            sum(record["scores"]) / len(record["scores"]))
    logger.info("wrote %s", dst)


def critic_params(cfg) -> dict:
    """What distinguishes one run of this critic from another."""
    return {} if cfg.search.critic == "thinkprm" else {"group_size": cfg.search.group_size}


def with_mem_fraction(args: list[str], fraction: float) -> list[str]:
    """The same serve args, asking for a different share of the card."""
    args = list(args)
    if "--mem-fraction-static" in args:
        args[args.index("--mem-fraction-static") + 1] = str(fraction)
        return args
    return args + ["--mem-fraction-static", str(fraction)]


def make_scorers(cfg, critic):
    """The scorer that prunes, and the one that scores what survived.

    Scorpio has a prompt for each; ThinkPRM uses one for both.
    """
    task = cfg.dataset.task
    if cfg.search.critic == "thinkprm":
        scorer = get_scorer("thinkprm").build(
            task, critic=critic, max_tokens=cfg.critic.max_tokens,
            temperature=cfg.critic.temperature,
            decision_temperature=cfg.critic.decision_temperature)
        return scorer, scorer
    return (build_partial(task, group_size=cfg.search.group_size,
                          model_path=cfg.model.path),
            get_scorer("scorpio").build(task, model_path=cfg.model.path,
                                        group_size=cfg.search.group_size))


@hydra.main(version_base=None, config_path="../../configs", config_name="search")
def main(cfg: DictConfig) -> None:
    dataset = get_dataset(cfg.dataset.task)
    problems, index = take_shard(
        dataset.load(cfg.dataset.path, cfg.dataset.name), str(cfg.shard))
    s = cfg.search
    logger.info("%s shard %s: %d problems, width %d, depth %d x %d tokens, "
                "keep >= p%s, critic %s K=%d M=%d",
                cfg.dataset.name, cfg.shard, len(problems), cfg.num_responses,
                s.max_depth, s.num_tokens_in_each_step, s.prune_percentile,
                s.critic, s.group_size, s.num_rounds)
    if not problems:
        logger.info("nothing to do")
        return

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampling = OmegaConf.to_container(cfg.model.sampling)
    per_dataset = cfg.model.get("per_dataset")
    if per_dataset is not None:
        sampling.update(OmegaConf.to_container(per_dataset).get(cfg.dataset.name, {}))

    # A critic of its own shares the card with the policy, so both ask for less.
    separate = s.critic != "scorpio"
    policy_args = list(cfg.model.serve_args)
    critic_server = nullcontext()
    if separate:
        policy_args = with_mem_fraction(policy_args, cfg.critic.policy_mem_fraction)
        critic_server = serve(cfg.critic.path, list(cfg.critic.serve_args))

    with serve(cfg.model.path, policy_args) as base_url, critic_server as critic_url:
        llm = LLM(base_url=base_url, model=cfg.model.path, sampling=sampling)
        critic = (LLM(base_url=critic_url, model=cfg.critic.path, sampling={})
                  if critic_url else None)

        pruner, final = make_scorers(cfg, critic)
        records = search_shard(llm, pruner, dataset, problems, cfg, out_dir, index)
        judge_inputs = post_process_shard(cfg, out_dir, index, records)
        score_shard(llm, final, problems, cfg, out_dir, index, judge_inputs)


if __name__ == "__main__":
    main()
