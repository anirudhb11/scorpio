"""Stage 2 entry point: score one shard's candidates.

    python -m scorpio.cli.score scorer=pointwise num_rounds=4 rep=0 shard=3/4

Output is per rep, so reps never read each other's rounds. Rounds are
appended as they finish: a job killed part-way keeps what it completed, and a
re-run continues from the next round.
"""

from __future__ import annotations

import collections
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scorpio.core.llm import LLM
from scorpio.core.runner import run_problem
from scorpio.core.server import serve
from scorpio.datasets import get as get_dataset
from scorpio.scorers import get as get_scorer
from scorpio.scorers import scorer_tag

logger = logging.getLogger(__name__)


def rounds_on_disk(path: Path) -> dict[str, list[dict]]:
    """Round records already written for this rep, per problem, in round order.

    Whole records, not just their scores: a scorer may need the judge calls it
    already made to decide its next move.
    """
    if not path.exists():
        return {}
    by_problem = collections.defaultdict(dict)
    for line in open(path):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break  # a write cut short: resume from the last whole round
        by_problem[record["problem_id"]][record["round"]] = record
    # Only an unbroken run from round 0 counts; a gap means redo from there.
    done = {}
    for problem_id, rounds in by_problem.items():
        records = []
        while len(records) in rounds:
            records.append(rounds[len(records)])
        done[problem_id] = records
    return done


@hydra.main(version_base=None, config_path="../../configs", config_name="score")
def main(cfg: DictConfig) -> None:
    index = int(str(cfg.shard).split("/")[0])
    out_dir = Path(cfg.out_dir)
    processor = cfg.dataset.get("post_processor") or cfg.model.post_processor

    src = out_dir / f"postprocessed_{processor}_{index:03d}.jsonl"
    params = OmegaConf.to_container(cfg.scorer_params)
    tag = scorer_tag(cfg.scorer, params)
    dst = out_dir / f"scores_{tag}_rep{cfg.rep}_{index:03d}.jsonl"
    judge_inputs = {r["problem_id"]: r["judge_inputs"]
                    for r in (json.loads(l) for l in open(src))}

    # The problem statement lives in the dataset, not in the stage-1 record.
    dataset = get_dataset(cfg.dataset.task)
    problem_text = {p.problem_id: p.messages[-1]["content"]
                    for p in dataset.load(cfg.dataset.path, cfg.dataset.name)}

    done = rounds_on_disk(dst)
    resumed = sum(1 for v in done.values() if v)
    if resumed:
        logger.info("resuming: %d problems already have rounds on disk", resumed)

    scorer = get_scorer(cfg.scorer).build(
        cfg.dataset.task, model_path=cfg.model.path, **params)
    sampling = OmegaConf.to_container(cfg.model.sampling)
    per_dataset = cfg.model.get("per_dataset")
    if per_dataset is not None:
        sampling.update(OmegaConf.to_container(per_dataset).get(cfg.dataset.name, {}))

    logger.info("%s shard %s: %d problems, scorer=%s rounds=%d rep=%d",
                cfg.dataset.name, cfg.shard, len(judge_inputs),
                cfg.scorer, cfg.num_rounds, cfg.rep)

    with serve(cfg.model.path, list(cfg.model.serve_args)) as base_url:
        llm = LLM(base_url=base_url, model=cfg.model.path, sampling=sampling)
        with open(dst, "a") as f:
            for problem_id, candidates in judge_inputs.items():
                for record in run_problem(
                        scorer, problem_text[problem_id], candidates, llm,
                        cfg.num_rounds, cfg.experiment, cfg.scorer, problem_id,
                        cfg.rep, done.get(problem_id)):
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    logger.info("%s round %d: %d calls, mean score %.2f",
                                problem_id, record["round"], len(record["calls"]),
                                sum(record["scores"]) / len(record["scores"]))

    logger.info("wrote %s", dst)


if __name__ == "__main__":
    main()
