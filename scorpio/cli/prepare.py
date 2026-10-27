"""Stage 1 entry point: generate and grade, one record per problem.

    python -m scorpio.cli.prepare
    python -m scorpio.cli.prepare dataset=lcb num_responses=8
    python -m scorpio.cli.prepare shard=3/4
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scorpio.core.llm import LLM
from scorpio.core.server import serve
from scorpio.datasets import get as get_dataset
from scorpio.prepare.generate import prepare_problem

logger = logging.getLogger(__name__)


def take_shard(problems: list, shard: str) -> tuple[list, int]:
    """The `i`th of `n` contiguous blocks, zero-based.

    Sharding only decides who does what, never what the answer is.
    """
    index, count = (int(x) for x in shard.split("/"))
    if not 0 <= index < count:
        raise ValueError(f"shard {shard} is out of range")
    start = len(problems) * index // count
    stop = len(problems) * (index + 1) // count
    return problems[start:stop], index


@hydra.main(version_base=None, config_path="../../configs", config_name="prepare")
def main(cfg: DictConfig) -> None:
    dataset = get_dataset(cfg.dataset.task)
    problems = dataset.load(cfg.dataset.path, cfg.dataset.name)
    problems, index = take_shard(problems, str(cfg.shard))
    logger.info("%s shard %s: %d problems, %d responses each",
                cfg.dataset.name, cfg.shard, len(problems), cfg.num_responses)
    if not problems:
        logger.info("nothing to do")
        return

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prepared_{index:03d}.jsonl"

    # A model may sample differently on some datasets; those values win.
    sampling = OmegaConf.to_container(cfg.model.sampling)
    per_dataset = cfg.model.get("per_dataset")
    if per_dataset is not None:
        sampling.update(OmegaConf.to_container(per_dataset).get(cfg.dataset.name, {}))
    logger.info("sampling: %s", sampling)

    with serve(cfg.model.path, list(cfg.model.serve_args)) as base_url:
        llm = LLM(base_url=base_url, model=cfg.model.path, sampling=sampling)
        with open(out_path, "w") as f:
            for i, problem in enumerate(problems, 1):
                record = prepare_problem(llm, problem, dataset.grade,
                                         cfg.num_responses, cfg.model.max_tokens)
                f.write(json.dumps(record) + "\n")
                f.flush()
                logger.info("[%d/%d] %s pass@1=%.3f pass@n=%d out_tokens=%d",
                            i, len(problems), record["problem_id"],
                            record["pass_at_1"], record["pass_at_n"],
                            record["tokens"]["generation_output"])

    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
