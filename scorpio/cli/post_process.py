"""Build the judge inputs for one shard.

    python -m scorpio.cli.post_process shard=3/4

The processor comes from the model, overridden by the dataset. Output is a
sibling of the prepared file named after the processor. The prepared file is
never modified.
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
from scorpio.prepare.post_process import PROCESSORS, summarize

logger = logging.getLogger(__name__)


def write(f, record, processor, judge_inputs, tokens_in=0, tokens_out=0):
    f.write(json.dumps({
        "problem_id": record["problem_id"],
        "post_processor": processor,
        "judge_inputs": judge_inputs,
        "tokens": {"summarizer_input": tokens_in, "summarizer_output": tokens_out},
    }) + "\n")
    f.flush()


@hydra.main(version_base=None, config_path="../../configs", config_name="prepare")
def main(cfg: DictConfig) -> None:
    processor = cfg.dataset.get("post_processor") or cfg.model.post_processor
    index = int(str(cfg.shard).split("/")[0])

    out_dir = Path(cfg.out_dir)
    records = [json.loads(line) for line in open(out_dir / f"prepared_{index:03d}.jsonl")]
    dst = out_dir / f"postprocessed_{processor}_{index:03d}.jsonl"
    logger.info("%s shard %s: %d problems, post_processor=%s",
                cfg.dataset.name, cfg.shard, len(records), processor)

    if processor in PROCESSORS:
        apply = PROCESSORS[processor]
        with open(dst, "w") as f:
            for record in records:
                write(f, record, processor, [apply(r) for r in record["responses"]])
        logger.info("wrote %s", dst)
        return

    dataset = get_dataset(cfg.dataset.task)
    problems = {p.problem_id: p for p in dataset.load(cfg.dataset.path, cfg.dataset.name)}
    sampling = OmegaConf.to_container(cfg.model.sampling)
    per_dataset = cfg.model.get("per_dataset")
    if per_dataset is not None:
        sampling.update(OmegaConf.to_container(per_dataset).get(cfg.dataset.name, {}))

    with serve(cfg.model.path, list(cfg.model.serve_args)) as base_url:
        llm = LLM(base_url=base_url, model=cfg.model.path, sampling=sampling)
        with open(dst, "w") as f:
            for record in records:
                problem = problems[record["problem_id"]]
                judge_inputs, tokens_in, tokens_out = summarize(
                    record["responses"], problem.messages[-1]["content"],
                    cfg.dataset.task, llm, cfg.summarizer_max_tokens)
                write(f, record, processor, judge_inputs, tokens_in, tokens_out)
                logger.info("%s summarized, %d output tokens", record["problem_id"], tokens_out)
    logger.info("wrote %s", dst)


if __name__ == "__main__":
    main()
