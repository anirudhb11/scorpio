"""LiveCodeBench problems, and whether a solution passes its tests.

`extract_code_from_model`, `postprocess_lcb_sample` and `lcb_check_correctness`
are copied from https://github.com/HarmanDotpy/pairwise-self-verification/blob/main/rewards/code_reward.py.

Not a sandbox: the harness runs generated code under signal timeouts and
resource limits, not process isolation. Unix only.
"""

from __future__ import annotations

import json
import multiprocessing
import re

import pyarrow.parquet as pq

from scorpio.core.types import Problem
from scorpio.datasets.livecodebench import run_test as lcb_run_test


def extract_code_from_model(model_response: str):
    """The last Markdown code block, or None."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def postprocess_lcb_sample(sample):
    sample_inputs = [s["input"] for s in sample]
    sample_outputs = [s["output"] for s in sample]

    sample_dict = {"inputs": sample_inputs, "outputs": sample_outputs}

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, (
            f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}")
        sample_dict["fn_name"] = fn_name

    return {"input_output": json.dumps(sample_dict)}


def lcb_check_correctness(sample, generation, timeout=6, debug=False) -> bool:
    """Run one candidate against a problem's tests, under a global timeout.

    The outer timeout catches cases the per-test timeouts inside `run_test`
    do not, which is why the child is killed rather than only joined.
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)

    num_inputs = len(json.loads(sample["input_output"])["inputs"])
    global_timeout = (timeout + 1) * num_inputs + 5

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)

    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=global_timeout)

    if p.is_alive():
        p.kill()
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)

    if not result:
        return False
    return all(x == True for x in result[0])  # noqa: E712


def load(path: str, name: str) -> list[Problem]:
    """Read an LCB parquet. Ground truth is the problem's test cases."""
    return [
        Problem(
            problem_id=f"{name}_{row['extra_info']['index']:04d}",
            messages=[dict(m) for m in row["prompt"]],
            ground_truth=row["reward_model"]["ground_truth"],
        )
        for row in pq.read_table(path).to_pylist()
    ]


def grade(response: str, ground_truth) -> float:
    """1.0 only if the extracted program passes every test."""
    code = extract_code_from_model(response)
    if code is None:
        return 0.0
    tests = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    return 1.0 if lcb_check_correctness(tests, code) else 0.0
