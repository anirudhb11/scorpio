"""Turn raw responses into the text a judge is shown."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from scorpio.core.types import MAX_CONCURRENT_REQUESTS
from scorpio.prompts import load

NOT_VALID = "The response is not a valid solution."


def final_channel(response: str) -> str:
    """Everything after the reasoning. A no-op when the model reasons inline."""
    s = response.split("</think>")[-1].strip() if "</think>" in response else response.strip()
    return s.split("</thinking>")[-1].strip() if "</thinking>" in s else s.strip()


def code_block(response: str) -> str:
    """The candidate's python block, re-fenced, or a stand-in when it has none."""
    for tag in ("</thinking>", "</think>", "<thinking>", "<think>"):
        if tag in response:
            response = response.split(tag)[-1].strip()
            break
    else:
        response = response.strip()

    if "```python" not in response:
        return NOT_VALID

    match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        return "```python\n" + match.group(1).strip() + "\n```"
    return "```python\n" + response.split("```python", 1)[1].strip()


PROCESSORS = {"final_channel": final_channel, "code_block": code_block}


def summarize(responses: list[str], problem_text: str, task: str, llm, max_tokens: int):
    """Rewrite each response concisely with the model itself, and report the
    summariser's input and output tokens."""
    template = load(f"summarize_{task}")

    def one(response: str):
        prompt = template.replace("{problem}", problem_text).replace("{candidate}", response)
        call = llm.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
        # The summariser's own reasoning is not part of the summary.
        return final_channel(call.texts[0]), call.input_tokens, call.output_tokens

    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(responses))) as pool:
        results = list(pool.map(one, responses))

    return ([r[0] for r in results],
            sum(r[1] for r in results),
            sum(r[2] for r in results))
