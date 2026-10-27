"""Score each candidate with the ThinkPRM generative verifier."""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from scorpio.core.clip import tokenizer_for
from scorpio.core.types import MAX_CONCURRENT_REQUESTS
from scorpio.prompts import load

PREDECISION = "Is the solution correct?"
YES, NO = " Yes", " No"


def strip_harmony(text: str) -> str:
    """Drop gpt-oss channel framing. The reasoning stays; the control tokens go."""
    if "<|channel|>" not in text and "<|start|>" not in text:
        return text
    text = re.sub(r"<\|channel\|>\w+<\|message\|>", "\n\n", text)
    text = re.sub(r"<\|start\|>\w+", "", text)
    text = re.sub(r"<\|[^|>]*\|>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class ThinkPRM:
    def __init__(self, task: str, critic, max_tokens: int = 32000,
                 temperature: float = 0.4, decision_temperature: float = 1.0):
        self.template = load(f"thinkprm_{task}")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.decision_temperature = decision_temperature
        self.root = critic.base_url.rsplit("/v1", 1)[0]
        self.tokenizer = tokenizer_for(critic.model)
        self.yes = self.tokenizer.encode(YES, add_special_tokens=False)[-1]
        self.no = self.tokenizer.encode(NO, add_special_tokens=False)[-1]

    def prompt(self, problem: str, candidate: str) -> str:
        user = (self.template.replace("{problem}", problem)
                .replace("{solution}", strip_harmony(candidate)))
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True)

    def generate(self, body: dict) -> dict:
        response = requests.post(f"{self.root}/generate", json=body, timeout=6000)
        response.raise_for_status()
        return response.json()

    def critique(self, chat: str) -> dict:
        return self.generate({
            "text": chat,
            "sampling_params": {"max_new_tokens": self.max_tokens,
                                "temperature": self.temperature,
                                "stop": [PREDECISION]},
        })

    def verdict(self, chat: str, critique: dict) -> dict:
        text = chat + critique["text"]
        if not text.endswith(PREDECISION):
            text = text + PREDECISION

        # Temperature 1 because the logprobs come back divided by it, and the
        # verdict is read rather than sampled.
        return self.generate({
            "text": text,
            "sampling_params": {"max_new_tokens": 1, "temperature": 1.0},
            "return_logprob": True,
            "token_ids_logprob": [self.yes, self.no],
        })

    def run_round(self, problem: str, judge_inputs: list[str], scores_so_far, llm, rng,
                  calls_so_far=None):
        # Every critique, then every verdict. A batch that mixed the two would
        # ask for one candidate's token logprobs and not another's.
        chats = [self.prompt(problem, candidate) for candidate in judge_inputs]
        workers = min(MAX_CONCURRENT_REQUESTS, len(chats))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            critiques = list(pool.map(self.critique, chats))
            verdicts = list(pool.map(self.verdict, chats, critiques))

        scores = []
        for i, (chat, critique, verdict) in enumerate(zip(chats, critiques, verdicts)):
            logprob = {token: value for value, token, _
                       in verdict["meta_info"]["output_token_ids_logprobs"][0]}
            yes = math.exp(logprob[self.yes] / self.decision_temperature)
            no = math.exp(logprob[self.no] / self.decision_temperature)
            scores.append(yes / (yes + no))
            llm.calls.append({
                "candidate_ids": [i],
                "prompt": chat,
                "response": critique["text"],
                # The verdict call re-reads the cached prompt and critique to
                # read one token's logprobs, so the prompt is counted once.
                "input_tokens": critique["meta_info"]["prompt_tokens"],
                "output_tokens": (critique["meta_info"]["completion_tokens"]
                                  + verdict["meta_info"]["completion_tokens"]),
            })
        return scores


def build(task: str, model_path: str | None = None, critic=None, **params) -> ThinkPRM:
    return ThinkPRM(task, critic, **params)
