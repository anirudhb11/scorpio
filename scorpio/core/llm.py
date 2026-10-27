"""Chat calls against an SGLang server."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import OpenAI

RETRIES = 3
RETRY_WAIT = 2.0



def _joined(choice) -> str:
    """The whole generation as one string.

    A model whose server splits reasoning onto its own channel gets it folded
    back in behind a `<think>` tag, which is where a model that reasons inline
    already puts it. Every response then has the same shape and one
    post-processor handles them all.
    """
    reasoning = getattr(choice.message, "reasoning_content", "") or ""
    if reasoning:
        return f"<think>{reasoning}</think>\n{choice.message.content}"
    return choice.message.content


@dataclass
class Call:
    """One request and what came back.

    `texts[i]` and `finish_reasons[i]` describe completion `i`. Token counts
    are the server's, for the request as a whole: one prefill shared by every
    completion, and `output_tokens` summed over all of them.
    """

    texts: list[str]
    finish_reasons: list[str]
    input_tokens: int
    output_tokens: int


@dataclass
class LLM:
    """A model served by SGLang, addressed over the OpenAI chat API.

    `sampling` is the model's config, passed through as given. A key that is
    absent is not sent and the server's default applies, so absence carries
    meaning and defaults do not belong here.
    """

    base_url: str
    model: str
    sampling: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key="EMPTY", base_url=self.base_url, timeout=6000)

    def chat(self, messages: list[dict], max_tokens: int, n: int = 1) -> Call:
        """Return `n` completions of one prompt, which is prefilled once."""
        s = self.sampling
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": s["enable_thinking"]},
            "separate_reasoning": s["separate_reasoning"],
        }
        if "top_k" in s:
            extra_body["top_k"] = s["top_k"]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": s["temperature"],
            "top_p": s["top_p"],
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        for opt in ("presence_penalty", "reasoning_effort"):
            if opt in s:
                kwargs[opt] = s[opt]
        if n > 1:
            kwargs["n"] = n

        response = self._retry(kwargs)
        return Call(
            texts=[_joined(c) for c in response.choices],
            finish_reasons=[c.finish_reason for c in response.choices],
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def _retry(self, kwargs: dict):
        for attempt in range(RETRIES):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(RETRY_WAIT)
