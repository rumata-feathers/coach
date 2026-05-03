"""Hugging Face provider backed by an OpenAI-compatible client.

HF's Inference Providers system exposes chat-tuned open models through an
OpenAI-protocol endpoint at https://router.huggingface.co/v1 (the modern
Inference Providers router, replacing the legacy api-inference endpoint).

We can reuse the ``openai`` async SDK by pointing its ``base_url`` at that
router. Keep this adapter thin — swapping providers is a config change, not a
code change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

from career_coach.llm.client import LLMClient, LLMResponse, Message, ResponseFormat


def _last_balanced_object(text: str) -> str | None:
    """Return the last balanced ``{…}`` JSON object in *text*, or ``None``.

    Scans backwards from the last ``}`` to find its matching ``{``, correctly
    handling nested braces.  Used to rescue JSON that DeepSeek-V4-Flash
    (thinking mode) sometimes writes inside ``<think>`` rather than after it.
    """
    last_close = text.rfind("}")
    if last_close < 0:
        return None
    depth = 0
    for i in range(last_close, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                return text[i : last_close + 1]
    return None


# Inference Providers router (modern replacement for api-inference.huggingface.co/v1).
# Serves Qwen3, Llama, Kimi, DeepSeek and others via Novita/Nscale/Together/etc.
DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"


class HuggingFaceClient(LLMClient):
    """LLM client backed by the Hugging Face Inference Providers router."""

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str = DEFAULT_HF_BASE_URL,
        client: AsyncOpenAI | None = None,
    ):
        """Create a client.

        Pass ``client`` (a pre-built ``AsyncOpenAI``) for tests. In production
        the factory supplies ``api_token`` from settings.
        """
        if client is not None:
            self._client = client
        else:
            if not api_token:
                raise ValueError(
                    "HuggingFaceClient requires HUGGINGFACE_API_TOKEN in the environment."
                )
            self._client = AsyncOpenAI(api_key=api_token, base_url=base_url)

    async def complete(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: ResponseFormat = "text",
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Run a chat completion.

        ``extra_body`` is forwarded verbatim to the provider — useful for
        model-specific parameters like
        ``{"thinking_mode": "thinking"}`` on DeepSeek-V4-Flash to enable
        think mode, or ``{"chat_template_kwargs": {"enable_thinking": False}}``
        on MiMo-V2-Flash to suppress chain-of-thought.
        Whether the underlying backend honours it depends on the provider;
        the ``<think>`` stripping below is a reliable fallback regardless.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""

        # Strip think-block present in DeepSeek-V4-Flash (think mode) and
        # MiMo-V2-Flash. The block appears as <think>...</think> before the answer.
        if "<think>" in text and "</think>" in text:
            end = text.rfind("</think>")
            after_think = text[end + len("</think>"):].strip()
            if after_think:
                # Normal path: model wrote its answer after </think>.
                text = after_think
            else:
                # Fallback: model wrote its answer *inside* the think block.
                # This is a known DeepSeek quirk when response_format=json_object
                # is active — the model embeds the JSON in its reasoning chain and
                # produces nothing after </think>. Extract the last {...} object.
                think_start = text.find("<think>") + len("<think>")
                think_content = text[think_start:end]
                # Find the last *balanced* JSON object by scanning backwards.
                # A greedy regex would merge multiple objects; backward scan
                # correctly isolates the final one (the model's actual answer).
                last_json = _last_balanced_object(think_content)
                # If we found JSON inside the think block, use it; otherwise
                # yield empty string so the caller's parse-error retry kicks in.
                text = last_json or ""

        usage = response.usage
        return LLMResponse(
            text=text,
            model=model,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            raw={"finish_reason": choice.finish_reason},
        )
