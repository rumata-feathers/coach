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
        ``{"chat_template_kwargs": {"enable_thinking": False}}`` on Qwen3 to
        suppress the chain-of-thought block (correct HF/vLLM syntax; the
        Alibaba-specific ``{"thinking": false}`` causes 400 errors here).
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

        # Strip Qwen3 think-block if thinking was accidentally left enabled.
        # The block appears as <think>...</think> before the actual answer.
        if "<think>" in text and "</think>" in text:
            end = text.rfind("</think>")
            text = text[end + len("</think>"):].strip()

        usage = response.usage
        return LLMResponse(
            text=text,
            model=model,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            raw={"finish_reason": choice.finish_reason},
        )
