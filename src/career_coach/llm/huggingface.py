"""Hugging Face provider backed by an OpenAI-compatible client.

HF's Inference API exposes chat-tuned open models through an OpenAI-protocol
endpoint, so we can reuse the ``openai`` async SDK by pointing its ``base_url``
at HF. Keep this adapter thin — the whole point of the LLM layer is that
provider changes are configuration, not code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

from career_coach.llm.client import LLMClient, LLMResponse, Message, ResponseFormat

DEFAULT_HF_BASE_URL = "https://api-inference.huggingface.co/v1"


class HuggingFaceClient(LLMClient):
    """LLM client backed by the Hugging Face Inference API."""

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
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            # Not every HF model honours this, but OpenAI-protocol endpoints
            # accept the field and many chat-tuned models respect it.
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage

        return LLMResponse(
            text=text,
            model=model,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            raw={"finish_reason": choice.finish_reason},
        )
