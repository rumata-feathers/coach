"""Embedding generation helper.

Uses the Hugging Face Inference API's feature-extraction endpoint to produce
dense vectors from text. The endpoint returns a nested list for batches, but
we expose a single-string interface to keep call sites simple.

Dimension matches EMBEDDING_DIM in config (default 384 for MiniLM-L6-v2).
"""

from __future__ import annotations

import httpx

from career_coach.config import get_settings

_HF_FEATURE_EXTRACTION_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
)


class EmbeddingsClient:
    """Generates embeddings via the HF Inference API."""

    def __init__(
        self,
        api_token: str | None = None,
        model: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._token = api_token or settings.huggingface_api_token or ""
        self._model = model or settings.embedding_model
        self._url = _HF_FEATURE_EXTRACTION_URL.format(model=self._model)
        self._http = http_client  # injected in tests to avoid real HTTP calls

    async def embed(self, text: str) -> list[float]:
        """Return a dense vector for ``text``.

        The HF feature-extraction endpoint returns either:
          - A flat list of floats (for some models)
          - A list of lists (sentence-transformers often do mean-pooled output)

        We handle both and always return a flat list.
        """
        client = self._http or httpx.AsyncClient()
        close_after = self._http is None
        try:
            response = await client.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()
        finally:
            if close_after:
                await client.aclose()

        return _flatten(payload)


def _flatten(payload: object) -> list[float]:
    """Recursively unwrap nested lists until we reach a flat list of floats.

    Some HF models return ``[[float, ...]]`` (batch of 1) or even
    ``[[[float, ...]]]`` for token-level embeddings with pooling applied.
    Mean-pool the innermost vectors if we end up with a 2D list.
    """
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Embedding API returned an empty list.")
        if isinstance(payload[0], float):
            return payload
        if isinstance(payload[0], int):
            return [float(v) for v in payload]
        # Nested — mean-pool the inner vectors.
        inner: list[list[float]] = [_flatten(item) for item in payload]
        n = len(inner)
        dim = len(inner[0])
        return [sum(inner[i][d] for i in range(n)) / n for d in range(dim)]
    raise TypeError(f"Unexpected embedding payload type: {type(payload)}")
