"""Provider-agnostic LLM adapter layer.

Agents import :class:`LLMClient` from this package; never a provider SDK
directly. Swapping Anthropic for Hugging Face (or anything else) is a
``config/models.yaml`` change, not a code change.
"""

from career_coach.llm.client import LLMClient, LLMResponse, Message
from career_coach.llm.factory import LLMFactory

__all__ = ["LLMClient", "LLMFactory", "LLMResponse", "Message"]
