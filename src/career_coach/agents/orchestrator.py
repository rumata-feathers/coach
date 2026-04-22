"""Orchestrator agent.

Pure Python — no LLM. Inspects the :class:`IntentPacket` and decides which
execution flow the turn should follow.

v0 flows:
  A — Coach only, no Critic. Used for vents and quick factual turns.
  B — Coach + Critic retry loop. Standard coaching conversation.
  C — Full deliberation (deferred to v1; routes to B for now).

See SPEC §6.2.
"""

from __future__ import annotations

from typing import Literal

from career_coach.models.intent import IntentPacket

Flow = Literal["A", "B"]


class Orchestrator:
    """Decides which flow to run based on the intent packet.

    No state, no LLM — just logic. Tests call ``decide`` directly.
    """

    def decide(self, intent: IntentPacket) -> Flow:
        """Return ``"A"`` (no Critic) or ``"B"`` (Coach + Critic).

        Rule per SPEC §6.2:
        - Vent or quick-budget turns → Flow A (skip Critic overhead).
        - Everything else → Flow B.
        """
        if intent.turn_intent == "vent" or intent.budget_hint == "quick":
            return "A"
        return "B"
