"""Orchestrator agent.

Pure Python — no LLM. Inspects the :class:`IntentPacket` and an ``is_new_user``
flag to decide which execution flow the turn should follow.

v0.5 flows:
  onboarding — Dedicated probe questions for users with < 5 facts OR < 3 turns.
  A          — Coach only, no Critic. Used for vents and quick factual turns.
  B          — Coach + Critic retry loop. Standard coaching conversation.
  C          — Full deliberation (deferred to v1; routes to B for now).

See SPEC §6.2 and SPEC_v0.5 §4.
"""

from __future__ import annotations

from typing import Literal

from career_coach.models.intent import IntentPacket

Flow = Literal["A", "B", "onboarding"]


class Orchestrator:
    """Decides which flow to run based on the intent packet and user state.

    No state, no LLM — just logic. Tests call ``decide`` directly.
    """

    def decide(self, intent: IntentPacket, *, is_new_user: bool = False) -> Flow:
        """Return the flow to execute for this turn.

        Args:
            intent: Parsed intent from the Understander.
            is_new_user: ``True`` when the user has fewer than 5 facts OR
                fewer than 3 total turns.  Supplied by
                :class:`~career_coach.pipeline.onboarding.OnboardingPolicy`.

        Returns:
            ``"onboarding"`` for new users, ``"A"`` for vents/quick turns,
            ``"B"`` for everything else.
        """
        if is_new_user:
            return "onboarding"
        if intent.turn_intent == "vent" or intent.budget_hint == "quick":
            return "A"
        return "B"
