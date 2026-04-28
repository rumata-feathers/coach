"""Orchestrator agent.

Pure Python — no LLM. Inspects the :class:`IntentPacket`, user state, and
fact count to decide which execution flow the turn should follow.

Flows:
  onboarding — Dedicated probe questions for users with < 5 facts OR < 3 turns.
  A          — Coach only, no Critic. Used for vents and quick factual turns.
  B          — Coach + Critic retry loop. Standard coaching conversation.
  C          — Full deliberation: parallel Researcher + Coach → DA →
               Synthesizer → Critic. Fires when user has ≥ 5 facts AND the
               intent signals a deep decision (§7.1).

See SPEC_v1.md §7.1 for Flow C routing conditions.
"""

from __future__ import annotations

from typing import Literal

from career_coach.models.intent import IntentPacket

Flow = Literal["A", "B", "C", "onboarding"]


class Orchestrator:
    """Decides which flow to run based on the intent packet and user state.

    No state, no LLM — just logic. Tests call ``decide`` directly.
    """

    def decide(
        self,
        intent: IntentPacket,
        *,
        is_new_user: bool = False,
        fact_count: int = 0,
    ) -> Flow:
        """Return the flow to execute for this turn.

        Args:
            intent: Parsed intent from the Understander.
            is_new_user: ``True`` when the user has fewer than 5 facts OR
                fewer than 3 total turns.  Supplied by
                :class:`~career_coach.pipeline.onboarding.OnboardingPolicy`.
            fact_count: Number of structured facts currently known about the
                user. Used to gate Flow C — requires ≥ 5 facts.

        Returns:
            ``"onboarding"`` for new users, ``"A"`` for vents/quick turns,
            ``"C"`` for deep decisions with sufficient context,
            ``"B"`` for everything else.
        """
        if is_new_user:
            return "onboarding"
        if intent.turn_intent == "vent" or intent.budget_hint == "quick":
            return "A"
        if (intent.budget_hint == "deep" or intent.turn_intent == "decide") and fact_count >= 5:
            return "C"
        return "B"
