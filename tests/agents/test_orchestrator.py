"""Unit tests for :class:`Orchestrator`."""

from __future__ import annotations

import pytest

from career_coach.agents.orchestrator import Orchestrator
from career_coach.models.intent import IntentPacket


def _packet(**kw: object) -> IntentPacket:
    defaults: dict[str, object] = {
        "session_theory": "test",
        "turn_intent": "explore",
        "specific_ask": "test ask",
        "emotional_tenor": "curious",
        "clarity_score": 0.8,
        "needs_clarification": False,
        "clarification_question": None,
        "inferred_constraints": [],
        "budget_hint": "standard",
    }
    defaults.update(kw)
    return IntentPacket(**defaults)


@pytest.mark.parametrize(
    "turn_intent,budget_hint,expected_flow",
    [
        ("explore", "standard", "B"),
        ("decide", "standard", "B"),
        ("reflect", "standard", "B"),
        ("research", "standard", "B"),
        ("research", "deep", "B"),
        ("vent", "standard", "A"),  # vent always → A
        ("explore", "quick", "A"),  # quick always → A
        ("vent", "quick", "A"),  # both conditions → A
    ],
)
def test_orchestrator_flow_routing(turn_intent: str, budget_hint: str, expected_flow: str) -> None:
    orch = Orchestrator()
    packet = _packet(turn_intent=turn_intent, budget_hint=budget_hint)
    assert orch.decide(packet) == expected_flow


def test_orchestrator_routes_new_users_to_onboarding() -> None:
    """is_new_user=True must override all other routing logic and return 'onboarding'."""
    orch = Orchestrator()
    # Even a vent or quick turn should go to onboarding for a new user.
    for turn_intent in ("vent", "explore", "decide"):
        for budget_hint in ("quick", "standard"):
            packet = _packet(turn_intent=turn_intent, budget_hint=budget_hint)
            result = orch.decide(packet, is_new_user=True)
            assert result == "onboarding", (
                f"Expected 'onboarding' for is_new_user=True, "
                f"got {result!r} (intent={turn_intent}, budget={budget_hint})"
            )


def test_orchestrator_existing_users_not_routed_to_onboarding() -> None:
    """is_new_user=False must never return 'onboarding'."""
    orch = Orchestrator()
    for turn_intent in ("explore", "decide", "reflect"):
        packet = _packet(turn_intent=turn_intent, budget_hint="standard")
        result = orch.decide(packet, is_new_user=False)
        assert result != "onboarding"
