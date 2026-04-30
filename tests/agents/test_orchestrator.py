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
        # fact_count=0 (default) — Flow C never fires
        ("explore", "standard", "B"),
        ("decide", "standard", "B"),   # decide but < 5 facts → B
        ("reflect", "standard", "B"),
        ("research", "standard", "B"),
        ("research", "deep", "B"),     # deep but < 5 facts → B
        ("vent", "standard", "A"),     # vent always → A
        ("explore", "quick", "A"),     # quick always → A
        ("vent", "quick", "A"),        # both conditions → A
    ],
)
def test_orchestrator_flow_routing(turn_intent: str, budget_hint: str, expected_flow: str) -> None:
    orch = Orchestrator()
    packet = _packet(turn_intent=turn_intent, budget_hint=budget_hint)
    assert orch.decide(packet) == expected_flow


@pytest.mark.parametrize(
    "turn_intent,budget_hint",
    [
        ("decide", "standard"),   # decide intent alone triggers C when ≥5 facts
        ("explore", "deep"),      # deep budget alone triggers C when ≥5 facts
        ("decide", "deep"),       # both conditions — still C
    ],
)
def test_orchestrator_routes_to_flow_c_with_enough_facts(
    turn_intent: str, budget_hint: str
) -> None:
    """Flow C fires when decide/deep AND fact_count ≥ 5 (§7.1)."""
    orch = Orchestrator()
    packet = _packet(turn_intent=turn_intent, budget_hint=budget_hint)
    assert orch.decide(packet, fact_count=5) == "C"
    assert orch.decide(packet, fact_count=10) == "C"


def test_orchestrator_flow_c_requires_five_facts() -> None:
    """Below 5 facts, decide/deep routes to B, not C."""
    orch = Orchestrator()
    packet = _packet(turn_intent="decide", budget_hint="deep")
    assert orch.decide(packet, fact_count=0) == "B"
    assert orch.decide(packet, fact_count=4) == "B"
    assert orch.decide(packet, fact_count=5) == "C"


def test_orchestrator_flow_c_never_fires_on_vent_or_quick() -> None:
    """Vent and quick budget override Flow C even with many facts."""
    orch = Orchestrator()
    for turn_intent, budget_hint in [("vent", "deep"), ("explore", "quick")]:
        packet = _packet(turn_intent=turn_intent, budget_hint=budget_hint)
        result = orch.decide(packet, fact_count=10)
        assert result == "A", (
            f"Expected 'A' for {turn_intent}/{budget_hint} but got {result!r}"
        )


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
