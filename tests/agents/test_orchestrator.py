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
