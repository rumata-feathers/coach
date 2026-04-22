"""Round-trip serialization tests for every agent I/O model.

Each test builds a minimally-valid instance, dumps it to JSON, parses it back,
and asserts structural equality. This is the cheapest possible contract check
— it catches any regression in schema shape, aliasing, or validator behaviour.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from career_coach.models import (
    Challenge,
    CoachInput,
    CoachOutput,
    CriticInput,
    CriticVerdict,
    EvidenceDraft,
    FactUpdate,
    Hypothesis,
    IntentPacket,
    ProfilerInput,
    ProfilerOutput,
    TurnSummary,
    UnderstanderInput,
)


def _intent_packet() -> IntentPacket:
    return IntentPacket(
        session_theory="User is weighing STEM vs humanities.",
        turn_intent="explore",
        specific_ask="Help me think about whether economics suits me.",
        emotional_tenor="curious",
        clarity_score=0.8,
        needs_clarification=False,
        clarification_question=None,
        inferred_constraints=["UK undergraduate", "undecided on location"],
        budget_hint="standard",
    )


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis_id=uuid4(),
        statement="User values intellectual breadth over narrow specialization.",
        confidence=0.4,
        status="active",
        open_questions=["How does this interact with pay ceiling?"],
    )


def _turn_summary() -> TurnSummary:
    return TurnSummary(
        turn_id=uuid4(),
        turn_index=3,
        user_message="I'm not sure I want to study finance.",
        assistant_message="What draws you away from it?",
        intent="explore",
    )


def _fact_update() -> FactUpdate:
    return FactUpdate(key="favourite_subject", value="economics", confidence=0.9)


def _evidence_draft() -> EvidenceDraft:
    return EvidenceDraft(
        source_type="user_statement",
        excerpt="I find abstract problems more energising than applied ones.",
        weight=0.8,
        hypothesis_hint="Prefers theoretical over applied work.",
    )


def _challenge() -> Challenge:
    return Challenge(
        title="Shadow an economist",
        description="Arrange a 30 min chat with a working economist.",
        estimated_effort="medium",
        ties_to=["career_interest_economics"],
    )


# ---- round-trip tests ---------------------------------------------------


@pytest.mark.parametrize(
    "original",
    [
        _intent_packet(),
        _hypothesis(),
        _turn_summary(),
        _fact_update(),
        _evidence_draft(),
        _challenge(),
    ],
)
def test_value_objects_round_trip(original: object) -> None:
    assert hasattr(original, "model_dump_json")
    clone = type(original).model_validate_json(original.model_dump_json())  # type: ignore[attr-defined]
    assert clone == original


def test_understander_input_round_trip() -> None:
    original = UnderstanderInput(
        user_message="I'm anxious about choosing a subject.",
        session_id=uuid4(),
        session_theory="First-year, strong in maths.",
        recent_turns=[_turn_summary()],
        user_facts={"age": 19, "location": "London"},
    )
    clone = UnderstanderInput.model_validate_json(original.model_dump_json())
    assert clone == original


def test_coach_input_and_output_round_trip() -> None:
    coach_input = CoachInput(
        intent_packet=_intent_packet(),
        user_facts={"age": 19},
        active_hypotheses=[_hypothesis()],
        recent_turns=[_turn_summary()],
        critic_feedback=None,
    )
    assert CoachInput.model_validate_json(coach_input.model_dump_json()) == coach_input

    coach_output = CoachOutput(
        response_text="Here's what stands out about your situation...",
        referenced_facts=["age", "education_stage"],
        referenced_hypotheses=[uuid4()],
        proposed_challenge=_challenge(),
        uncertainty_flags=["We don't know your location yet."],
    )
    assert CoachOutput.model_validate_json(coach_output.model_dump_json()) == coach_output


def test_critic_verdict_round_trip_and_validation() -> None:
    passing = CriticVerdict(verdict="pass", failure_modes=[], specific_complaints=[])
    assert CriticVerdict.model_validate_json(passing.model_dump_json()) == passing

    rejecting = CriticVerdict(
        verdict="reject",
        failure_modes=["generic"],
        specific_complaints=["Response could apply to any 19 year old."],
        suggested_fix="Ground response in user_facts and active hypotheses.",
    )
    assert CriticVerdict.model_validate_json(rejecting.model_dump_json()) == rejecting


def test_critic_verdict_reject_requires_failure_mode() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict(verdict="reject", failure_modes=[], specific_complaints=[])


def test_critic_verdict_pass_rejects_failure_modes() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict(verdict="pass", failure_modes=["ungrounded"], specific_complaints=[])


def test_critic_input_round_trip() -> None:
    critic_input = CriticInput(
        coach_output=CoachOutput(
            response_text="Short reply.",
            referenced_facts=["age"],
            referenced_hypotheses=[],
        ),
        user_facts={"age": 19},
        active_hypotheses=[_hypothesis()],
        intent_packet=_intent_packet(),
    )
    assert CriticInput.model_validate_json(critic_input.model_dump_json()) == critic_input


def test_profiler_io_round_trip() -> None:
    profiler_input = ProfilerInput(
        turn_id=uuid4(),
        user_message="I hated my internship at the bank.",
        assistant_message="What specifically bothered you?",
        existing_facts={"age": 19},
    )
    assert ProfilerInput.model_validate_json(profiler_input.model_dump_json()) == profiler_input

    profiler_output = ProfilerOutput(
        new_facts=[_fact_update()],
        fact_updates=[],
        hypothesis_evidence=[_evidence_draft()],
    )
    assert ProfilerOutput.model_validate_json(profiler_output.model_dump_json()) == profiler_output


def test_intent_packet_rejects_clarification_on_quick_budget() -> None:
    with pytest.raises(ValidationError):
        IntentPacket(
            session_theory="...",
            turn_intent="vent",
            specific_ask="just let me rant",
            emotional_tenor="frustrated",
            clarity_score=0.3,
            needs_clarification=True,
            clarification_question="what are you asking?",
            inferred_constraints=[],
            budget_hint="quick",
        )


def test_intent_packet_rejects_clarification_question_when_not_needed() -> None:
    with pytest.raises(ValidationError):
        IntentPacket(
            session_theory="...",
            turn_intent="explore",
            specific_ask="Am I built for software engineering?",
            emotional_tenor="uncertain",
            clarity_score=0.2,
            needs_clarification=True,
            clarification_question=None,
            inferred_constraints=[],
            budget_hint="standard",
        )


def test_evidence_weight_must_be_nonzero() -> None:
    with pytest.raises(ValidationError):
        EvidenceDraft(
            source_type="user_statement",
            excerpt="I feel meh about it.",
            weight=0.0,
        )
