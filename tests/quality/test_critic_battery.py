"""Critic test battery — the quality canary.

10 hand-crafted CoachOutput specimens: 5 that should PASS and 5 that should
REJECT. The Critic must correctly classify ≥ 8/10 to pass this battery.

Auto-skipped when ``ANTHROPIC_API_KEY`` is not set.

Context: test user is a 19-year-old undergrad in London, weighing whether to
study economics. Active hypothesis: "User gravitates toward analytical work."
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from career_coach.agents.critic import Critic
from career_coach.llm.factory import LLMFactory
from career_coach.models.agent_io import CoachOutput, CriticInput
from career_coach.models.intent import IntentPacket
from career_coach.models.user_model import Hypothesis

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

pytestmark = pytest.mark.asyncio

USER_FACTS = {
    "name": "Alex",
    "age": 19,
    "location": "London",
    "education_stage": "undergrad",
}

HYPO_ID: UUID = uuid4()
HYPOTHESES = [
    Hypothesis(
        hypothesis_id=HYPO_ID,
        statement="User gravitates toward analytical and abstract work over applied tasks.",
        confidence=0.6,
        status="active",
    )
]

INTENT = IntentPacket(
    session_theory="User weighing economics vs other degree options.",
    turn_intent="decide",
    specific_ask="Should I study economics given what you know about me?",
    emotional_tenor="uncertain",
    clarity_score=0.8,
    needs_clarification=False,
    clarification_question=None,
    inferred_constraints=["UK undergrad", "London-based"],
    budget_hint="standard",
)


# ---- PASS specimens (should return verdict="pass") ---------------------

PASS_SPECIMENS: list[CoachOutput] = [
    # 1. Well-grounded, specific, hedged confidence.
    CoachOutput(
        response_text=(
            "Alex, given that you're 19 and doing your undergrad in London, "
            "economics is worth seriously considering — the London job market "
            "for economics graduates is genuinely strong, particularly in finance "
            "and policy roles. Your tendency toward analytical and abstract "
            "thinking (which I've noticed across our conversations) maps well onto "
            "what economics actually involves day-to-day at university level. "
            "That said, whether the career outcomes justify it depends heavily on "
            "what you want after graduation, which we haven't fully mapped yet."
        ),
        referenced_facts=["age", "location"],
        referenced_hypotheses=[HYPO_ID],
        uncertainty_flags=["career_outcome_depends_on_post_grad_goals"],
    ),
    # 2. Honest 'I need more context' — sparse model, correct behaviour.
    CoachOutput(
        response_text=(
            "To give you a genuinely useful answer about economics, I'd need to "
            "know more about what draws you to it and what you'd want to do "
            "afterwards. From what I know — you're 19, based in London, at "
            "undergrad stage — I can say the London job market is broad for "
            "economics graduates, but that's not enough to advise you on whether "
            "it's the right choice for you specifically."
        ),
        referenced_facts=["age", "location"],
        referenced_hypotheses=[],
        uncertainty_flags=["insufficient_context_for_specific_advice"],
    ),
    # 3. Directly addresses specific ask, uses hypothesis explicitly.
    CoachOutput(
        response_text=(
            "Based on your undergrad stage and what you've told me about yourself, "
            "my honest view is that economics is a plausible fit. The subject is "
            "heavily analytical — models, data interpretation, theory — which "
            "aligns with the pattern I've observed where you find abstract problems "
            "more engaging than applied ones. "
            "One thing I'd flag as uncertain: whether the analytical rigour you'd "
            "find satisfying in year 1 stays interesting in year 3 when it gets "
            "very technical and narrow."
        ),
        referenced_facts=["education_stage"],
        referenced_hypotheses=[HYPO_ID],
        uncertainty_flags=["long_term_subject_satisfaction_unknown"],
    ),
    # 4. Confident in hedged claims, uses two facts, on-intent.
    CoachOutput(
        response_text=(
            "You're in a good position to make this decision, Alex. Being 19 and "
            "based in London means you have access to internships and networking "
            "that students elsewhere don't. Economics from a London university "
            "opens doors in finance, consulting, and public policy — though exact "
            "outcomes vary a lot by university and your own effort, so I won't "
            "quote specific numbers. "
            "What I'd actually suggest is talking to a second-year economics "
            "student at your university before deciding."
        ),
        referenced_facts=["age", "location"],
        referenced_hypotheses=[],
        proposed_challenge=None,
        uncertainty_flags=["exact_outcomes_vary_by_institution"],
    ),
    # 5. References hypothesis + fact, proposes a challenge, hedges salary.
    CoachOutput(
        response_text=(
            "Given your pull toward analytical thinking and your London undergrad "
            "context, economics feels like a reasonable fit — at least on paper. "
            "The degree is abstract and model-heavy in a way that tends to reward "
            "people who like pulling systems apart. I don't know your salary "
            "expectations, so I won't speculate there, but the graduate market "
            "in London for economics is real. "
            "One challenge I'd suggest: spend one afternoon reading the Economist "
            "and one afternoon reading a sociology journal. Notice which one pulls "
            "you in — that's useful signal."
        ),
        referenced_facts=["location"],
        referenced_hypotheses=[HYPO_ID],
        proposed_challenge=None,
        uncertainty_flags=["salary_expectations_unknown"],
    ),
]


# ---- REJECT specimens (should return verdict="reject") -----------------

REJECT_SPECIMENS: list[tuple[CoachOutput, str]] = [
    # 1. Generic — could apply to literally any 19-year-old.
    (
        CoachOutput(
            response_text=(
                "Choosing a university subject is one of the most important decisions "
                "you'll make. Many young people find economics rewarding because it "
                "combines analytical thinking with real-world application. I'd encourage "
                "you to explore your options, talk to current students, and reflect on "
                "what kind of work energises you. Career choices are deeply personal "
                "and ultimately only you can make them."
            ),
            referenced_facts=[],
            referenced_hypotheses=[],
            uncertainty_flags=[],
        ),
        "generic",
    ),
    # 2. Ungrounded — references nothing from user model despite having facts available.
    (
        CoachOutput(
            response_text=(
                "Economics is a versatile degree that opens many career paths. "
                "You should consider your interests and strengths before committing. "
                "Speak to your career advisor for personalised guidance tailored to "
                "your situation."
            ),
            referenced_facts=[],  # nothing referenced despite having user facts
            referenced_hypotheses=[],
            uncertainty_flags=[],
        ),
        "ungrounded",
    ),
    # 3. False confidence — specific salary number with no hedging.
    (
        CoachOutput(
            response_text=(
                "Economics is definitely the right choice for you. In London, "
                "economics graduates earn £52,000 on average in their first job, "
                "which is one of the highest starting salaries of any undergraduate "
                "degree. Most economics graduates find work within 3 months, and "
                "the majority end up in finance or consulting within 2 years."
            ),
            referenced_facts=["location"],
            referenced_hypotheses=[],
            uncertainty_flags=[],  # no hedging on the specific numbers
        ),
        "false_confidence",
    ),
    # 4. Off-intent — user asked about economics specifically; response pivots to
    #    generic career exploration instead.
    (
        CoachOutput(
            response_text=(
                "Alex, before deciding on any specific subject, it's worth stepping "
                "back and thinking about what you want your life to look like in 10 "
                "years. There are many paths — finance, tech, public sector, arts — "
                "and economics is just one option. I'd recommend doing a broader "
                "career exploration exercise before narrowing down."
            ),
            referenced_facts=["age"],
            referenced_hypotheses=[],
            uncertainty_flags=[],
        ),
        "off_intent",  # ignores the specific ask about economics
    ),
    # 5. Generic + ungrounded — double failure.
    (
        CoachOutput(
            response_text=(
                "It sounds like you're at an important crossroads. Career decisions "
                "at your age can feel overwhelming, but remember that many paths lead "
                "to success. Economics is a great option for those who enjoy numbers "
                "and analysis. Trust your instincts and seek guidance from mentors "
                "who know you well."
            ),
            referenced_facts=[],
            referenced_hypotheses=[],
            uncertainty_flags=[],
        ),
        "generic",
    ),
]


# ---- Battery runner ----------------------------------------------------


@pytest.fixture(scope="module")
def critic() -> Critic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping Critic battery")
    return Critic(LLMFactory(config_path=_CONFIG))


async def test_critic_battery(critic: Critic) -> None:
    """Run all 10 specimens and assert ≥ 8/10 correct classifications."""
    results: list[dict[str, object]] = []

    # PASS specimens
    for i, specimen in enumerate(PASS_SPECIMENS):
        verdict = await critic.run(
            CriticInput(
                coach_output=specimen,
                user_facts=USER_FACTS,
                active_hypotheses=HYPOTHESES,
                intent_packet=INTENT,
            )
        )
        correct = verdict.verdict == "pass"
        results.append(
            {
                "index": i + 1,
                "expected": "pass",
                "got": verdict.verdict,
                "correct": correct,
                "failure_modes": verdict.failure_modes,
            }
        )

    # REJECT specimens
    for i, (specimen, expected_mode) in enumerate(REJECT_SPECIMENS):
        verdict = await critic.run(
            CriticInput(
                coach_output=specimen,
                user_facts=USER_FACTS,
                active_hypotheses=HYPOTHESES,
                intent_packet=INTENT,
            )
        )
        correct = verdict.verdict == "reject"
        results.append(
            {
                "index": i + 6,
                "expected": "reject",
                "expected_mode": expected_mode,
                "got": verdict.verdict,
                "correct": correct,
                "failure_modes": verdict.failure_modes,
            }
        )

    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)

    # Print a summary so failures are diagnosable.
    print(f"\nCritic battery: {correct_count}/{total} correct")
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(
            f"  [{status}] #{r['index']} expected={r['expected']} got={r['got']}"
            f"  modes={r.get('failure_modes', [])}"
        )

    assert correct_count >= 8, (
        f"Critic battery: only {correct_count}/{total} correct (need ≥ 8). "
        f"See output above for which specimens failed."
    )
