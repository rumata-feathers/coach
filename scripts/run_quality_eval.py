#!/usr/bin/env python
"""Quality evaluation harness.

Runs 20 golden scenarios (10 Critic binary classifications + 10 Coach
categorical assertions) and produces a Markdown report.

Usage:
    uv run python scripts/run_quality_eval.py
    uv run python scripts/run_quality_eval.py --output reports/quality_baseline.md

The Critic battery asserts 'pass'/'reject' classification on hand-crafted
specimens. The Coach assertions check categorical properties of live responses
(e.g., "references user age", "does not hallucinate salary numbers").

Auto-skips if HUGGINGFACE_API_TOKEN is not set.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_coach.agents.coach import Coach
from career_coach.agents.critic import Critic
from career_coach.config import get_settings
from career_coach.llm.factory import LLMFactory
from career_coach.models.agent_io import CoachInput, CoachOutput, CriticInput
from career_coach.models.intent import IntentPacket
from career_coach.models.user_model import Hypothesis

# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------

HYPO_ID: UUID = uuid4()
USER_FACTS = {"name": "Alex", "age": 19, "location": "London", "education_stage": "undergrad"}
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

# ---------------------------------------------------------------------------
# Critic classification battery (10 specimens)
# ---------------------------------------------------------------------------

PASS_SPECIMENS: list[CoachOutput] = [
    CoachOutput(
        response_text=(
            "Alex, you're 19, studying in London — the job market here for economics "
            "grads is genuine, especially in finance and policy. Your analytical pull "
            "(which I've noticed) maps well onto what econ involves at undergrad level. "
            "Whether it's right for you post-grad depends on goals we haven't mapped yet."
        ),
        referenced_facts=["age", "location"],
        referenced_hypotheses=[HYPO_ID],
        uncertainty_flags=["post_grad_goals_unknown"],
    ),
    CoachOutput(
        response_text=(
            "To advise you well on economics I'd need to know your post-graduation goals. "
            "From what I know — 19, London, undergrad stage — the graduate market is real. "
            "That alone isn't enough to say whether it's right for you specifically."
        ),
        referenced_facts=["age", "location"],
        referenced_hypotheses=[],
        uncertainty_flags=["insufficient_context"],
    ),
    CoachOutput(
        response_text=(
            "Economics at undergrad level is heavily analytical — models, theory, data. "
            "Given your pattern of finding abstract problems more engaging than applied ones, "
            "that aligns well. I'd flag that year-3 specialisation gets narrow — "
            "unclear yet whether that'll stay interesting for you long-term."
        ),
        referenced_facts=["education_stage"],
        referenced_hypotheses=[HYPO_ID],
        uncertainty_flags=["long_term_subject_satisfaction_unknown"],
    ),
    CoachOutput(
        response_text=(
            "Being 19 and London-based gives you internship access many students lack. "
            "Econ opens doors in finance, consulting, public policy — though exact outcomes "
            "vary by university and your own effort, so I won't quote numbers. "
            "My suggestion: speak to a second-year econ student at your uni first."
        ),
        referenced_facts=["age", "location"],
        referenced_hypotheses=[],
        uncertainty_flags=["exact_outcomes_vary_by_institution"],
    ),
    CoachOutput(
        response_text=(
            "Given your analytical tendency and London undergrad context, economics looks "
            "like a plausible fit. The degree is abstract and model-heavy, which tends to "
            "reward people who enjoy pulling systems apart — which you do. "
            "I won't speculate on salary without knowing your expectations. "
            "One challenge: read both the Economist and a sociology journal for one "
            "afternoon each and notice which pulls you in more."
        ),
        referenced_facts=["location"],
        referenced_hypotheses=[HYPO_ID],
        uncertainty_flags=["salary_expectations_unknown"],
    ),
]

REJECT_SPECIMENS: list[tuple[CoachOutput, str]] = [
    (
        CoachOutput(
            response_text=(
                "Choosing a university subject is one of the most important decisions you'll make. "
                "Many young people find economics rewarding because it combines analytical thinking "
                "with real-world application. Career choices are deeply personal."
            ),
            referenced_facts=[],
            referenced_hypotheses=[],
            uncertainty_flags=[],
        ),
        "generic",
    ),
    (
        CoachOutput(
            response_text=(
                "Economics is a versatile degree that opens many career paths. "
                "You should consider your interests before committing. "
                "Speak to your career advisor for personalised guidance."
            ),
            referenced_facts=[],
            referenced_hypotheses=[],
            uncertainty_flags=[],
        ),
        "ungrounded",
    ),
    (
        CoachOutput(
            response_text=(
                "Economics is definitely the right choice for you. In London, "
                "economics graduates earn £52,000 on average in their first job. "
                "Most economics graduates find work within 3 months."
            ),
            referenced_facts=["location"],
            referenced_hypotheses=[],
            uncertainty_flags=[],
        ),
        "false_confidence",
    ),
    (
        CoachOutput(
            response_text=(
                "Alex, before deciding on any specific subject, it's worth stepping back and "
                "thinking about what you want your life to look like in 10 years. "
                "There are many paths — finance, tech, public sector — and economics is just one."
            ),
            referenced_facts=["age"],
            referenced_hypotheses=[],
            uncertainty_flags=[],
        ),
        "off_intent",
    ),
    (
        CoachOutput(
            response_text=(
                "It sounds like you're at an important crossroads. Career decisions at your age "
                "can feel overwhelming, but remember that many paths lead to success. "
                "Trust your instincts and seek guidance from mentors who know you well."
            ),
            referenced_facts=[],
            referenced_hypotheses=[],
            uncertainty_flags=[],
        ),
        "generic",
    ),
]


# ---------------------------------------------------------------------------
# Coach assertion scenarios (10 scenarios)
# ---------------------------------------------------------------------------

def _intent(turn_intent: str = "decide", specific_ask: str = INTENT.specific_ask) -> IntentPacket:
    return IntentPacket(
        session_theory=INTENT.session_theory,
        turn_intent=turn_intent,
        specific_ask=specific_ask,
        emotional_tenor="uncertain",
        clarity_score=0.8,
        needs_clarification=False,
        clarification_question=None,
        inferred_constraints=["UK undergrad"],
        budget_hint="standard",
    )


COACH_SCENARIOS: list[dict] = [
    {
        "name": "references_age_and_location",
        "description": "Response should reference user's age or location",
        "coach_input": CoachInput(
            intent_packet=_intent(),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("contains_user_fact", lambda out: any(
                word in out.response_text.lower()
                for word in ["19", "london", "undergrad"]
            ), "Response references a known user fact"),
        ],
    },
    {
        "name": "no_hallucinated_salary",
        "description": "Response should not cite a specific salary number without hedging",
        "coach_input": CoachInput(
            intent_packet=_intent(),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("no_unhedged_salary", lambda out: (
                "£" not in out.response_text or
                any(w in out.response_text.lower() for w in ["vary", "depend", "average", "roughly", "around", "uncertain", "i don't"])
            ), "If salary mentioned it must be hedged"),
        ],
    },
    {
        "name": "uncertainty_flags_populated",
        "description": "Response to a decide-intent should include at least one uncertainty flag",
        "coach_input": CoachInput(
            intent_packet=_intent(turn_intent="decide"),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("has_uncertainty_flags", lambda out: len(out.uncertainty_flags) > 0,
             "At least one uncertainty_flag present for decide-intent"),
        ],
    },
    {
        "name": "response_not_empty",
        "description": "Response text must be non-empty and substantial",
        "coach_input": CoachInput(
            intent_packet=_intent(),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("response_text_length", lambda out: len(out.response_text.split()) >= 30,
             "Response has at least 30 words"),
        ],
    },
    {
        "name": "referenced_facts_listed",
        "description": "Coach must list which facts it grounded on",
        "coach_input": CoachInput(
            intent_packet=_intent(),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("referenced_facts_nonempty", lambda out: len(out.referenced_facts) > 0,
             "referenced_facts is non-empty"),
        ],
    },
    {
        "name": "vent_intent_handled_gently",
        "description": "Vent intents should receive an empathetic response",
        "coach_input": CoachInput(
            intent_packet=_intent(
                turn_intent="vent",
                specific_ask="I'm so stressed about this decision, I can't think straight.",
            ),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("empathetic_words", lambda out: any(
                w in out.response_text.lower()
                for w in ["hear", "understand", "stress", "difficult", "feel", "tough", "overwhelming"]
            ), "Response acknowledges the emotional state"),
        ],
    },
    {
        "name": "explore_intent_broad_response",
        "description": "Explore intents should widen the conversation, not narrow it",
        "coach_input": CoachInput(
            intent_packet=_intent(
                turn_intent="explore",
                specific_ask="What should I be thinking about when it comes to my career options?",
            ),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("mentions_options", lambda out: any(
                w in out.response_text.lower()
                for w in ["option", "consider", "explore", "path", "interest", "question"]
            ), "Response broadens exploration"),
        ],
    },
    {
        "name": "reflect_intent_introspective",
        "description": "Reflect intents should be introspective and draw on memory",
        "coach_input": CoachInput(
            intent_packet=_intent(
                turn_intent="reflect",
                specific_ask="Looking back, what patterns do you notice in how I approach decisions?",
            ),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("references_hypothesis", lambda out: (
                len(out.referenced_hypotheses) > 0 or
                any(w in out.response_text.lower()
                    for w in ["analytical", "abstract", "pattern", "notice", "tend"])
            ), "Response references known patterns about the user"),
        ],
    },
    {
        "name": "json_schema_valid",
        "description": "Coach output always satisfies the Pydantic schema",
        "coach_input": CoachInput(
            intent_packet=_intent(),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("is_coach_output", lambda out: isinstance(out, CoachOutput),
             "Output is a valid CoachOutput instance"),
        ],
    },
    {
        "name": "no_direct_contradiction_of_facts",
        "description": "Response must not contradict known user facts",
        "coach_input": CoachInput(
            intent_packet=_intent(),
            user_facts=USER_FACTS,
            active_hypotheses=HYPOTHESES,
            recent_turns=[],
        ),
        "assertions": [
            ("no_wrong_age", lambda out: "20" not in out.response_text and "18" not in out.response_text,
             "Response does not claim user is 18 or 20 when they're 19"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_critic_battery(critic: Critic) -> list[dict]:
    results = []
    for i, specimen in enumerate(PASS_SPECIMENS):
        verdict = await critic.run(
            CriticInput(
                coach_output=specimen,
                user_facts=USER_FACTS,
                active_hypotheses=HYPOTHESES,
                intent_packet=INTENT,
            )
        )
        results.append({
            "id": f"critic_{i+1:02d}_pass",
            "expected": "pass",
            "got": verdict.verdict,
            "correct": verdict.verdict == "pass",
            "failure_modes": verdict.failure_modes,
            "description": f"PASS specimen #{i+1}",
        })

    for i, (specimen, mode) in enumerate(REJECT_SPECIMENS):
        verdict = await critic.run(
            CriticInput(
                coach_output=specimen,
                user_facts=USER_FACTS,
                active_hypotheses=HYPOTHESES,
                intent_packet=INTENT,
            )
        )
        results.append({
            "id": f"critic_{i+6:02d}_reject_{mode}",
            "expected": "reject",
            "expected_mode": mode,
            "got": verdict.verdict,
            "correct": verdict.verdict == "reject",
            "failure_modes": verdict.failure_modes,
            "description": f"REJECT specimen #{i+1} ({mode})",
        })
    return results


async def run_coach_assertions(coach: Coach) -> list[dict]:
    results = []
    for scenario in COACH_SCENARIOS:
        out = await coach.run(scenario["coach_input"])
        scenario_results = []
        for name, check_fn, desc in scenario["assertions"]:
            passed = False
            try:
                passed = bool(check_fn(out))
            except Exception as exc:
                passed = False
                desc = f"{desc} [EXCEPTION: {exc}]"
            scenario_results.append({"assertion": name, "passed": passed, "desc": desc})

        all_passed = all(r["passed"] for r in scenario_results)
        results.append({
            "id": f"coach_{scenario['name']}",
            "description": scenario["description"],
            "correct": all_passed,
            "assertions": scenario_results,
            "response_preview": out.response_text[:120].replace("\n", " "),
        })
    return results


def _render_report(
    critic_results: list[dict],
    coach_results: list[dict],
    elapsed_s: float,
) -> str:
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    critic_correct = sum(1 for r in critic_results if r["correct"])
    coach_correct = sum(1 for r in coach_results if r["correct"])
    total = len(critic_results) + len(coach_results)
    total_correct = critic_correct + coach_correct

    lines = [
        "# Career Coach — Quality Eval Baseline",
        f"\nGenerated: {now}  |  Elapsed: {elapsed_s:.1f}s",
        f"\n**Score: {total_correct}/{total}** "
        f"(Critic: {critic_correct}/{len(critic_results)}, Coach: {coach_correct}/{len(coach_results)})",
        "\n---\n",
        "## Critic Classification Battery\n",
    ]
    for r in critic_results:
        icon = "✓" if r["correct"] else "✗"
        lines.append(
            f"- [{icon}] `{r['id']}` — expected={r['expected']} got={r['got']}"
            + (f"  modes={r['failure_modes']}" if r["failure_modes"] else "")
        )

    lines += ["\n## Coach Assertion Scenarios\n"]
    for r in coach_results:
        icon = "✓" if r["correct"] else "✗"
        lines.append(f"- [{icon}] `{r['id']}`: {r['description']}")
        for a in r["assertions"]:
            a_icon = "✓" if a["passed"] else "✗"
            lines.append(f"    - [{a_icon}] {a['desc']}")
        lines.append(f"  > Preview: _{r['response_preview']}..._")

    return "\n".join(lines)


async def _main(output_path: Path | None) -> int:
    settings = get_settings()
    if not settings.huggingface_api_token:
        print("ERROR: HUGGINGFACE_API_TOKEN not set. Set it in .env and retry.", file=sys.stderr)
        return 1

    factory = LLMFactory()
    critic = Critic(factory)
    coach = Coach(factory)

    print("Running Critic battery (10 specimens)…", flush=True)
    t0 = asyncio.get_event_loop().time()
    critic_results = await run_critic_battery(critic)

    print("Running Coach assertions (10 scenarios)…", flush=True)
    coach_results = await run_coach_assertions(coach)
    elapsed = asyncio.get_event_loop().time() - t0

    report = _render_report(critic_results, coach_results, elapsed)
    print(report)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"\nReport written to {output_path}")

    critic_correct = sum(1 for r in critic_results if r["correct"])
    # Return non-zero exit code if Critic score < 8/10 (quality gate)
    return 0 if critic_correct >= 8 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quality evaluation harness.")
    parser.add_argument("--output", type=Path, help="Path to write Markdown report.")
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args.output)))
