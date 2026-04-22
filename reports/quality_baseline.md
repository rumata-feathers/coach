# Career Coach — Quality Eval Baseline

Generated: 2026-04-22 13:02 UTC  |  Elapsed: 296.7s

**Score: 19/20** (Critic: 10/10, Coach: 9/10)

---

## Critic Classification Battery

- [✓] `critic_01_pass` — expected=pass got=pass
- [✓] `critic_02_pass` — expected=pass got=pass
- [✓] `critic_03_pass` — expected=pass got=pass
- [✓] `critic_04_pass` — expected=pass got=pass
- [✓] `critic_05_pass` — expected=pass got=pass
- [✓] `critic_06_reject_generic` — expected=reject got=reject  modes=['generic', 'ungrounded', 'false_confidence', 'off_intent']
- [✓] `critic_07_reject_ungrounded` — expected=reject got=reject  modes=['generic', 'ungrounded', 'false_confidence']
- [✓] `critic_08_reject_false_confidence` — expected=reject got=reject  modes=['generic', 'ungrounded', 'false_confidence']
- [✓] `critic_09_reject_off_intent` — expected=reject got=reject  modes=['generic', 'ungrounded']
- [✓] `critic_10_reject_generic` — expected=reject got=reject  modes=['generic', 'ungrounded', 'off_intent']

## Coach Assertion Scenarios

- [✗] `coach_references_age_and_location`: Response should reference user's age or location
    - [✗] Response references a known user fact
  > Preview: _Alex, given your interest in analytical and abstract work, economics could align well with your strengths. The field oft..._
- [✓] `coach_no_hallucinated_salary`: Response should not cite a specific salary number without hedging
    - [✓] If salary mentioned it must be hedged
  > Preview: _Alex, based on what we know, economics could align with your preference for analytical and abstract work (as suggested b..._
- [✓] `coach_uncertainty_flags_populated`: Response to a decide-intent should include at least one uncertainty flag
    - [✓] At least one uncertainty_flag present for decide-intent
  > Preview: _Alex, based on what I know, economics could align with your analytical preferences but requires careful consideration. Y..._
- [✓] `coach_response_not_empty`: Response text must be non-empty and substantial
    - [✓] Response has at least 30 words
  > Preview: _Alex, considering your interest in analytical and abstract thinking, economics could align well with your strengths. The..._
- [✓] `coach_referenced_facts_listed`: Coach must list which facts it grounded on
    - [✓] referenced_facts is non-empty
  > Preview: _Alex, based on what I know, economics could align with your analytical interests, but let’s unpack this carefully. The f..._
- [✓] `coach_vent_intent_handled_gently`: Vent intents should receive an empathetic response
    - [✓] Response acknowledges the emotional state
  > Preview: _Alex, it's completely valid to feel overwhelmed when facing decisions that align with your analytical nature. As a Londo..._
- [✓] `coach_explore_intent_broad_response`: Explore intents should widen the conversation, not narrow it
    - [✓] Response broadens exploration
  > Preview: _Alex, since you're in your undergrad years in London, this is a great time to explore foundational questions without pre..._
- [✓] `coach_reflect_intent_introspective`: Reflect intents should be introspective and draw on memory
    - [✓] Response references known patterns about the user
  > Preview: _Looking back, I notice a few patterns that might resonate with you. First, your tendency toward analytical thinking show..._
- [✓] `coach_json_schema_valid`: Coach output always satisfies the Pydantic schema
    - [✓] Output is a valid CoachOutput instance
  > Preview: _Alex, based on your interest in analytical and abstract thinking (as suggested by our hypothesis about your preferences)..._
- [✓] `coach_no_direct_contradiction_of_facts`: Response must not contradict known user facts
    - [✓] Response does not claim user is 18 or 20 when they're 19
  > Preview: _Given your interest in analytical and abstract work, economics could align well with your strengths. The field blends th..._