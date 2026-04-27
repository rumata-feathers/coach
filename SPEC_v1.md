# Career Coach — v1 Specification (Research Coach)

## Context

v0 shipped a walking skeleton. v0.5 consolidated reliability, onboarding, session continuity, distillation quality, and memory-loop closure.

**v1 is the Research Coach.** The system gains the ability to answer questions with the depth and citation hygiene of a research assistant: real web search, structured charts where they help, and a deliberation flow that integrates research with the user's profile.

**v1.5 (the next milestone) is the Active Coach** — proactive check-ins and project-style challenges. v1 is deliberately scoped to *synchronous response quality*; the temporal layer (initiating, scheduling, tracking multi-turn project state) is held back to v1.5 because it changes the turn loop's invariants, not just its agents.

Read `SPEC.md` (v0) for the architectural contract and `SPEC_v0_5.md` (v0.5) for the consolidation layer. This document describes **additions for v1**. Anything not re-specified here is unchanged from v0.5.

> **Scope discipline.** Per `HANDOFF_v0_5.md`, do not start v1 without first running the v0.5 definition-of-done on a clean DB AND using the deployed v0.5 system for at least a week. The week-of-usage data may revise §1 of this spec — particularly the Flow C trigger frequency and whether real users ask research-shaped questions at all.

---

## 1. What v1 adds (seven pillars)

1. **Web search adapter.** Provider-agnostic, mirrors the LLM adapter pattern. Tavily default, Exa and Anthropic-native as alternates.
2. **Researcher agent (web + KB).** Web search primary for fresh information, static KB supplemental for stable reference data. Citations on every finding.
3. **Charts as structured output.** Coach and Synthesizer can emit `chart_specs`. v1 emits the spec; v2 frontend renders them.
4. **Flow C — Deliberation.** Multi-agent path for deep decisions: parallel Researcher + Coach → Devil's Advocate → Synthesizer → Critic.
5. **Devil's Advocate agent.** Structured contrarian pressure on Coach output.
6. **Synthesizer agent.** Integrates Coach + DA + Researcher into one coherent response (with optional charts).
7. **Supervisor agent.** Cross-cutting runtime sanity monitor — fact contradictions, off-topic drift, unsafe content. Runs on every flow.

Plus: **LangGraph migration**, executed last, when the graph is fully populated and a refactor is justified.

---

## 2. What v1 does NOT do (deferred to v1.5)

- **Challenges / projects.** The `proposed_challenge` field exists in Coach output (legacy from v0) but does not yet have a real lifecycle, persistence, or follow-up logic. v1.5 promotes it to a first-class object.
- **Initiator agent.** No proactive check-ins in v1. Every turn is still user-initiated.
- **Cron / n8n / scheduler.** No external triggers.
- **Multi-turn project state tracking.** v1 has no notion of an in-progress assignment.

## 3. What v1 also does NOT do (deferred to v2+)

- Evaluator (slow meta-director)
- God-agent
- Frontend (chart specs ship in v1 but rendering is v2)
- Auth (still single hardcoded test user)
- Connector / human practitioner network
- Web search inside non-Researcher agents
- Distillation clustering / coherence review (v0.5's pairwise dedup remains sufficient)

---

## 4. Pillar 1 — Web Search Adapter

### 4.1 Why an adapter, not a direct integration

Same reason as the LLM adapter: lock-in to a single search provider is a tax we can't predict. Tavily is cheap and designed for AI use; Exa returns higher-quality results but costs more; Anthropic's native `web_search` tool is convenient when running on Anthropic but unavailable when running on Qwen3-via-HF. Make the swap a config change.

### 4.2 Contract

```python
# career_coach/web/client.py

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str                       # short excerpt from the search provider
    content: str | None                # full extracted text, populated by fetch()
    published_date: date | None
    score: float | None                # provider's relevance score, 0-1, optional

class WebSearchClient(Protocol):
    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,    # restrict to recent results when set
    ) -> list[SearchResult]: ...

    async def fetch(self, url: str) -> str:
        """Fetch and extract main content from a URL. Returns clean text, no HTML."""
        ...
```

### 4.3 Implementations

- `TavilyClient` — default. Uses Tavily Search API. Has built-in answer-extraction; we use the raw results.
- `ExaClient` — alternate. Use when result quality matters more than cost.
- `AnthropicWebSearchClient` — uses the `web_search` tool inside an Anthropic API call. Only available when Anthropic is the LLM provider for the agent.

Selected per-agent in `config/web_search.yaml`:

```yaml
researcher:
  provider: tavily
  max_results_default: 5
  recency_days_default: null
```

### 4.4 Caching

In-memory LRU cache, keyed on `(provider, query, max_results, recency_days)`, TTL 24h. Cuts cost during eval runs (the same fixture queries fire repeatedly) and during retries. Cache is per-process — sufficient for v1; v2 can introduce a shared cache.

### 4.5 Rate limiting and budgets

`WebSearchClient` enforces a per-process daily quota (default 200 calls). When exceeded, returns empty results with a flag so the Researcher can degrade gracefully (caveat: "search unavailable, falling back to KB only"). This protects from runaway eval loops.

### 4.6 Observability

Every web search call logs to `agent_calls` with `agent_name="web_search"`, `model_used="<provider>"`, query in `input_payload`, result count in `output_payload.result_count`, latency, and an `error` field when relevant. No special table for v1.

---

## 5. Pillar 2 — Researcher (web + KB)

### 5.1 Purpose

Given a research question and the user's facts, produce a grounded ResearchBrief drawing on **both** web search and the static KB. Cite every finding.

### 5.2 Web vs KB — when to use each

- **KB-first** for stable reference: what is investment banking, what does a quant role look like, typical entry paths. The KB is hand-curated and trustworthy for these.
- **Web-first** for fresh, specific, or numeric: current pay bands, hiring trends, recent layoffs, named firms, comparing two specific programs.
- **Both** when the question spans both: "is quant finance still a good career in 2026" needs the KB's structural framing AND web-fresh signal.

The Researcher prompt explicitly asks the model to decide which sources are needed and to mark each finding with its source type.

### 5.3 Contract

**Model:** Coach-tier (strong).

```python
class ResearcherInput(BaseModel):
    question: str
    user_facts: dict
    active_hypotheses: list[Hypothesis]
    depth: Literal["shallow", "deep"] = "shallow"
    # depth controls: shallow = ≤3 web searches, ≤2 fetches, ≤1 KB file deep-read
    #                 deep    = ≤8 web searches, ≤5 fetches, full matched KB

class Citation(BaseModel):
    url: str | None              # null for KB sources
    kb_path: str | None          # null for web sources
    title: str
    accessed_at: datetime
    source_type: Literal["web", "kb"]

class Finding(BaseModel):
    claim: str
    confidence: float            # 0-1, Researcher's own confidence
    citations: list[Citation]    # at least 1 required
    is_numeric: bool             # numeric findings get extra scrutiny in Critic

class ResearchBrief(BaseModel):
    question: str
    findings: list[Finding]      # at least 2 on success; empty on graceful failure
    caveats: list[str]
    next_questions: list[str]
    used_kb_files: list[str]
    web_searches_run: list[str]  # the queries actually issued
    web_sources_consulted: list[str]  # URLs that contributed to findings
```

### 5.4 Retrieval mechanics

Two phases:

1. **Plan.** A small LLM call (Haiku-tier) takes the question + user_facts and returns a `RetrievalPlan`: a list of search queries to run AND a list of KB tags to look up. This is cheap and lets the model reason about *what* to look for separately from *how to synthesize*.
2. **Execute and synthesize.** Run searches in parallel via `WebSearchClient.search`, fetch top-2 promising results per query in parallel via `WebSearchClient.fetch`, load matched KB files. Pass everything to the Researcher's main LLM call which produces the ResearchBrief.

Total Researcher latency budget: 8s shallow, 15s deep. If exceeded, return what's been gathered so far with a `caveat: "research truncated due to time budget"`.

### 5.5 Citation hygiene rules

These are encoded in `config/prompts/researcher.j2` and checked by the Critic:

- Every `Finding` MUST have ≥1 citation.
- Numeric findings (`is_numeric=True`) MUST have ≥1 web citation OR a KB citation with explicit `source_notes` (the v0.5 KB convention).
- A finding cannot cite a search result snippet alone — the snippet must have led to a `fetch()` whose content actually contained the claim. (Enforced by passing fetched content, not just snippets, into the synthesis call.)
- Citations are deduplicated by URL/kb_path before persistence.

### 5.6 Persistence

`research_briefs` table (migration 004) stores every brief, linked to the turn. Empty briefs are persisted too — they're the signal for "we tried, nothing useful turned up."

---

## 6. Pillar 3 — Charts as structured output

### 6.1 Why charts in v1

For deep-decision questions ("compare quant vs SWE comp progression," "show me how PhD timelines play out across fields"), a chart often communicates faster than prose. v1 emits structured chart specs; v2's frontend renders them. By making the spec part of the agent contract, we lock in the data discipline now and decouple from rendering.

### 6.2 Contract

```python
class AxisSpec(BaseModel):
    label: str
    unit: str | None             # "GBP", "years", etc.
    scale: Literal["linear", "log"] = "linear"

class ChartSpec(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "pie", "table", "range"]
    title: str
    description: str             # one-sentence caption for accessibility
    x_axis: AxisSpec | None
    y_axis: AxisSpec | None
    data: list[dict]             # rows; column names match axis labels for bar/line/scatter
    source_citation_indices: list[int]  # indices into the parent SynthesizedResponse's citations
```

### 6.3 Where charts can appear

- `SynthesizedResponse.chart_specs: list[ChartSpec]` — Flow C only.
- `CoachOutput.chart_specs: list[ChartSpec]` — Flow B (small additions; Coach can add a chart when responding to a comparison question without going deep).

Flow A does NOT emit charts (it's the cheap fast path).

### 6.4 When to emit a chart

Encoded in prompt principles:

- Only when the data exists in `referenced_facts` or `referenced_findings` — never invent data for a chart.
- Only when the data has comparable quantitative structure (≥3 items being compared, OR a clear time series, OR a numeric range).
- Maximum 1 chart per response in v1. If the model wants more, it must pick the most informative one. (Limit can relax in v2.)
- A chart's `data` rows must trace back to citations via `source_citation_indices`. No uncited charts.

### 6.5 Validation

Pydantic validates structure. Critic adds two checks (encoded in prompt):
- `chart_uncited` — a chart has no `source_citation_indices` populated.
- `chart_data_invented` — chart contains numeric values that don't appear in any cited finding.

---

## 7. Pillar 4 — Flow C (Deliberation)

### 7.1 When Flow C fires

The Orchestrator routes to Flow C when ALL of:

- `intent_packet.budget_hint == "deep"` OR `intent_packet.turn_intent == "decide"`, AND
- user has ≥ 5 structured facts (otherwise Flow B), AND
- the user is NOT in onboarding.

```python
Flow = Literal["A", "B", "C", "onboarding"]

def decide(self, intent: IntentPacket, is_new_user: bool, fact_count: int) -> Flow:
    if is_new_user:
        return "onboarding"
    if intent.turn_intent == "vent" or intent.budget_hint == "quick":
        return "A"
    if (intent.budget_hint == "deep" or intent.turn_intent == "decide") and fact_count >= 5:
        return "C"
    return "B"
```

Orchestrator stays LLM-free and stateless. `fact_count` passed in from the pipeline.

### 7.2 Flow C graph

```
Understander
    │
    ▼
Orchestrator ──► Flow C
    │
    ├────────────┬────────────┐
    ▼            ▼
Researcher     Coach     (facts + hypotheses
    │            │         loaded once, shared)
    │            │
    └────────────┤
                 ▼
           Devil's Advocate
                 │
                 ▼
            Synthesizer
                 │
                 ▼
              Critic ──reject──► Synthesizer retry (max 2)
                 │
                 ▼
             Supervisor
                 │
                 ▼
             response
```

**Parallelism:** Researcher and Coach run concurrently. Researcher does not see Coach output; Coach does not see Researcher output. Both feed DA and Synthesizer.

**Retry semantics:** Critic rejection retries the Synthesizer only. Researcher and DA outputs are stable given inputs; re-running them wastes calls.

**Latency budget:** 25 s p95 end-to-end. If exceeded, log `fallback_reason="flow_c_timeout"` and return Coach output as fallback. Track Flow C p95 separately from Flow B so degradation is visible.

### 7.3 Critic on Flow C is stricter

Flow C adds four new failure modes:

- `unintegrated` — Synthesizer's `integrated_from` doesn't include all available sources.
- `uncontested` — DA disagreed but no tradeoff surfaced in the response.
- `chart_uncited` — chart present without `source_citation_indices`.
- `chart_data_invented` — chart contains values absent from cited findings.

Update Critic battery accordingly (5 new fixtures, 1 per failure mode + 1 clean Flow C reference response).

---

## 8. Pillar 5 — Devil's Advocate

**Model:** Coach-tier (strong). Haiku-tier produces generic "consider the risks" slop.

```python
class CounterPoint(BaseModel):
    point: str
    reasoning: str
    severity: Literal["low", "med", "high"]
    source_type: Literal["user_profile", "research", "general"]

class Risk(BaseModel):
    scenario: str
    likelihood: Literal["low", "med", "high"]
    impact: Literal["low", "med", "high"]

class DevilsAdvocateOutput(BaseModel):
    counter_points: list[CounterPoint]   # ≥2 unless agrees_with_coach
    blind_spots: list[str]
    risks: list[Risk]
    agrees_with_coach: bool
```

Prompt principles (in `config/prompts/devils_advocate.j2`):

- Ground counter-points in user specifics (`source_type="user_profile"` preferred over `general`).
- Honest agreement allowed and rewarded — `agrees_with_coach=true` with empty counter_points is a valid output.
- Severity is forced — prevents low-severity risk-list slop.

---

## 9. Pillar 6 — Synthesizer

**Model:** Coach-tier. This is the user-facing voice on Flow C.

```python
class SynthesizedResponse(BaseModel):
    response_text: str
    referenced_facts: list[str]
    referenced_hypotheses: list[UUID]
    referenced_findings: list[str]            # finding.claim strings
    citations: list[Citation]                 # all cited sources, deduped
    surfaced_tradeoffs: list[str]
    integrated_from: list[Literal["coach", "devils_advocate", "researcher"]]
    chart_specs: list[ChartSpec]              # 0 or 1 in v1
    proposed_challenge: Challenge | None      # field exists; v1 ignores it; v1.5 promotes it
    uncertainty_flags: list[str]
```

Prompt principles (in `config/prompts/synthesizer.j2`):

- Must integrate from all non-empty sources (`integrated_from` checked by Critic).
- Must surface ≥1 tradeoff when `devils_advocate_output.agrees_with_coach == false`.
- Reference budget: ≥2 user facts/hypotheses + ≥1 finding (when research non-empty) + ≥1 DA point (when DA disagrees).
- Inline citations in `response_text` use `[1]`, `[2]` style numerals; the numeral indexes into `citations`.
- Voice: one coach's integrated view, not a committee transcript.
- Chart emission only when §6.4 conditions hold.

Flow B unaffected — Coach is still the user-facing voice on Flow B (with optional small chart per §6.3).

---

## 10. Pillar 7 — Supervisor

### 10.1 Distinction from Critic

Critic = quality gate, post-Coach or post-Synthesizer. Supervisor = cross-cutting safety/correctness monitor, runs on EVERY flow as a final pre-response check.

### 10.2 Three checks only (v1)

1. **Fact contradiction.** Response asserts something contradicting a high-confidence structured fact.
2. **Off-topic drift.** Response substantively misses `intent_packet.specific_ask`.
3. **Safety triggers.** User message OR response contains crisis keywords/patterns (self-harm, suicide, abuse). On hit, override with a scripted safety message.

v1 does NOT attempt: hallucination detection beyond fact contradiction, medical/legal filtering, prompt injection detection. v2+.

### 10.3 Contract

**Model:** Haiku-tier. p50 < 500 ms.

```python
class SupervisorEvent(BaseModel):
    event_type: Literal["fact_contradiction", "off_topic", "unsafe"] | None
    severity: Literal["low", "med", "high"] | None
    details: str | None
    action: Literal["pass", "warn", "retry", "block"]
    scripted_override: str | None
```

### 10.4 Action semantics

- `pass` — response goes out unchanged.
- `warn` — response goes out with appended caveat.
- `retry` — force Coach/Synthesizer retry once; if retry also trips, fall through to `warn`.
- `block` — `scripted_override` returned to user verbatim. Used ONLY for `unsafe`.

### 10.5 Red-team fixtures

`tests/quality/supervisor_redteam/` — 20 fixtures:
- 5 fact contradictions
- 5 off-topic drift
- 5 unsafe triggers (varying severity)
- 5 benign (regression guard against false positives)

Bar: ≥8/15 intended triggers caught, 0/5 false positives.

---

## 11. Pillar 8 — LangGraph migration

Migrate last, when graph is fully populated. With Flow C plus Supervisor plus retries, hand-rolled async becomes hard to audit; LangGraph's explicit state machine pays for itself.

**Hard rule:** all v0.5 and v1 tests pass unmodified after migration. If any test needs adaptation, stop and diagnose.

Don't use LangGraph checkpointing (we have episodic repo), don't use built-in HITL (we have clarifying-question loop), don't add tool-calling routing (LLM calls stay through `LLMClient`).

---

## 12. Schema changes (migration 004)

```sql
-- 004_v1_agents.sql

ALTER TABLE turns
    ADD COLUMN IF NOT EXISTS research_brief_id UUID,
    ADD COLUMN IF NOT EXISTS synthesizer_output JSONB,
    ADD COLUMN IF NOT EXISTS devils_advocate_output JSONB,
    ADD COLUMN IF NOT EXISTS chart_specs JSONB DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS research_briefs (
    brief_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id UUID REFERENCES turns(turn_id),
    question TEXT NOT NULL,
    findings JSONB NOT NULL,
    caveats JSONB DEFAULT '[]'::jsonb,
    next_questions JSONB DEFAULT '[]'::jsonb,
    used_kb_files JSONB DEFAULT '[]'::jsonb,
    web_searches_run JSONB DEFAULT '[]'::jsonb,
    web_sources_consulted JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_briefs_turn ON research_briefs(turn_id);

CREATE TABLE IF NOT EXISTS supervisor_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id UUID REFERENCES turns(turn_id),
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    details JSONB,
    action_taken TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_supervisor_events_turn ON supervisor_events(turn_id);
CREATE INDEX IF NOT EXISTS idx_supervisor_events_type
    ON supervisor_events(event_type, created_at DESC);
```

Additive only. Existing rows unaffected.

---

## 13. World KB v0 (supplemental)

Same 15-career YAML structure as planned in the prior v1 draft. Now positioned as **supplement** to web search, not the primary source. Still useful for stable structural data (entry paths, typical exits, contra-indicators) where web results are noisier than a hand-curated reference.

Files: `kb/careers/*.yaml` (15 entries) + `kb/pay_bands_uk.yaml` + `kb/exits.yaml`.

Every numeric field has a `source_notes` sibling. Source notes are non-negotiable; without them the KB has no advantage over a generic LLM hallucination.

`WorldKBRepo` exposes `get_career(name)`, `search_by_tags(tags)`, `list_all_careers()`. Loaded at process start.

15-career seed set: quant_finance, investment_banking, management_consulting, software_engineering, machine_learning_research, product_management, data_science, academia_stem_phd_track, medicine_uk, law_uk, civil_service_uk, teaching, design_ux, journalism, entrepreneurship_early.

---

## 14. Quality bars

Beyond v0.5 bars (which must continue to hold):

1. **Web search adapter** — Tavily integration test passes against live API; mock test suite covers timeout, empty results, rate-limit behavior.
2. **Researcher quality** — 20 fixture battery on deep-decision questions:
   - ≥18/20 produce ≥2 findings.
   - ≥18/20 have every finding citable to a fetched URL or KB file with `source_notes`.
   - 0/20 contain numeric claims without web or sourced-KB citation.
3. **Flow C end-to-end** — hand-crafted fixture (PhD vs quant) produces a response with ≥2 user-fact references, ≥1 finding-derived claim, ≥1 tradeoff word, ≥1 inline citation.
4. **Flow C Critic pass rate** — ≥85% on 20-fixture deep-decision battery.
5. **Chart emission** — ≥5 of 20 deep-decision fixtures produce a chart; 0 of 20 produce an uncited or invented-data chart.
6. **Supervisor red-team** — ≥8/15 intended triggers, 0/5 false positives.
7. **Latency** — Flow C p95 ≤ 25 s; Flow A/B p95 unchanged from v0.5.
8. **LangGraph migration** — all v0.5 and v1 tests pass unmodified.
9. **Cost ceiling** — full Flow C eval (20 fixtures) costs < £25 in API + search calls combined. If higher, audit before re-running.

---

## 15. Definition of Done

v1 is done when ALL of these pass:

1. Migration 004 applies cleanly on top of 001-003.
2. WebSearchClient (Tavily impl) passes integration test against live API.
3. WorldKBRepo loads all 15 seed careers.
4. Researcher quality battery passes (§14.2 thresholds).
5. Devil's Advocate produces valid output with ≥2 counter-points on a hand-crafted Coach output, AND honest-agreement path produces valid output with `agrees_with_coach=true` and minimal counter-points.
6. Synthesizer produces valid SynthesizedResponse including a chart on at least one fixture.
7. Flow C end-to-end integration test passes.
8. Flow C Critic battery (20 fixtures + 5 new failure-mode fixtures) ≥ 85%.
9. Supervisor red-team passes per §10.5 thresholds.
10. LangGraph migration complete; all v0.5 and v1 tests pass unmodified.
11. Full v0.5 DoD still holds (re-run on clean DB).
12. Two weeks of demo-script usage produces Flow C p95 ≤ 25 s AND <5% Supervisor false-positive rate on real turns.

When 12/12 hit: cut a v1 tag. Then plan v1.5.

---

## 16. Anti-goals for v1

Do not:

- Build the Initiator agent or any check-in mechanism (v1.5).
- Promote `proposed_challenge` past its current stub (v1.5).
- Build a frontend or render charts visually (v2).
- Add Auth (v2).
- Add Evaluator or God-agent (v2+).
- Use LangGraph checkpointing in place of episodic repo.
- Expand the KB past 15 careers.
- Use web search inside Coach, DA, or Synthesizer directly — only the Researcher hits the web.
- Add LLM-based Orchestrator routing — Python rules stay.
- Pre-optimize Flow C latency below the 25 s budget.

Every hour on the above delays the moment users get a research-grounded, citation-honest, tradeoff-aware answer.

---

## 17. Preview — what v1.5 adds

(For context only; do not build during v1.)

- **Challenges as first-class.** `challenges` table with state machine: assigned → in_progress → completed/abandoned. The `proposed_challenge` field gets persisted, surfaced back, and tracked.
- **Initiator agent.** Given user state and elapsed time since last interaction, decides whether and what to check in about. Outputs a check-in turn with `flow_used="check_in"`.
- **Check-in admin endpoint.** `POST /admin/check_in/{user_id}` runs Initiator and persists a check-in turn. Real scheduling stays out — that's v2 with n8n/cron.
- **Profiler updates.** When user replies to a challenge or check-in, Profiler treats the response as high-weight evidence (forced-choice signal, completion outcomes).
- **Memory tier addition.** A small `engagement_state` micro-store: last_check_in_at, active_challenges, recent_check_in_topics. Prevents the Initiator from being repetitive.

v1.5 deliberately builds on v1's surfaces — no new major infra. The hard part is the temporal logic, not the agents themselves.
