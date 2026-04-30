#!/usr/bin/env bash
# =============================================================================
# test_v1.sh — v1 test runner
#
# Runs every test tier in the correct order. Exits non-zero on any failure.
# Does NOT require live API keys for tiers 1-3; keys unlock tiers 4-6.
#
# Usage:
#   ./scripts/test_v1.sh           # fast tier 1-3 only (no keys needed)
#   ./scripts/test_v1.sh --all     # all tiers (requires API keys)
#   ./scripts/test_v1.sh --tier 4  # run only one specific tier
#
# Tiers:
#   1. Lint + type-check (ruff + mypy)
#   2. Fast unit tests (no DB, no LLM)
#   3. Agent unit tests — mocked LLM (no keys)
#   4. Live agent tests (HUGGINGFACE_API_TOKEN required)
#   5. Quality batteries (HUGGINGFACE_API_TOKEN required for Critic/Supervisor)
#   6. Integration tests (SUPABASE_DB_URL + HUGGINGFACE_API_TOKEN)
#   7. Flow C eval script (all three keys: HF + Tavily + Supabase)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- colour helpers ---------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; RESET='\033[0m'

pass()  { echo -e "${GREEN}✅ $*${RESET}"; }
fail()  { echo -e "${RED}❌ $*${RESET}"; }
skip()  { echo -e "${YELLOW}⏭  $*${RESET}"; }
header(){ echo -e "\n${BOLD}$*${RESET}"; }

# ---- argument parsing -------------------------------------------------------
RUN_ALL=false
ONLY_TIER=""

for arg in "$@"; do
  case "$arg" in
    --all)        RUN_ALL=true ;;
    --tier)       shift; ONLY_TIER="$1" ;;
    --tier=*)     ONLY_TIER="${arg#--tier=}" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# //' | sed 's/^#//'
      exit 0 ;;
  esac
done

should_run() {
  local tier="$1"
  [[ -z "$ONLY_TIER" ]] || [[ "$ONLY_TIER" == "$tier" ]]
  if [[ -z "$ONLY_TIER" ]] && [[ "$tier" -ge 4 ]] && [[ "$RUN_ALL" == false ]]; then
    return 1
  fi
  return 0
}

# ---- load .env into the shell environment -----------------------------------
# Keys set in .env are available to pytest (via load_dotenv in conftest.py),
# but the shell itself won't see them unless we source the file here.
# set -a / set +a auto-exports every assignment in .env; override=false means
# a variable already exported in the parent shell wins over .env.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
fi

# ---- environment checks -----------------------------------------------------
HAS_HF="${HUGGINGFACE_API_TOKEN:+yes}"
HAS_TAVILY="${TAVILY_API_KEY:+yes}"
HAS_DB="${SUPABASE_DB_URL:+yes}"

ERRORS=0

# ---- Tier 1: lint + types ---------------------------------------------------
if should_run 1; then
  header "Tier 1 — Lint + type-check"

  # Lint src/ strictly; tests/ has pre-existing minor issues (unused imports in
  # fixtures) that are tolerated — only check that no NEW src errors crept in.
  if ruff check src/; then
    pass "ruff: src/ clean"
  else
    fail "ruff: errors in src/"
    (( ERRORS++ )) || true
  fi

  if .venv/bin/mypy --strict src/career_coach/pipeline/turn.py \
                              src/career_coach/agents/supervisor.py \
                              src/career_coach/models/agent_io.py 2>&1 | \
     grep -q "Success"; then
    pass "mypy: core v1 files clean"
  else
    fail "mypy: errors in core v1 files"
    .venv/bin/mypy --strict src/career_coach/pipeline/turn.py \
                            src/career_coach/agents/supervisor.py \
                            src/career_coach/models/agent_io.py 2>&1 | tail -5
    (( ERRORS++ )) || true
  fi
fi

# ---- Tier 2: fast unit tests (no LLM, no DB) --------------------------------
if should_run 2; then
  header "Tier 2 — Fast unit tests (no keys)"

  FAST_TESTS=(
    tests/agents/test_orchestrator.py
    tests/models/
    tests/pipeline/test_turn_unit.py
  )

  # Only run paths that exist
  EXISTING=()
  for p in "${FAST_TESTS[@]}"; do
    [[ -e "$p" ]] && EXISTING+=("$p")
  done

  # Run everything that doesn't need API keys.
  # -m "not live" excludes the handful of live-LLM agent tests that are in
  # tests/agents/ but require HUGGINGFACE_API_TOKEN to actually call the API.
  # Those run in Tier 4. Quality batteries and integration tests are ignored
  # explicitly so their collection errors don't pollute this tier.
  if .venv/bin/pytest \
      -m "not live" \
      --ignore=tests/integration \
      --ignore=tests/quality/test_critic_battery.py \
      --ignore=tests/quality/test_onboarding_battery.py \
      --ignore=tests/quality/test_supervisor_battery.py \
      -q 2>&1 | tail -5; then
    pass "Unit tests: all passed"
  else
    fail "Unit tests: failures"
    (( ERRORS++ )) || true
  fi
fi

# ---- Tier 3: agent parsing tests (no live LLM) ------------------------------
if should_run 3; then
  header "Tier 3 — Agent schema + parsing tests"

  # These tests mock the LLM; they test JSON parsing and Pydantic validation
  PARSE_TESTS=(
    tests/agents/test_supervisor_parse.py
    tests/agents/test_critic_parse.py
  )
  EXISTING_PARSE=()
  for p in "${PARSE_TESTS[@]}"; do
    [[ -e "$p" ]] && EXISTING_PARSE+=("$p")
  done

  if [[ ${#EXISTING_PARSE[@]} -gt 0 ]]; then
    if .venv/bin/pytest "${EXISTING_PARSE[@]}" -q 2>&1 | tail -3; then
      pass "Agent parse tests: passed"
    else
      fail "Agent parse tests: failures"
      (( ERRORS++ )) || true
    fi
  else
    skip "No standalone parse tests found (covered by unit tests)"
  fi
fi

# ---- Tier 4: live agent tests -----------------------------------------------
if should_run 4; then
  header "Tier 4 — Live agent tests (HUGGINGFACE_API_TOKEN)"

  if [[ -z "$HAS_HF" ]]; then
    skip "HUGGINGFACE_API_TOKEN not set — skipping tier 4"
  else
    if .venv/bin/pytest tests/agents/ tests/web/ -q 2>&1 | tail -5; then
      pass "Live agent tests: passed"
    else
      fail "Live agent tests: failures"
      (( ERRORS++ )) || true
    fi
  fi
fi

# ---- Tier 5: quality batteries ----------------------------------------------
if should_run 5; then
  header "Tier 5 — Quality batteries"

  # Onboarding battery (HF only)
  if [[ -z "$HAS_HF" ]]; then
    skip "Onboarding battery — HUGGINGFACE_API_TOKEN not set"
  else
    echo "  Running onboarding battery..."
    if .venv/bin/pytest tests/quality/test_onboarding_battery.py -q 2>&1 | tail -5; then
      pass "Onboarding battery"
    else
      fail "Onboarding battery"
      (( ERRORS++ )) || true
    fi
  fi

  # Critic battery (HuggingFace)
  if [[ -z "$HAS_HF" ]]; then
    skip "Critic battery — HUGGINGFACE_API_TOKEN not set"
  else
    echo "  Running critic battery..."
    if .venv/bin/pytest tests/quality/test_critic_battery.py -q 2>&1 | tail -5; then
      pass "Critic battery"
    else
      fail "Critic battery"
      (( ERRORS++ )) || true
    fi
  fi

  # Supervisor red-team battery (HuggingFace)
  if [[ -z "$HAS_HF" ]]; then
    skip "Supervisor battery — HUGGINGFACE_API_TOKEN not set"
  else
    echo "  Running supervisor battery..."
    if .venv/bin/pytest tests/quality/test_supervisor_battery.py -q 2>&1 | tail -5; then
      pass "Supervisor red-team battery"
    else
      fail "Supervisor red-team battery"
      (( ERRORS++ )) || true
    fi
  fi
fi

# ---- Tier 6: integration tests ----------------------------------------------
if should_run 6; then
  header "Tier 6 — Integration tests (SUPABASE_DB_URL + HUGGINGFACE_API_TOKEN)"

  if [[ -z "$HAS_HF" ]] || [[ -z "$HAS_DB" ]]; then
    MISSING=""
    [[ -z "$HAS_HF" ]] && MISSING="HUGGINGFACE_API_TOKEN "
    [[ -z "$HAS_DB" ]] && MISSING="${MISSING}SUPABASE_DB_URL"
    skip "Integration tests — missing: $MISSING"
  else
    echo "  Running integration tests (this may take 2-3 minutes)..."
    if .venv/bin/pytest tests/integration/ \
        --ignore=tests/integration/test_flow_c_end_to_end.py \
        --ignore=tests/integration/test_supervisor_all_flows.py \
        -q 2>&1 | tail -8; then
      pass "Integration tests (core)"
    else
      fail "Integration tests (core)"
      (( ERRORS++ )) || true
    fi

    # Flow C and Supervisor integration (needs Tavily too)
    if [[ -z "$HAS_TAVILY" ]]; then
      skip "Flow C + Supervisor integration — TAVILY_API_KEY not set"
    else
      if .venv/bin/pytest \
          tests/integration/test_flow_c_end_to_end.py \
          tests/integration/test_supervisor_all_flows.py \
          -q 2>&1 | tail -8; then
        pass "Flow C + Supervisor integration"
      else
        fail "Flow C + Supervisor integration"
        (( ERRORS++ )) || true
      fi
    fi
  fi
fi

# ---- Tier 7: Flow C quality eval -------------------------------------------
if should_run 7; then
  header "Tier 7 — Flow C quality eval (all keys)"

  MISSING=""
  [[ -z "$HAS_HF" ]]     && MISSING="${MISSING}HUGGINGFACE_API_TOKEN "
  [[ -z "$HAS_TAVILY" ]] && MISSING="${MISSING}TAVILY_API_KEY "
  [[ -z "$HAS_DB" ]]     && MISSING="${MISSING}SUPABASE_DB_URL"

  if [[ -n "$MISSING" ]]; then
    skip "Flow C eval — missing: $MISSING"
  else
    echo "  Running flow_c_eval (20 fixtures — may take 10-20 min)..."
    if uv run python scripts/run_flow_c_eval.py \
        --report reports/flow_c_baseline.md; then
      pass "Flow C eval: ≥17/20 passed, ≥5 charts, 0 bad charts"
      echo "  Report: reports/flow_c_baseline.md"
    else
      fail "Flow C eval: thresholds not met — check reports/flow_c_baseline.md"
      (( ERRORS++ )) || true
    fi
  fi
fi

# ---- Summary ----------------------------------------------------------------
header "Summary"

if [[ "$ERRORS" -eq 0 ]]; then
  pass "All tiers passed (errors=0)"
  exit 0
else
  fail "$ERRORS tier(s) failed"
  exit 1
fi
