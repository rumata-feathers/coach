#!/usr/bin/env python3
"""Morning triage report generator for the career coaching friend test.

Generates a structured Markdown report from the last N hours of DB activity.

Usage
-----
    uv run python scripts/daily_report.py
    uv run python scripts/daily_report.py --since 48h
    uv run python scripts/daily_report.py --since "2026-05-03T00:00:00"
    uv run python scripts/daily_report.py --user-id <uuid>
    uv run python scripts/daily_report.py --output reports/daily_custom.md
    uv run python scripts/daily_report.py --dsn "postgresql://..."

If --dsn is omitted the script falls back to SUPABASE_DB_URL / .env.
Output defaults to reports/daily_YYYY-MM-DD.md (today's date, UTC).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _parse_since(raw: str) -> datetime:
    """Parse "24h", "48h", or ISO-8601 string → timezone-aware UTC datetime."""
    raw = raw.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)h", raw)
    if m:
        return datetime.now(UTC) - timedelta(hours=float(m.group(1)))
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _parse_args() -> tuple[datetime, UUID | None, str | None, Path]:
    """Return (since_dt, user_id | None, dsn | None, output_path)."""
    argv = sys.argv[1:]
    since_raw = "24h"
    user_id: UUID | None = None
    dsn: str | None = None
    output: str | None = None

    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--since", "-s"):
            i += 1
            since_raw = argv[i] if i < len(argv) else "24h"
        elif a.startswith("--since="):
            since_raw = a[len("--since="):]
        elif a in ("--user-id", "-u"):
            i += 1
            user_id = UUID(argv[i]) if i < len(argv) else None
        elif a.startswith("--user-id="):
            user_id = UUID(a[len("--user-id="):])
        elif a in ("--dsn", "-d"):
            i += 1
            dsn = argv[i] if i < len(argv) else None
        elif a.startswith("--dsn="):
            dsn = a[len("--dsn="):]
        elif a in ("--output", "-o"):
            i += 1
            output = argv[i] if i < len(argv) else None
        elif a.startswith("--output="):
            output = a[len("--output="):]
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        i += 1

    since = _parse_since(since_raw)
    if output is None:
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        output = f"reports/daily_{date_str}.md"

    return since, user_id, dsn, Path(output)


def _resolve_dsn(dsn_override: str | None) -> str:
    if dsn_override:
        return dsn_override
    try:
        from career_coach.config import get_settings
        return get_settings().supabase_db_url
    except Exception:
        return os.environ.get("SUPABASE_DB_URL", "")


# ---------------------------------------------------------------------------
# asyncpg pool helpers
# ---------------------------------------------------------------------------

_LOCAL_HINTS = ("localhost", "127.0.0.1", "host.docker.internal")


def _needs_ssl(dsn: str) -> bool:
    return not any(h in dsn for h in _LOCAL_HINTS)


def _uf(user_id: UUID | None, alias: str = "t", base_param: int = 1) -> tuple[str, list[Any]]:
    """Return (sql_fragment, params) for optional user_id filter."""
    if user_id is None:
        return "", []
    return f" AND {alias}.user_id = ${base_param + 1}", [user_id]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trunc(text: str | None, n: int = 250) -> str:
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


def _fmt_ts(ts: Any) -> str:
    if ts is None:
        return "—"
    return str(ts)[:16].replace("T", " ")


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{100 * num / denom:.1f}%"


# ---------------------------------------------------------------------------
# Section 1: Summary stats
# ---------------------------------------------------------------------------


async def _fetch_summary(conn: Any, since: datetime, user_id: UUID | None) -> dict[str, Any]:
    uf, up = _uf(user_id, alias="t")

    # Scalar stats
    row = await conn.fetchrow(
        f"""
        SELECT
          COUNT(DISTINCT t.user_id)                                          AS active_users,
          COUNT(DISTINCT t.session_id)                                       AS sessions,
          COUNT(*)                                                            AS turns,
          COUNT(*) FILTER (WHERE t.flow_used = 'B')                         AS flow_b_turns,
          COUNT(*) FILTER (WHERE
              t.flow_used = 'B'
              AND t.critic_verdicts IS NOT NULL
              AND jsonb_array_length(t.critic_verdicts) > 0
              AND t.critic_verdicts->-1->>'verdict' = 'reject')              AS critic_rejects,
          COUNT(*) FILTER (WHERE t.flow_used = 'onboarding')                AS onboarding_turns
        FROM turns t
        WHERE t.created_at >= $1{uf}
        """,
        since, *up,
    )

    # Total tokens: aggregate from agent_calls (authoritative source — turns.tokens_used
    # was historically NULL; agent_calls.user_id is populated since migration 006).
    uf_ac, up_ac = _uf(user_id, alias="ac", base_param=1)
    token_row = await conn.fetchrow(
        f"""
        SELECT
          COALESCE(SUM(ac.tokens_in),  0) AS total_in,
          COALESCE(SUM(ac.tokens_out), 0) AS total_out
        FROM agent_calls ac
        WHERE ac.created_at >= $1{uf_ac}
        """,
        since, *up_ac,
    )
    total_tokens = int(
        (token_row["total_in"] or 0) + (token_row["total_out"] or 0)
    )

    # Agent calls
    uf2, up2 = _uf(user_id, alias="t", base_param=1)
    ac_row = await conn.fetchrow(
        f"""
        SELECT
          COUNT(*)                                              AS total_calls,
          COUNT(*) FILTER (WHERE ac.fallback_reason IS NOT NULL) AS fallback_calls
        FROM agent_calls ac
        JOIN turns t ON t.turn_id = ac.turn_id
        WHERE t.created_at >= $1{uf2}
        """,
        since, *up2,
    )

    # Deployment versions
    ver_rows = await conn.fetch(
        f"SELECT DISTINCT t.deployment_version FROM turns t WHERE t.created_at >= $1{uf}",
        since, *up,
    )
    versions = [r["deployment_version"] or "unknown" for r in ver_rows]

    return {
        "active_users":    int(row["active_users"]),
        "sessions":        int(row["sessions"]),
        "turns":           int(row["turns"]),
        "flow_b_turns":    int(row["flow_b_turns"]),
        "critic_rejects":  int(row["critic_rejects"]),
        "onboarding_turns": int(row["onboarding_turns"]),
        "total_tokens":    total_tokens,
        "total_calls":     int(ac_row["total_calls"]),
        "fallback_calls":  int(ac_row["fallback_calls"]),
        "versions":        sorted(set(versions)),
    }


# ---------------------------------------------------------------------------
# Section 2: Per-user breakdown
# ---------------------------------------------------------------------------


async def _fetch_user_breakdown(
    conn: Any, since: datetime, user_id: UUID | None
) -> list[dict[str, Any]]:
    uf, up = _uf(user_id, alias="t")

    rows = await conn.fetch(
        f"""
        WITH user_turns AS (
            SELECT t.*, u.display_name
            FROM turns t
            JOIN users u ON u.user_id = t.user_id
            WHERE t.created_at >= $1{uf}
        ),
        per_user AS (
            SELECT
                user_id,
                MAX(display_name)                                                AS display_name,
                COUNT(DISTINCT session_id)                                       AS session_count,
                COUNT(*)                                                          AS turn_count,
                MIN(created_at)                                                   AS first_turn,
                MAX(created_at)                                                   AS last_turn,
                COUNT(*) FILTER (WHERE flow_used = 'B')                          AS flow_b,
                COUNT(*) FILTER (WHERE
                    flow_used = 'B'
                    AND critic_verdicts IS NOT NULL
                    AND jsonb_array_length(critic_verdicts) > 0
                    AND critic_verdicts->-1->>'verdict' = 'reject')               AS critic_rejects,
                COUNT(*) FILTER (WHERE flow_used = 'onboarding')                 AS onboarding_count
            FROM user_turns
            GROUP BY user_id
        ),
        fallbacks AS (
            SELECT t.user_id,
                   COUNT(*) FILTER (WHERE ac.fallback_reason IS NOT NULL)        AS fallback_count,
                   COUNT(*)                                                        AS ac_total
            FROM agent_calls ac
            JOIN turns t ON t.turn_id = ac.turn_id
            WHERE t.created_at >= $1{uf}
            GROUP BY t.user_id
        ),
        token_totals AS (
            -- Aggregate from agent_calls.user_id (populated since migration 006).
            -- This covers pre-persist calls whose turn_id is NULL, giving accurate
            -- per-user token counts across the full pipeline.
            SELECT ac.user_id,
                   COALESCE(SUM(ac.tokens_in),  0) +
                   COALESCE(SUM(ac.tokens_out), 0)                               AS total_tokens
            FROM agent_calls ac
            WHERE ac.created_at >= $1
              AND ac.user_id IS NOT NULL
            GROUP BY ac.user_id
        ),
        fact_delta AS (
            SELECT user_id, COUNT(*) AS fact_count
            FROM structured_facts
            WHERE created_at >= $1
            GROUP BY user_id
        ),
        evidence_delta AS (
            SELECT h.user_id, COUNT(*) AS ev_count
            FROM hypothesis_evidence he
            JOIN hypotheses h ON h.hypothesis_id = he.hypothesis_id
            WHERE he.created_at >= $1
            GROUP BY h.user_id
        )
        SELECT
            pu.*,
            COALESCE(fb.fallback_count, 0)   AS fallback_count,
            COALESCE(fb.ac_total, 0)          AS ac_total,
            COALESCE(tt.total_tokens, 0)      AS total_tokens,
            COALESCE(fd.fact_count, 0)        AS fact_delta,
            COALESCE(ed.ev_count, 0)          AS evidence_delta
        FROM per_user pu
        LEFT JOIN fallbacks      fb ON fb.user_id = pu.user_id
        LEFT JOIN token_totals   tt ON tt.user_id = pu.user_id
        LEFT JOIN fact_delta     fd ON fd.user_id = pu.user_id
        LEFT JOIN evidence_delta ed ON ed.user_id = pu.user_id
        ORDER BY pu.turn_count DESC
        """,
        since, *up,
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Section 3: Quality concerns
# ---------------------------------------------------------------------------


async def _fetch_quality(
    conn: Any, since: datetime, user_id: UUID | None
) -> dict[str, list[dict[str, Any]]]:
    uf, up = _uf(user_id, alias="t")

    # 1. Critic exhausted retries (≥2 verdicts, last = reject)
    critic_rows = await conn.fetch(
        f"""
        SELECT t.turn_id, u.display_name, t.user_id,
               t.user_message, t.assistant_message,
               jsonb_array_length(t.critic_verdicts) AS retry_count
        FROM turns t
        JOIN users u ON u.user_id = t.user_id
        WHERE t.created_at >= $1{uf}
          AND t.critic_verdicts IS NOT NULL
          AND jsonb_array_length(t.critic_verdicts) >= 2
          AND t.critic_verdicts->-1->>'verdict' = 'reject'
        ORDER BY t.created_at
        """,
        since, *up,
    )

    # 2. Agent fallbacks
    fallback_rows = await conn.fetch(
        f"""
        SELECT DISTINCT ON (t.turn_id)
               t.turn_id, u.display_name, t.user_id,
               t.user_message, t.assistant_message,
               string_agg(ac.agent_name || ': ' || ac.fallback_reason, '; ')
                   OVER (PARTITION BY t.turn_id) AS fallback_details
        FROM agent_calls ac
        JOIN turns t ON t.turn_id = ac.turn_id
        JOIN users u ON u.user_id = t.user_id
        WHERE t.created_at >= $1{uf}
          AND ac.fallback_reason IS NOT NULL
        ORDER BY t.turn_id, t.created_at
        """,
        since, *up,
    )

    # 3. Suspiciously short/long responses (exclude clarification turns)
    length_rows = await conn.fetch(
        f"""
        SELECT t.turn_id, u.display_name, t.user_id,
               t.user_message, t.assistant_message,
               LENGTH(t.assistant_message) AS msg_len
        FROM turns t
        JOIN users u ON u.user_id = t.user_id
        WHERE t.created_at >= $1{uf}
          AND t.assistant_message IS NOT NULL
          AND t.flow_used != 'clarification'
          AND (LENGTH(t.assistant_message) < 100 OR LENGTH(t.assistant_message) > 2000)
        ORDER BY t.created_at
        """,
        since, *up,
    )

    # 4. Late clarifications (clarification turn after the 3rd turn in a session)
    late_clarif_rows = await conn.fetch(
        f"""
        SELECT t.turn_id, u.display_name, t.user_id,
               t.user_message, t.assistant_message, t.turn_index
        FROM turns t
        JOIN users u ON u.user_id = t.user_id
        WHERE t.created_at >= $1{uf}
          AND t.flow_used = 'clarification'
          AND t.turn_index >= 3
        ORDER BY t.created_at
        """,
        since, *up,
    )

    # 5. Retried turns (user sent "[retried]")
    retry_rows = await conn.fetch(
        f"""
        SELECT t.turn_id, u.display_name, t.user_id,
               t.user_message, t.assistant_message
        FROM turns t
        JOIN users u ON u.user_id = t.user_id
        WHERE t.created_at >= $1{uf}
          AND t.user_message LIKE '%[retried]%'
        ORDER BY t.created_at
        """,
        since, *up,
    )

    return {
        "critic_exhausted": [dict(r) for r in critic_rows],
        "fallbacks":         [dict(r) for r in fallback_rows],
        "length_anomalies":  [dict(r) for r in length_rows],
        "late_clarifs":      [dict(r) for r in late_clarif_rows],
        "user_retries":      [dict(r) for r in retry_rows],
    }


# ---------------------------------------------------------------------------
# Section 4: New extractions
# ---------------------------------------------------------------------------


async def _fetch_extractions(
    conn: Any, since: datetime, user_id: UUID | None
) -> dict[str, dict[str, Any]]:
    uf_sf, up_sf = _uf(user_id, alias="sf")
    uf_h, up_h = _uf(user_id, alias="h")

    fact_rows = await conn.fetch(
        f"""
        SELECT sf.user_id, u.display_name, sf.key,
               sf.value::text AS value_json,
               sf.source, sf.confidence
        FROM structured_facts sf
        JOIN users u ON u.user_id = sf.user_id
        WHERE sf.created_at >= $1{uf_sf}
        ORDER BY u.display_name, sf.key
        """,
        since, *up_sf,
    )

    hyp_rows = await conn.fetch(
        f"""
        SELECT h.user_id, u.display_name, h.hypothesis_id, h.statement,
               h.confidence, h.status,
               (SELECT COUNT(*) FROM hypothesis_evidence he
                WHERE he.hypothesis_id = h.hypothesis_id) AS evidence_count
        FROM hypotheses h
        JOIN users u ON u.user_id = h.user_id
        WHERE h.created_at >= $1{uf_h}
        ORDER BY u.display_name, h.confidence DESC
        """,
        since, *up_h,
    )

    # Group by user
    result: dict[str, dict[str, Any]] = {}
    for r in fact_rows:
        uid = str(r["user_id"])
        result.setdefault(uid, {"display_name": r["display_name"], "facts": [], "hypotheses": []})
        try:
            val = json.loads(r["value_json"])
            val_str = str(val).strip('"')
        except Exception:
            val_str = str(r["value_json"])
        result[uid]["facts"].append(
            f"{r['key']}={val_str} (source={r['source'] or '?'}, confidence={r['confidence']:.2f})"
        )
    for r in hyp_rows:
        uid = str(r["user_id"])
        result.setdefault(uid, {"display_name": r["display_name"], "facts": [], "hypotheses": []})
        result[uid]["hypotheses"].append(
            f"{r['statement']} "
            f"(confidence={r['confidence']:.2f}, evidence_count={r['evidence_count']}, "
            f"status={r['status']})"
        )

    return result


# ---------------------------------------------------------------------------
# Section 5: Full transcripts
# ---------------------------------------------------------------------------


async def _fetch_transcripts(
    conn: Any, since: datetime, user_id: UUID | None
) -> list[dict[str, Any]]:
    """Return all sessions (with turns) that had activity in the window."""
    uf, up = _uf(user_id, alias="t")

    # All sessions that have at least one turn in the window
    session_rows = await conn.fetch(
        f"""
        SELECT DISTINCT s.session_id, s.user_id, u.display_name,
                        s.started_at, s.session_theory
        FROM sessions s
        JOIN turns t ON t.session_id = s.session_id
        JOIN users u ON u.user_id = s.user_id
        WHERE t.created_at >= $1{uf}
        ORDER BY s.user_id, s.started_at
        """,
        since, *up,
    )

    if not session_rows:
        return []

    session_ids = [r["session_id"] for r in session_rows]

    turn_rows = await conn.fetch(
        """
        SELECT t.turn_id, t.session_id, t.turn_index, t.flow_used,
               t.user_message, t.assistant_message, t.created_at,
               t.deployment_version
        FROM turns t
        WHERE t.session_id = ANY($1::uuid[])
        ORDER BY t.session_id, t.turn_index
        """,
        session_ids,
    ) if session_ids else []

    # Group turns by session
    turns_by_session: dict[str, list[dict[str, Any]]] = {}
    for tr in turn_rows:
        sid = str(tr["session_id"])
        turns_by_session.setdefault(sid, []).append(dict(tr))

    sessions = []
    for sr in session_rows:
        sid = str(sr["session_id"])
        sessions.append({
            "session_id":    sid,
            "user_id":       str(sr["user_id"]),
            "display_name":  sr["display_name"],
            "started_at":    sr["started_at"],
            "session_theory": sr["session_theory"],
            "turns":         turns_by_session.get(sid, []),
        })

    return sessions


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_report(
    since: datetime,
    now: datetime,
    summary: dict[str, Any],
    users: list[dict[str, Any]],
    quality: dict[str, list[dict[str, Any]]],
    extractions: dict[str, dict[str, Any]],
    transcripts: list[dict[str, Any]],
    user_filter: UUID | None,
) -> str:
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str) -> None:
        lines.append(text)

    # ── Title ──────────────────────────────────────────────────────────────
    filter_note = f" (user `{str(user_filter)[:8]}…`)" if user_filter else ""
    h(1, f"Daily Report — {now.strftime('%Y-%m-%d %H:%M UTC')}{filter_note}")

    # ── Section 1: Summary ─────────────────────────────────────────────────
    h(2, "1 — Summary")

    p(f"**Window:** {since.strftime('%Y-%m-%d %H:%M UTC')} → {now.strftime('%Y-%m-%d %H:%M UTC')}")
    p("")

    s = summary
    p("| Metric | Value |")
    p("|--------|-------|")
    p(f"| Active users | {s['active_users']} |")
    p(f"| Sessions started | {s['sessions']} |")
    p(f"| Total turns | {s['turns']} |")
    p(f"| Total agent calls | {s['total_calls']} |")
    p(f"| Total tokens | {s['total_tokens']:,} |")
    p(f"| Critic rejection rate | {_pct(s['critic_rejects'], s['flow_b_turns'])} ({s['critic_rejects']}/{s['flow_b_turns']} flow-B turns) |")
    p(f"| Agent fallback rate | {_pct(s['fallback_calls'], s['total_calls'])} ({s['fallback_calls']}/{s['total_calls']} calls) |")
    p(f"| Onboarding share | {_pct(s['onboarding_turns'], s['turns'])} ({s['onboarding_turns']}/{s['turns']} turns) |")

    if len(s["versions"]) > 1:
        p(f"\n> ⚠️ **Multiple deployment versions** in this window — transcripts may span releases: {', '.join(s['versions'])}")
    else:
        v = s["versions"][0] if s["versions"] else "none"
        p(f"\nDeployment version: `{v}`")

    # ── Section 2: Per-user breakdown ──────────────────────────────────────
    h(2, "2 — Per-user breakdown")

    if not users:
        p("_No active users in this window._")
    else:
        p("| User | ID | Sessions | Turns | First turn | Last turn | Tokens | Critic rejects | Fallbacks | Onboarding turns | Fact Δ | Evidence Δ |")
        p("|------|----|----------|-------|------------|-----------|--------|----------------|-----------|------------------|--------|------------|")
        for u in users:
            uid8 = str(u["user_id"])[:8]
            p(
                f"| {u['display_name'] or '?'} "
                f"| `{uid8}` "
                f"| {u['session_count']} "
                f"| {u['turn_count']} "
                f"| {_fmt_ts(u['first_turn'])} "
                f"| {_fmt_ts(u['last_turn'])} "
                f"| {int(u['total_tokens']):,} "
                f"| {_pct(u['critic_rejects'], u['flow_b'])} ({u['critic_rejects']}/{u['flow_b']}) "
                f"| {_pct(u['fallback_count'], u['ac_total'])} ({u['fallback_count']}/{u['ac_total']}) "
                f"| {u['onboarding_count']} "
                f"| +{u['fact_delta']} "
                f"| +{u['evidence_delta']} |"
            )

    # ── Section 3: Quality concerns ────────────────────────────────────────
    h(2, "3 — Quality concerns")

    def _quality_item(r: dict[str, Any], desc: str, extra: str = "") -> str:
        uid8 = str(r["user_id"])[:8]
        tid = str(r["turn_id"])[:8]
        user = r.get("display_name") or "?"
        user_q = _trunc(r.get("user_message"), 200)
        coach_q = _trunc(r.get("assistant_message"), 200)
        out = (
            f"- **turn** `{tid}…` | **user** {user} (`{uid8}…`) | {desc}{extra}\n"
            f"  - **U:** {user_q or '(none)'}\n"
            f"  - **C:** {coach_q or '(none)'}"
        )
        return out

    # 3a. Critic exhausted retries
    h(3, "3a — Critic exhausted retries")
    if not quality["critic_exhausted"]:
        p("_None._")
    else:
        for r in quality["critic_exhausted"]:
            p(_quality_item(r, f"critic retried {r['retry_count']}x ending in reject"))

    # 3b. Agent fallbacks
    h(3, "3b — Agent fallbacks")
    if not quality["fallbacks"]:
        p("_None._")
    else:
        for r in quality["fallbacks"]:
            details = r.get("fallback_details") or ""
            p(_quality_item(r, "agent fell back", f" — {details}"))

    # 3c. Short / long responses
    h(3, "3c — Suspiciously short / long coach responses")
    if not quality["length_anomalies"]:
        p("_None._")
    else:
        for r in quality["length_anomalies"]:
            n = r["msg_len"]
            tag = "SHORT (<100 chars)" if n < 100 else "LONG (>2000 chars)"
            p(_quality_item(r, f"{tag}, {n} chars"))

    # 3d. Late clarifications
    h(3, "3d — Clarification after ≥3 prior turns")
    if not quality["late_clarifs"]:
        p("_None._")
    else:
        for r in quality["late_clarifs"]:
            p(_quality_item(r, f"clarification at turn_index={r['turn_index']}"))

    # 3e. User retries
    h(3, "3e — User re-fired prompt (dissatisfaction signal)")
    if not quality["user_retries"]:
        p("_None._")
    else:
        for r in quality["user_retries"]:
            p(_quality_item(r, "user message contains `[retried]`"))

    # ── Section 4: New extractions ─────────────────────────────────────────
    h(2, "4 — New extractions")

    if not extractions:
        p("_No new facts or hypotheses in this window._")
    else:
        for uid, data in extractions.items():
            h(3, f"{data['display_name'] or '?'} (`{uid[:8]}…`)")

            facts = data["facts"]
            hyps = data["hypotheses"]

            if facts:
                p("**Structured facts**\n")
                for f_ in facts:
                    p(f"- {f_}")
                p("")
            else:
                p("_No new facts._\n")

            if hyps:
                p("**Hypotheses**\n")
                for h_ in hyps:
                    p(f"- {h_}")
                p("")
            else:
                p("_No new hypotheses._\n")

    # ── Section 5: Full transcripts ────────────────────────────────────────
    h(2, "5 — Full transcripts")

    if not transcripts:
        p("_No sessions with activity in this window._")

    # Group sessions by user
    sessions_by_user: dict[str, list[dict[str, Any]]] = {}
    for sess in transcripts:
        sessions_by_user.setdefault(sess["user_id"], []).append(sess)

    for uid, sessions in sessions_by_user.items():
        display = sessions[0]["display_name"] or "?"
        h(3, f"{display} (`{uid[:8]}…`)")

        for sess in sessions:
            sid8 = sess["session_id"][:8]
            theory = sess.get("session_theory") or ""
            theory_note = f" — _{theory}_" if theory else ""
            h(4, f"Session `{sid8}…` started {_fmt_ts(sess['started_at'])}{theory_note}")

            if not sess["turns"]:
                p("_No turns._\n")
                continue

            for t in sess["turns"]:
                flow = t.get("flow_used") or "?"
                ts = _fmt_ts(t.get("created_at"))
                ver = t.get("deployment_version") or ""
                ver_note = f"  `{ver}`" if ver else ""
                idx = (t.get("turn_index") or 0) + 1
                p(f"\n**Turn {idx}** | flow=`{flow}` | {ts}{ver_note}\n")
                user_msg = (t.get("user_message") or "").strip()
                coach_msg = (t.get("assistant_message") or "").strip()
                p(f"> **USER:** {user_msg}\n")
                p(f"> **COACH:** {coach_msg}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _main() -> None:
    import asyncpg

    since, user_id, dsn_override, output_path = _parse_args()
    dsn = _resolve_dsn(dsn_override)

    if not dsn:
        print(
            "ERROR: no DSN. Pass --dsn 'postgresql://...' or set SUPABASE_DB_URL.",
            file=sys.stderr,
        )
        sys.exit(1)

    ssl = "require" if _needs_ssl(dsn) else None
    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=3,
        ssl=ssl,
        statement_cache_size=0 if ssl else 100,
        timeout=20.0,
        command_timeout=60.0,
    )

    now = datetime.now(UTC)
    print(
        f"Generating report: {since.strftime('%Y-%m-%d %H:%M UTC')} → {now.strftime('%Y-%m-%d %H:%M UTC')}",
        file=sys.stderr,
    )

    try:
        async with pool.acquire() as conn:
            # asyncpg connections are single-query-at-a-time; run sequentially.
            summary     = await _fetch_summary(conn, since, user_id)
            users       = await _fetch_user_breakdown(conn, since, user_id)
            quality     = await _fetch_quality(conn, since, user_id)
            extractions = await _fetch_extractions(conn, since, user_id)
            transcripts = await _fetch_transcripts(conn, since, user_id)
    finally:
        await pool.close()

    report = _render_report(
        since=since,
        now=now,
        summary=summary,
        users=users,
        quality=quality,
        extractions=extractions,
        transcripts=transcripts,
        user_filter=user_id,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {output_path}", file=sys.stderr)
    print(f"  {summary['active_users']} active user(s), {summary['turns']} turns", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(_main())
