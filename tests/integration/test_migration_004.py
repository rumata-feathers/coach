"""Integration test for migration 004.

Verifies that the v1 schema additions apply cleanly on top of the existing
migrations and that the new tables + columns are queryable.

Requires a live Postgres instance (skipped if none is reachable — see conftest).
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

import asyncpg
import pytest

pytestmark = pytest.mark.asyncio


async def test_new_turns_columns_exist(db_conn: asyncpg.Connection) -> None:
    """research_brief_id, synthesizer_output, devils_advocate_output, and
    chart_specs must exist on the turns table after migration 004."""
    rows = await db_conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'turns' AND table_schema = 'public'
        """,
    )
    column_names = {r["column_name"] for r in rows}
    for col in ("research_brief_id", "synthesizer_output", "devils_advocate_output", "chart_specs"):
        assert col in column_names, f"turns.{col} is missing after migration 004"


async def test_research_briefs_table_exists(db_conn: asyncpg.Connection) -> None:
    """research_briefs table must exist."""
    result = await db_conn.fetchval(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'research_briefs' AND table_schema = 'public'"
    )
    assert result == 1, "research_briefs table not found"


async def test_supervisor_events_table_exists(db_conn: asyncpg.Connection) -> None:
    """supervisor_events table must exist."""
    result = await db_conn.fetchval(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'supervisor_events' AND table_schema = 'public'"
    )
    assert result == 1, "supervisor_events table not found"


async def test_research_briefs_fk_and_roundtrip(db_conn: asyncpg.Connection) -> None:
    """Insert a minimal fixture chain (user → session → turn → research_brief)
    and read back the brief row, asserting JSON fields survive the round-trip."""
    user_id = uuid4()
    session_id = uuid4()
    turn_id = uuid4()
    brief_id = uuid4()

    now = datetime.utcnow()

    await db_conn.execute(
        "INSERT INTO users (user_id, display_name, is_test) VALUES ($1, $2, TRUE)",
        user_id,
        "migration-004-test-user",
    )
    await db_conn.execute(
        "INSERT INTO sessions (session_id, user_id, started_at) VALUES ($1, $2, $3)",
        session_id,
        user_id,
        now,
    )
    await db_conn.execute(
        """
        INSERT INTO turns
            (turn_id, session_id, user_id, turn_index, user_message, flow_used)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        turn_id,
        session_id,
        user_id,
        0,
        "What is quant finance?",
        "C",
    )

    findings = [{"claim": "Quant finance requires strong maths.", "confidence": 0.9,
                 "citations": [{"url": "https://example.com", "kb_path": None,
                                "title": "Quant roles",
                                "accessed_at": now.isoformat(), "source_type": "web"}],
                 "is_numeric": False}]

    await db_conn.execute(
        """
        INSERT INTO research_briefs
            (brief_id, turn_id, question, findings, caveats, next_questions,
             used_kb_files, web_searches_run, web_sources_consulted)
        VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb,
                $7::jsonb, $8::jsonb, $9::jsonb)
        """,
        brief_id,
        turn_id,
        "What is quant finance?",
        json.dumps(findings),
        json.dumps(["Information may be outdated."]),
        json.dumps(["Is quant finance hiring in 2026?"]),
        json.dumps([]),
        json.dumps(["quant finance careers 2026"]),
        json.dumps(["https://example.com"]),
    )

    row = await db_conn.fetchrow(
        "SELECT brief_id, question, findings FROM research_briefs WHERE brief_id = $1",
        brief_id,
    )
    assert row is not None
    assert row["question"] == "What is quant finance?"
    stored_findings = json.loads(row["findings"])
    assert len(stored_findings) == 1
    assert stored_findings[0]["claim"] == "Quant finance requires strong maths."

    # FK: linking turn back to the brief via research_brief_id
    await db_conn.execute(
        "UPDATE turns SET research_brief_id = $1 WHERE turn_id = $2",
        brief_id,
        turn_id,
    )
    linked = await db_conn.fetchval(
        "SELECT research_brief_id FROM turns WHERE turn_id = $1",
        turn_id,
    )
    assert linked == brief_id


async def test_supervisor_events_fk_and_roundtrip(db_conn: asyncpg.Connection) -> None:
    """Insert a supervisor event linked to a turn and read it back."""
    user_id = uuid4()
    session_id = uuid4()
    turn_id = uuid4()
    event_id = uuid4()

    now = datetime.utcnow()

    await db_conn.execute(
        "INSERT INTO users (user_id, display_name, is_test) VALUES ($1, $2, TRUE)",
        user_id,
        "migration-004-supervisor-test",
    )
    await db_conn.execute(
        "INSERT INTO sessions (session_id, user_id, started_at) VALUES ($1, $2, $3)",
        session_id,
        user_id,
        now,
    )
    await db_conn.execute(
        """
        INSERT INTO turns
            (turn_id, session_id, user_id, turn_index, user_message, flow_used)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        turn_id,
        session_id,
        user_id,
        0,
        "I'm feeling really hopeless today.",
        "A",
    )

    await db_conn.execute(
        """
        INSERT INTO supervisor_events
            (event_id, turn_id, event_type, severity, details, action_taken)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6)
        """,
        event_id,
        turn_id,
        "unsafe",
        "high",
        json.dumps({"keyword_matched": "hopeless"}),
        "block",
    )

    row = await db_conn.fetchrow(
        "SELECT event_type, severity, action_taken, details "
        "FROM supervisor_events WHERE event_id = $1",
        event_id,
    )
    assert row is not None
    assert row["event_type"] == "unsafe"
    assert row["severity"] == "high"
    assert row["action_taken"] == "block"
    details = json.loads(row["details"])
    assert details["keyword_matched"] == "hopeless"


async def test_chart_specs_default_is_empty_array(db_conn: asyncpg.Connection) -> None:
    """turns.chart_specs must default to an empty JSON array when not supplied."""
    user_id = uuid4()
    session_id = uuid4()
    turn_id = uuid4()

    now = datetime.utcnow()

    await db_conn.execute(
        "INSERT INTO users (user_id, display_name, is_test) VALUES ($1, $2, TRUE)",
        user_id,
        "migration-004-chart-test",
    )
    await db_conn.execute(
        "INSERT INTO sessions (session_id, user_id, started_at) VALUES ($1, $2, $3)",
        session_id,
        user_id,
        now,
    )
    await db_conn.execute(
        """
        INSERT INTO turns
            (turn_id, session_id, user_id, turn_index, user_message, flow_used)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        turn_id,
        session_id,
        user_id,
        0,
        "Compare quant vs SWE comp.",
        "C",
    )

    raw = await db_conn.fetchval(
        "SELECT chart_specs FROM turns WHERE turn_id = $1",
        turn_id,
    )
    # asyncpg returns JSONB as a raw string; parse it
    value = json.loads(raw) if isinstance(raw, str) else raw
    assert value == [], f"chart_specs default should be [] but got {raw!r}"


async def test_all_v1_tables_visible(db_conn: asyncpg.Connection) -> None:
    """research_briefs and supervisor_events must appear in pg_tables after 004."""
    rows = await db_conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    tables = {r["tablename"] for r in rows}
    for expected in ("research_briefs", "supervisor_events"):
        assert expected in tables, f"Expected table {expected!r} not found in public schema"
