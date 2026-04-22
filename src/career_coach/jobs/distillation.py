"""Distillation job — v0 stub.

Processes raw evidence signals queued by the Profiler and matches them to
hypotheses about the user. Runs as a CLI script or via the admin API.

Usage:
    python -m career_coach.jobs.distillation --user-id <UUID>

See SPEC §6.6 for the architectural contract. v1 will add clustering,
conflict detection, and dormancy logic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from career_coach.db import close_pool
from career_coach.llm.client import LLMClient, Message
from career_coach.llm.factory import LLMFactory
from career_coach.memory.semantic import SemanticRepo

logger = logging.getLogger("career_coach.jobs.distillation")

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "config" / "prompts"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape([]),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


async def run_distillation(
    user_id: UUID,
    *,
    factory: LLMFactory | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Process all unmatched evidence drafts for a user.

    Args:
        user_id: The user whose evidence queue to drain.
        factory: Optional factory override (uses default if ``None``).
        llm_client: Inject a mock client for tests (takes precedence over factory).

    Returns:
        A summary dict: ``{"processed": N, "created": M, "matched": K}``.
    """
    semantic = SemanticRepo()

    evidence_items = await semantic.get_unmatched_evidence(user_id)
    if not evidence_items:
        logger.info("No unmatched evidence for user %s. Nothing to do.", user_id)
        return {"processed": 0, "created": 0, "matched": 0}

    hypotheses = await semantic.get_active(user_id)
    logger.info(
        "Distilling %d evidence item(s) against %d hypothesis/hypotheses for user %s.",
        len(evidence_items),
        len(hypotheses),
        user_id,
    )

    # Render prompt
    template = _JINJA_ENV.get_template("distillation.j2")
    prompt = template.render(hypotheses=hypotheses, evidence_items=evidence_items)
    messages = [Message(role="user", content=prompt)]

    # LLM call
    client = llm_client or _get_default_client(factory)
    response = await client.complete(
        messages,
        model=_get_model(factory),
        temperature=0.1,
        max_tokens=2000,
        response_format="json",
    )

    try:
        result = _parse_response(response.text)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Distillation LLM response parse error: %s", exc)
        return {"processed": 0, "created": 0, "matched": 0, "error": str(exc)}

    # Index evidence by ID for lookup
    evidence_by_id = {str(e["evidence_id"]): e for e in evidence_items}

    # Running confidence for each hypothesis (accumulates deltas within a single run)
    running_conf: dict[UUID, float] = {h.hypothesis_id: h.confidence for h in hypotheses}

    # Collect new hypothesis IDs created in this run (keyed by statement)
    new_hyp_cache: dict[str, UUID] = {}
    new_hyp_conf: dict[UUID, float] = {}  # initial confidence for newly created hypotheses
    created = 0
    matched = 0

    for match in result.get("matches", []):
        evidence_id_str = str(match.get("evidence_id", ""))
        ev = evidence_by_id.get(evidence_id_str)
        if ev is None:
            logger.warning("Distillation returned unknown evidence_id %s; skipping.", evidence_id_str)
            continue

        evidence_id = ev["evidence_id"]
        action = match.get("action")
        delta = float(match.get("confidence_delta", 0.0))

        if action == "match_existing":
            try:
                hyp_id = UUID(str(match["hypothesis_id"]))
            except (ValueError, KeyError):
                logger.warning("Invalid hypothesis_id in match; skipping.")
                continue
            await semantic.assign_evidence(evidence_id, hyp_id)
            # Accumulate delta against the running confidence (not the DB value)
            current = running_conf.get(hyp_id, new_hyp_conf.get(hyp_id, 0.3))
            new_conf = max(0.0, min(1.0, current + delta))
            running_conf[hyp_id] = new_conf
            await semantic.update_confidence(hyp_id, new_conf)
            matched += 1

        elif action == "create_new":
            statement = str(match.get("statement", "")).strip()
            if not statement:
                logger.warning("create_new action has empty statement; skipping.")
                continue
            # Reuse if we already created this hypothesis in the same run
            if statement in new_hyp_cache:
                hyp_id = new_hyp_cache[statement]
                # Accumulate delta on the new hypothesis
                current = running_conf.get(hyp_id, new_hyp_conf.get(hyp_id, 0.3))
                new_conf = max(0.0, min(1.0, current + delta))
                running_conf[hyp_id] = new_conf
                await semantic.update_confidence(hyp_id, new_conf)
            else:
                initial_conf = float(match.get("initial_confidence", 0.3))
                initial_conf = max(0.0, min(1.0, initial_conf))
                hyp_id = await semantic.create_hypothesis(
                    user_id, statement, confidence=initial_conf
                )
                new_hyp_cache[statement] = hyp_id
                new_hyp_conf[hyp_id] = initial_conf
                running_conf[hyp_id] = initial_conf
                created += 1
            await semantic.assign_evidence(evidence_id, hyp_id)

        else:
            logger.warning("Unknown action '%s' in distillation output; skipping.", action)

    summary = {
        "processed": len(result.get("matches", [])),
        "created": created,
        "matched": matched,
    }
    logger.info("Distillation complete for user %s: %s", user_id, summary)
    return summary


# ---- helpers ---------------------------------------------------------------


def _get_default_client(factory: LLMFactory | None) -> LLMClient:
    f = factory or LLMFactory()
    return f.client_for("profiler")  # reuse the profiler's cheap model


def _get_model(factory: LLMFactory | None) -> str:
    f = factory or LLMFactory()
    return f.config_for("profiler").model


def _parse_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    return json.loads(text)  # type: ignore[no-any-return]


# ---- CLI entry point -------------------------------------------------------


async def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(
        description="Run the distillation job for one user.",
    )
    parser.add_argument("--user-id", required=True, help="UUID of the user to distil for.")
    args = parser.parse_args()

    user_id = UUID(args.user_id)
    summary = await run_distillation(user_id)
    print(json.dumps(summary, indent=2))
    await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
