"""World Knowledge Base repository.

:class:`WorldKBRepo` loads all YAML files from the ``kb/`` directory at
construction time and exposes three query methods used by the Researcher
agent.

Loading is eager and synchronous — the KB is small (15 careers, 2 reference
files) and must be available at process start without async overhead.

See SPEC_v1.md §13.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from career_coach.kb.models import CareerEntry, ExitPath, PayBand

logger = logging.getLogger("career_coach.kb.repo")

_DEFAULT_KB_ROOT = Path(__file__).resolve().parents[4] / "kb"


class WorldKBRepo:
    """Immutable in-memory view of the World Knowledge Base.

    Args:
        kb_root: Path to the ``kb/`` directory.  Defaults to the project-root
            ``kb/`` directory resolved relative to this file.

    Raises:
        FileNotFoundError: If ``kb_root`` does not exist.
        pydantic.ValidationError: If any YAML file fails schema validation.
    """

    def __init__(self, kb_root: Path = _DEFAULT_KB_ROOT) -> None:
        if not kb_root.is_dir():
            raise FileNotFoundError(f"KB root not found: {kb_root}")

        self._careers: dict[str, CareerEntry] = {}
        self._pay_bands: list[PayBand] = []
        self._exits: dict[str, ExitPath] = {}

        self._load_careers(kb_root / "careers")
        self._load_pay_bands(kb_root / "pay_bands_uk.yaml")
        self._load_exits(kb_root / "exits.yaml")

        logger.info(
            "WorldKBRepo loaded: %d careers, %d pay bands, %d exits",
            len(self._careers),
            len(self._pay_bands),
            len(self._exits),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_career(self, name: str) -> CareerEntry | None:
        """Return the career entry for *name*, or ``None`` if not found.

        Args:
            name: Canonical snake_case name (e.g. ``"quant_finance"``).
        """
        return self._careers.get(name)

    def search_by_tags(self, tags: list[str]) -> list[CareerEntry]:
        """Return all careers whose ``tags`` list contains ALL of *tags*.

        Args:
            tags: Required tags. A career is returned only if it has every
                tag in this list (AND semantics, not OR).

        Returns:
            Matching :class:`CareerEntry` objects sorted by name.
        """
        tag_set = set(tags)
        matches = [
            entry
            for entry in self._careers.values()
            if tag_set.issubset(set(entry.tags))
        ]
        return sorted(matches, key=lambda e: e.name)

    def list_all_careers(self) -> list[CareerEntry]:
        """Return all loaded career entries sorted alphabetically by name."""
        return sorted(self._careers.values(), key=lambda e: e.name)

    def list_pay_bands(self) -> list[PayBand]:
        """Return all loaded pay bands."""
        return list(self._pay_bands)

    def get_exit(self, exit_id: str) -> ExitPath | None:
        """Return the exit path for *exit_id*, or ``None`` if not found."""
        return self._exits.get(exit_id)

    def list_exits(self) -> list[ExitPath]:
        """Return all loaded exit paths sorted by id."""
        return sorted(self._exits.values(), key=lambda e: e.id)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_careers(self, careers_dir: Path) -> None:
        """Load all ``*.yaml`` files under *careers_dir*."""
        if not careers_dir.is_dir():
            logger.warning("careers directory not found: %s", careers_dir)
            return
        for path in sorted(careers_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            try:
                entry = CareerEntry(**raw)
                self._careers[entry.name] = entry
                logger.debug("loaded career: %s", entry.name)
            except Exception as exc:
                logger.error("failed to load career %s: %s", path.name, exc)
                raise

    def _load_pay_bands(self, path: Path) -> None:
        """Load ``kb/pay_bands_uk.yaml``."""
        if not path.exists():
            logger.warning("pay_bands_uk.yaml not found: %s", path)
            return
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for item in raw.get("pay_bands", []):
            try:
                self._pay_bands.append(PayBand(**item))
            except Exception as exc:
                logger.error("failed to load pay band %s: %s", item, exc)
                raise

    def _load_exits(self, path: Path) -> None:
        """Load ``kb/exits.yaml``."""
        if not path.exists():
            logger.warning("exits.yaml not found: %s", path)
            return
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for item in raw.get("exits", []):
            try:
                exit_path = ExitPath(**item)
                self._exits[exit_path.id] = exit_path
            except Exception as exc:
                logger.error("failed to load exit %s: %s", item, exc)
                raise
