"""Structural tests for the World Knowledge Base.

Covers:
- All 15 career YAML files load and validate against Pydantic models.
- Every career's source_notes has non-empty pay, hours, and entry_paths fields.
- Round-trip serialisation: model_dump() → CareerEntry(**data) is stable.
- WorldKBRepo.list_all_careers() returns exactly 15 entries.
- WorldKBRepo.search_by_tags(["math_heavy"]) returns the three expected careers.
- Pay bands and exit paths load without error.

See SPEC_v1.md §13 and TASKS_v1.md Task 3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from career_coach.kb.models import CareerEntry, ExitPath, PayBand
from career_coach.kb.repo import WorldKBRepo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_KB_ROOT = Path(__file__).resolve().parents[2] / "kb"
_CAREERS_DIR = _KB_ROOT / "careers"

# The 15 canonical career names the spec requires.
EXPECTED_CAREER_NAMES = {
    "quant_finance",
    "investment_banking",
    "software_engineering",
    "academia_stem_phd_track",
    "medicine_uk",
    "management_consulting",
    "machine_learning_research",
    "product_management",
    "data_science",
    "law_uk",
    "civil_service_uk",
    "teaching",
    "design_ux",
    "journalism",
    "entrepreneurship_early",
}

# Careers that must appear when searching for the "math_heavy" tag.
MATH_HEAVY_CAREERS = {
    "quant_finance",
    "machine_learning_research",
    "academia_stem_phd_track",
}


@pytest.fixture(scope="module")
def repo() -> WorldKBRepo:
    """WorldKBRepo loaded from the real kb/ directory."""
    return WorldKBRepo(kb_root=_KB_ROOT)


@pytest.fixture(scope="module")
def all_career_yamls() -> list[tuple[Path, dict]]:
    """All raw YAML dicts from kb/careers/*.yaml, paired with their paths."""
    results = []
    for path in sorted(_CAREERS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        results.append((path, raw))
    return results


# ---------------------------------------------------------------------------
# 1. Load count
# ---------------------------------------------------------------------------


class TestCareerCount:
    def test_exactly_15_career_files(self, all_career_yamls: list) -> None:
        """There must be exactly 15 YAML files in kb/careers/."""
        assert len(all_career_yamls) == 15, (
            f"Expected 15 career YAML files, found {len(all_career_yamls)}: "
            f"{[p.name for p, _ in all_career_yamls]}"
        )

    def test_repo_returns_15_careers(self, repo: WorldKBRepo) -> None:
        """WorldKBRepo.list_all_careers() must return exactly 15 entries."""
        careers = repo.list_all_careers()
        assert len(careers) == 15, (
            f"Expected 15 careers from repo, got {len(careers)}: "
            f"{[c.name for c in careers]}"
        )


# ---------------------------------------------------------------------------
# 2. Pydantic validation — each file loads cleanly
# ---------------------------------------------------------------------------


class TestCareerValidation:
    def test_all_files_validate(self, all_career_yamls: list) -> None:
        """Every YAML file must parse into a valid CareerEntry without errors."""
        errors: list[str] = []
        for path, raw in all_career_yamls:
            try:
                CareerEntry(**raw)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        assert not errors, "Validation errors:\n" + "\n".join(errors)

    def test_entry_paths_non_empty(self, all_career_yamls: list) -> None:
        """Every career must declare at least one entry path."""
        for path, raw in all_career_yamls:
            entry = CareerEntry(**raw)
            assert entry.entry_paths, f"{path.name}: entry_paths is empty"

    def test_hours_is_min_max_pair(self, all_career_yamls: list) -> None:
        """typical_hours_per_week must be exactly [min, max] with max >= min."""
        for path, raw in all_career_yamls:
            entry = CareerEntry(**raw)
            h = entry.typical_hours_per_week
            assert len(h) == 2, f"{path.name}: hours list has {len(h)} elements"
            assert h[1] >= h[0], f"{path.name}: max hours < min hours"

    def test_pay_band_p90_gte_p50(self, all_career_yamls: list) -> None:
        """band_p90_gbp must be ≥ band_p50_gbp for every career."""
        for path, raw in all_career_yamls:
            entry = CareerEntry(**raw)
            pay = entry.early_career_pay_uk
            assert pay.band_p90_gbp >= pay.band_p50_gbp, (
                f"{path.name}: p90 ({pay.band_p90_gbp}) < p50 ({pay.band_p50_gbp})"
            )


# ---------------------------------------------------------------------------
# 3. Source-notes structural lint
# ---------------------------------------------------------------------------


class TestSourceNotesLint:
    """Every numeric field must have a non-empty source_notes sibling.

    The relevant numeric fields are:
    - early_career_pay_uk.band_p50_gbp / band_p90_gbp  → source_notes.pay
    - typical_hours_per_week                            → source_notes.hours
    - entry_paths[*].typical_timeline_years             → source_notes.entry_paths

    Each source_notes string must contain at least two words (a single word
    is not an acceptable citation).
    """

    @staticmethod
    def _is_substantive(note: str) -> bool:
        """True if the note has at least two whitespace-separated tokens."""
        return len(note.split()) >= 2

    def test_source_notes_pay_non_trivial(self, all_career_yamls: list) -> None:
        for path, raw in all_career_yamls:
            entry = CareerEntry(**raw)
            note = entry.source_notes.pay
            assert self._is_substantive(note), (
                f"{path.name}: source_notes.pay is too short: {note!r}"
            )

    def test_source_notes_hours_non_trivial(self, all_career_yamls: list) -> None:
        for path, raw in all_career_yamls:
            entry = CareerEntry(**raw)
            note = entry.source_notes.hours
            assert self._is_substantive(note), (
                f"{path.name}: source_notes.hours is too short: {note!r}"
            )

    def test_source_notes_entry_paths_non_trivial(
        self, all_career_yamls: list
    ) -> None:
        for path, raw in all_career_yamls:
            entry = CareerEntry(**raw)
            note = entry.source_notes.entry_paths
            assert self._is_substantive(note), (
                f"{path.name}: source_notes.entry_paths is too short: {note!r}"
            )

    def test_pay_figures_positive(self, all_career_yamls: list) -> None:
        """Pay band figures must be positive integers."""
        for path, raw in all_career_yamls:
            entry = CareerEntry(**raw)
            pay = entry.early_career_pay_uk
            assert pay.band_p50_gbp > 0, f"{path.name}: band_p50_gbp is not positive"
            assert pay.band_p90_gbp > 0, f"{path.name}: band_p90_gbp is not positive"


# ---------------------------------------------------------------------------
# 4. Round-trip serialisation
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_model_dump_is_stable(self, all_career_yamls: list) -> None:
        """CareerEntry → model_dump() → CareerEntry must produce identical objects."""
        for path, raw in all_career_yamls:
            original = CareerEntry(**raw)
            dumped = original.model_dump()
            reconstituted = CareerEntry(**dumped)
            assert original == reconstituted, (
                f"{path.name}: round-trip produced a different CareerEntry"
            )


# ---------------------------------------------------------------------------
# 5. Tag search
# ---------------------------------------------------------------------------


class TestTagSearch:
    def test_math_heavy_returns_expected_careers(self, repo: WorldKBRepo) -> None:
        """search_by_tags(['math_heavy']) must include quant_finance, ML research,
        and academia_stem_phd_track."""
        results = repo.search_by_tags(["math_heavy"])
        names = {c.name for c in results}
        missing = MATH_HEAVY_CAREERS - names
        assert not missing, (
            f"math_heavy search missing expected careers: {missing}. "
            f"Got: {names}"
        )

    def test_math_heavy_returns_only_tagged_careers(self, repo: WorldKBRepo) -> None:
        """Every result from search_by_tags(['math_heavy']) must actually have the tag."""
        results = repo.search_by_tags(["math_heavy"])
        for entry in results:
            assert "math_heavy" in entry.tags, (
                f"{entry.name} returned by math_heavy search but lacks the tag"
            )

    def test_multi_tag_and_semantics(self, repo: WorldKBRepo) -> None:
        """search_by_tags with two tags uses AND semantics — result must have both."""
        results = repo.search_by_tags(["math_heavy", "high_comp"])
        for entry in results:
            assert "math_heavy" in entry.tags, f"{entry.name} missing math_heavy"
            assert "high_comp" in entry.tags, f"{entry.name} missing high_comp"

    def test_nonexistent_tag_returns_empty(self, repo: WorldKBRepo) -> None:
        """A tag that no career has must return an empty list."""
        results = repo.search_by_tags(["tag_that_does_not_exist_xyz"])
        assert results == []

    def test_results_sorted_by_name(self, repo: WorldKBRepo) -> None:
        """search_by_tags results must be sorted alphabetically by name."""
        results = repo.search_by_tags(["math_heavy"])
        names = [c.name for c in results]
        assert names == sorted(names), f"Results not sorted: {names}"


# ---------------------------------------------------------------------------
# 6. get_career lookup
# ---------------------------------------------------------------------------


class TestGetCareer:
    def test_get_career_known_name(self, repo: WorldKBRepo) -> None:
        """get_career returns the correct entry for a known name."""
        entry = repo.get_career("quant_finance")
        assert entry is not None
        assert entry.name == "quant_finance"
        assert entry.display_name == "Quantitative Finance"

    def test_get_career_unknown_name_returns_none(self, repo: WorldKBRepo) -> None:
        """get_career returns None for an unknown name."""
        assert repo.get_career("does_not_exist") is None


# ---------------------------------------------------------------------------
# 7. Pay bands
# ---------------------------------------------------------------------------


class TestPayBands:
    def test_pay_bands_load(self, repo: WorldKBRepo) -> None:
        """list_pay_bands() must return at least one pay band."""
        bands = repo.list_pay_bands()
        assert bands, "No pay bands loaded"

    def test_pay_band_ordering(self, repo: WorldKBRepo) -> None:
        """p25 ≤ p50 ≤ p75 for every pay band."""
        for band in repo.list_pay_bands():
            assert band.band_p25_gbp <= band.band_p50_gbp <= band.band_p75_gbp, (
                f"Pay band {band.sector}: ordering violated "
                f"({band.band_p25_gbp}/{band.band_p50_gbp}/{band.band_p75_gbp})"
            )

    def test_pay_band_source_notes_non_empty(self, repo: WorldKBRepo) -> None:
        """Every pay band must have non-empty source_notes."""
        for band in repo.list_pay_bands():
            assert band.source_notes.strip(), (
                f"Pay band {band.sector} has empty source_notes"
            )


# ---------------------------------------------------------------------------
# 8. Exit paths
# ---------------------------------------------------------------------------


class TestExitPaths:
    def test_exits_load(self, repo: WorldKBRepo) -> None:
        """list_exits() must return at least one exit path."""
        exits = repo.list_exits()
        assert exits, "No exit paths loaded"

    def test_get_exit_known_id(self, repo: WorldKBRepo) -> None:
        """get_exit returns the correct entry for a known id."""
        exit_path = repo.get_exit("hedge_fund")
        assert exit_path is not None
        assert exit_path.id == "hedge_fund"

    def test_get_exit_unknown_id_returns_none(self, repo: WorldKBRepo) -> None:
        """get_exit returns None for an unknown id."""
        assert repo.get_exit("does_not_exist_xyz") is None

    def test_exit_source_notes_non_empty(self, repo: WorldKBRepo) -> None:
        """Every exit path must have non-empty source_notes."""
        for exit_path in repo.list_exits():
            assert exit_path.source_notes.strip(), (
                f"Exit {exit_path.id} has empty source_notes"
            )

    def test_exits_sorted_by_id(self, repo: WorldKBRepo) -> None:
        """list_exits() results must be sorted alphabetically by id."""
        exits = repo.list_exits()
        ids = [e.id for e in exits]
        assert ids == sorted(ids), f"Exits not sorted: {ids}"


# ---------------------------------------------------------------------------
# 9. Missing KB root raises FileNotFoundError
# ---------------------------------------------------------------------------


class TestRepoInit:
    def test_missing_root_raises(self, tmp_path: Path) -> None:
        """WorldKBRepo raises FileNotFoundError when kb_root does not exist."""
        with pytest.raises(FileNotFoundError, match="KB root not found"):
            WorldKBRepo(kb_root=tmp_path / "nonexistent")

    def test_list_all_careers_sorted(self, repo: WorldKBRepo) -> None:
        """list_all_careers() results must be sorted alphabetically by name."""
        careers = repo.list_all_careers()
        names = [c.name for c in careers]
        assert names == sorted(names), f"Careers not sorted: {names}"
