"""Pydantic models for World KB YAML files.

Every numeric field in the YAML must have a corresponding ``source_notes``
entry. This is enforced structurally by the model validators and by the
unit-test lint (``tests/kb/test_kb_structure.py``).

See SPEC_v1.md §13.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Career entry
# ---------------------------------------------------------------------------


class EntryPath(BaseModel):
    """One route into a career.

    Attributes:
        path: Human-readable description of the entry route.
        typical_timeline_years: Approximate years from starting this path
            to a qualified entry-level role. Numeric — must be cited in
            the parent :class:`CareerSourceNotes`.
        selectivity: Qualitative description of how competitive this path is.
    """

    path: str
    typical_timeline_years: int
    selectivity: str


class EarlyCareerPay(BaseModel):
    """Approximate UK early-career salary bands.

    All figures are gross annual GBP. ``band_p50_gbp`` and
    ``band_p90_gbp`` are numeric — must be cited in :class:`CareerSourceNotes`.

    Attributes:
        band_p50_gbp: Approximate median gross annual salary.
        band_p90_gbp: Approximate 90th-percentile gross annual salary.
        year_ref: Reference year for the figures.
        notes: Optional clarifying note (e.g. "includes trainee stipend").
    """

    band_p50_gbp: int
    band_p90_gbp: int
    year_ref: int
    notes: str | None = None

    @field_validator("band_p50_gbp", "band_p90_gbp")
    @classmethod
    def positive(cls, v: int) -> int:
        """Pay figures must be positive."""
        if v <= 0:
            raise ValueError("pay band must be positive")
        return v

    @model_validator(mode="after")
    def p90_gte_p50(self) -> EarlyCareerPay:
        """p90 must be ≥ p50 (otherwise the data is wrong)."""
        if self.band_p90_gbp < self.band_p50_gbp:
            raise ValueError(
                f"band_p90_gbp ({self.band_p90_gbp}) must be ≥ band_p50_gbp ({self.band_p50_gbp})"
            )
        return self


class CareerSourceNotes(BaseModel):
    """Source citations for the numeric fields in :class:`CareerEntry`.

    Every numeric field in ``CareerEntry`` maps to a field here.
    Validators reject empty strings — a one-word source is not acceptable.

    Attributes:
        pay: Source for ``early_career_pay_uk`` figures.
        hours: Source for ``typical_hours_per_week`` figures.
        entry_paths: Source for ``entry_paths[*].typical_timeline_years``.
    """

    pay: str
    hours: str
    entry_paths: str

    @field_validator("pay", "hours", "entry_paths")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Source notes must be non-empty strings."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("source_notes field must not be empty")
        return stripped


class CareerEntry(BaseModel):
    """One entry in the World Knowledge Base.

    Attributes:
        name: Canonical snake_case identifier (matches the YAML filename stem).
        display_name: Human-readable name shown to users.
        tags: List of searchable tags (e.g. ``math_heavy``, ``high_comp``).
        variants: Specialist sub-roles within this career.
        entry_paths: One or more routes into this career.
        typical_hours_per_week: ``[min, max]`` weekly hours, approximately.
        early_career_pay_uk: Pay band for early-career professionals in the UK.
        typical_exits: Common next destinations from this career.
        risks: Known downsides or hazards of this career path.
        required_signals: Positive indicators that someone may fit this career.
        contra_indicators: Signals that suggest poor fit.
        source_notes: Citations for all numeric fields.
    """

    name: str
    display_name: str
    tags: list[str]
    variants: list[str] = []
    entry_paths: list[EntryPath]
    typical_hours_per_week: list[int]
    early_career_pay_uk: EarlyCareerPay
    typical_exits: list[str] = []
    risks: list[str] = []
    required_signals: list[str] = []
    contra_indicators: list[str] = []
    source_notes: CareerSourceNotes

    @field_validator("typical_hours_per_week")
    @classmethod
    def hours_is_min_max(cls, v: list[int]) -> list[int]:
        """Must be exactly two non-negative ints: [min, max]."""
        if len(v) != 2:
            raise ValueError("typical_hours_per_week must be [min, max]")
        if v[0] < 0 or v[1] < v[0]:
            raise ValueError("typical_hours_per_week: min must be ≥0 and max ≥ min")
        return v

    @field_validator("entry_paths")
    @classmethod
    def at_least_one_path(cls, v: list[EntryPath]) -> list[EntryPath]:
        """Every career must have at least one entry path."""
        if not v:
            raise ValueError("entry_paths must contain at least one path")
        return v


# ---------------------------------------------------------------------------
# Pay band reference
# ---------------------------------------------------------------------------


class PayBand(BaseModel):
    """A broad sector pay band entry from ``kb/pay_bands_uk.yaml``.

    Attributes:
        sector: Sector identifier (matches ``career.name`` where possible).
        role_example: Illustrative role the band applies to.
        band_p25_gbp: Approximate 25th-percentile gross annual salary.
        band_p50_gbp: Approximate median gross annual salary.
        band_p75_gbp: Approximate 75th-percentile gross annual salary.
        year_ref: Reference year.
        source_notes: Citation for the numeric figures.
    """

    sector: str
    role_example: str
    band_p25_gbp: int
    band_p50_gbp: int
    band_p75_gbp: int
    year_ref: int
    source_notes: str

    @field_validator("source_notes")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Source notes must be non-empty."""
        if not v.strip():
            raise ValueError("source_notes must not be empty")
        return v

    @model_validator(mode="after")
    def ordered_bands(self) -> PayBand:
        """p25 ≤ p50 ≤ p75."""
        if not (self.band_p25_gbp <= self.band_p50_gbp <= self.band_p75_gbp):
            raise ValueError("pay bands must satisfy p25 ≤ p50 ≤ p75")
        return self


# ---------------------------------------------------------------------------
# Exit path reference
# ---------------------------------------------------------------------------


class ExitPath(BaseModel):
    """One exit destination from ``kb/exits.yaml``.

    Attributes:
        id: Canonical snake_case identifier (referenced in ``CareerEntry.typical_exits``).
        label: Human-readable label.
        description: One or two sentences describing this exit destination.
        typical_from: Career names this exit is commonly reached from.
        source_notes: Citation for factual claims in ``description``.
    """

    id: str
    label: str
    description: str
    typical_from: list[str]
    source_notes: str

    @field_validator("source_notes")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Source notes must be non-empty."""
        if not v.strip():
            raise ValueError("source_notes must not be empty")
        return v
