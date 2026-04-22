"""Typed Pydantic schemas used at every agent boundary.

Agents never exchange raw ``dict`` values — they use the models in this
package. The modules are intentionally small and boring; the structural
contract is what matters, not any logic.
"""

from career_coach.models.agent_io import (
    CoachInput,
    CoachOutput,
    CriticInput,
    CriticVerdict,
    ProfilerInput,
    ProfilerOutput,
    UnderstanderInput,
)
from career_coach.models.intent import IntentPacket
from career_coach.models.user_model import (
    Challenge,
    EvidenceDraft,
    FactUpdate,
    Hypothesis,
    HypothesisStatus,
    TurnSummary,
)

__all__ = [
    "Challenge",
    "CoachInput",
    "CoachOutput",
    "CriticInput",
    "CriticVerdict",
    "EvidenceDraft",
    "FactUpdate",
    "Hypothesis",
    "HypothesisStatus",
    "IntentPacket",
    "ProfilerInput",
    "ProfilerOutput",
    "TurnSummary",
    "UnderstanderInput",
]
