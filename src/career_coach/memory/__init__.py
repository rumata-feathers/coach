"""Three-tier memory persistence layer.

Each module owns a single cognitive tier: structured facts, episodic turns,
or semantic hypotheses. Agents depend on these repos — never on raw SQL.
"""

from career_coach.memory.embeddings import EmbeddingsClient
from career_coach.memory.episodic import EpisodicRepo
from career_coach.memory.semantic import SemanticRepo
from career_coach.memory.structured import StructuredFactsRepo

__all__ = [
    "EmbeddingsClient",
    "EpisodicRepo",
    "SemanticRepo",
    "StructuredFactsRepo",
]
