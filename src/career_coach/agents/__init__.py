"""Agent modules for the career-coach pipeline.

Task 6 implements: Understander, Agent base.
Tasks 7-8 implement: Orchestrator, Coach, Critic, Profiler.
"""

from career_coach.agents.base import Agent
from career_coach.agents.understander import Understander

__all__ = ["Agent", "Understander"]
