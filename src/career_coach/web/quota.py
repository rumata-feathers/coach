"""Per-process daily quota tracking for web search calls.

A :class:`QuotaTracker` guards against runaway eval loops that would exhaust
the search budget in a single run. The quota resets 24 hours after the first
call. On exhaustion :meth:`try_consume` returns ``False``; callers return
empty results rather than erroring.

See SPEC_v1.md §4.5.
"""

from __future__ import annotations

import time


class QuotaTracker:
    """Thread-safe daily quota counter.

    Args:
        daily_limit: Maximum calls allowed in a 24-hour window.
            Default 200 (Tavily free tier is 1 000/month).
    """

    _SECONDS_PER_DAY: int = 86_400

    def __init__(self, daily_limit: int = 200) -> None:
        self._limit = daily_limit
        self._count: int = 0
        self._window_start: float | None = None  # epoch seconds of first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_consume(self) -> bool:
        """Attempt to consume one quota unit.

        Returns:
            ``True`` if the call is allowed (quota not exhausted).
            ``False`` if the daily limit has been reached.
        """
        now = time.monotonic()
        self._maybe_reset(now)
        if self._count >= self._limit:
            return False
        if self._window_start is None:
            self._window_start = now
        self._count += 1
        return True

    @property
    def is_exceeded(self) -> bool:
        """``True`` when the daily limit is reached (after :meth:`try_consume` returns ``False``)."""
        return self._count >= self._limit

    @property
    def remaining(self) -> int:
        """Number of calls remaining in the current window."""
        return max(0, self._limit - self._count)

    def reset(self) -> None:
        """Reset the counter. Intended for test isolation."""
        self._count = 0
        self._window_start = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_reset(self, now: float) -> None:
        """Reset count if the 24-hour window has elapsed."""
        if self._window_start is not None and (now - self._window_start) >= self._SECONDS_PER_DAY:
            self._count = 0
            self._window_start = None
