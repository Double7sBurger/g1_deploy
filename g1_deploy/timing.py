"""Wall-clock pacing without replaying missed control periods."""

from __future__ import annotations

import math
import time


class PeriodicDeadline:
    """Maintain a cadence, but discard overdue deadlines after a stall.

    Call ``wait`` once at the end of each iteration. Small sleep overshoots retain the
    cadence; work overruns or a wake-up over one period late rebase it to the present.
    In particular, a 400 ms pause must not trigger 200 back-to-back physics steps.
    """

    def __init__(self, period: float, *, clock=None, sleep=None):
        if not math.isfinite(period) or period <= 0:
            raise ValueError("period must be finite and positive")
        self.period = period
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._deadline = self._clock()
        self.overruns = 0
        self.max_lateness = 0.0

    def wait(self) -> None:
        self._deadline += self.period
        now = self._clock()
        if now < self._deadline:
            self._sleep(self._deadline - now)
            now = self._clock()
            if now - self._deadline < self.period:
                return
        if now > self._deadline:
            self.overruns += 1
            self.max_lateness = max(self.max_lateness, now - self._deadline)
            self._deadline = now
