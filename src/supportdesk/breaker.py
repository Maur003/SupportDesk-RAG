"""Circuit breaker and the shared retry delay.

A closed breaker allows calls. After `threshold` dependency failures it opens.
A later call is allowed once `reset_seconds` have passed. That trial closes the
breaker on success and opens it again on failure. Client errors are not failures.
"""

from collections.abc import Callable

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def retry_delay(backoff: float, attempt: int, rng: Callable[[], float]) -> float:
    """`backoff * 2^attempt * uniform(0.5, 1.0)`. `rng` returns a float in [0, 1)."""

    return backoff * (2**attempt) * (0.5 + 0.5 * rng())


class CircuitBreaker:
    def __init__(
        self,
        threshold: int,
        reset_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if reset_seconds < 0:
            raise ValueError("reset_seconds must be >= 0")
        self._threshold = threshold
        self._reset_seconds = reset_seconds
        self._clock = clock or _monotonic
        self._failures = 0
        self._opened_at: float | None = None
        self._trial = False

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if self._clock() - self._opened_at < self._reset_seconds or self._trial:
            return False
        self._trial = True
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._trial = False

    def record_failure(self) -> None:
        self._trial = False
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._clock()

    def end_trial(self) -> None:
        """A client error during a half-open trial does not count as success or failure."""

        self._trial = False


def _monotonic() -> float:
    import time

    return time.monotonic()
