"""Stage timing utilities for pipeline latency tracking.

Provides both a context-manager-based StageTimer for fine-grained stage
measurement and a decorator for timing entire function calls.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StageResult:
    """Result of a single timed stage."""
    name: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class StageTimer:
    """Context-manager-based timer for tracking pipeline stage latencies.

    Usage:
        timer = StageTimer()
        with timer.stage("embedding"):
            embeddings = model.encode(text)
        with timer.stage("retrieval"):
            results = retriever.search(embeddings)
        print(timer.summary())
    """

    def __init__(self) -> None:
        self._stages: list[StageResult] = []
        self._start: float | None = None
        self._current_stage: str | None = None
        self._stage_start: float | None = None

    class _StageContext:
        """Context manager for an individual stage."""

        def __init__(self, timer: StageTimer, name: str) -> None:
            self._timer = timer
            self._name = name

        def __enter__(self) -> StageContext:
            self._timer._current_stage = self._name
            self._timer._stage_start = time.perf_counter()
            return self  # type: ignore

        def __exit__(self, *exc: Any) -> None:
            if self._timer._stage_start is not None:
                duration_ms = (time.perf_counter() - self._timer._stage_start) * 1000
                self._timer._stages.append(
                    StageResult(name=self._name, duration_ms=round(duration_ms, 2))
                )
                self._timer._current_stage = None
                self._timer._stage_start = None

    def stage(self, name: str) -> _StageContext:
        """Create a timed context for a pipeline stage.

        Args:
            name: Human-readable stage name.

        Returns:
            Context manager that records the stage duration.
        """
        return self._StageContext(self, name)

    @property
    def stages(self) -> list[StageResult]:
        """Get all recorded stages."""
        return list(self._stages)

    @property
    def total_ms(self) -> float:
        """Total elapsed time across all stages in milliseconds."""
        return round(sum(s.duration_ms for s in self._stages), 2)

    def summary(self) -> dict[str, Any]:
        """Get a summary dict of all stages and total time."""
        return {
            "stages": {s.name: s.duration_ms for s in self._stages},
            "total_ms": self.total_ms,
            "stage_count": len(self._stages),
        }

    def log_summary(self) -> None:
        """Log the timing summary."""
        summary = self.summary()
        logger.info(
            "pipeline_timing",
            total_ms=summary["total_ms"],
            stages=summary["stages"],
        )


# Type alias for the context used by external code
StageContext = StageTimer._StageContext


def timed(name: str | None = None) -> Callable:
    """Decorator that logs the execution time of a function.

    Args:
        name: Optional name for the timed operation. Defaults to function name.

    Returns:
        Decorated function that logs its execution time.
    """
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.info("operation_timed", operation=op_name, duration_ms=round(duration_ms, 2))
                return result
            except Exception:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error("operation_failed", operation=op_name, duration_ms=round(duration_ms, 2))
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.info("operation_timed", operation=op_name, duration_ms=round(duration_ms, 2))
                return result
            except Exception:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error("operation_failed", operation=op_name, duration_ms=round(duration_ms, 2))
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
