from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any


@asynccontextmanager
async def timed_step(logger, event: str, **fields: Any) -> AsyncIterator[None]:
    start = perf_counter()
    workflow_step = fields.pop("workflow_step", event)
    start_summary = fields.pop("start_summary", None)
    base_fields = {"event": event, "workflow_step": workflow_step, **fields}
    start_fields = base_fields
    if start_summary is not None:
        start_fields = {"summary": start_summary, **base_fields}
    logger.info("step_started", extra=start_fields)
    try:
        yield
    except Exception as exc:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        logger.exception(
            "step_failed",
            extra={"duration_ms": duration_ms, "error": str(exc), **base_fields},
        )
        raise
    else:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        logger.info(
            "step_completed",
            extra={"duration_ms": duration_ms, **base_fields},
        )
