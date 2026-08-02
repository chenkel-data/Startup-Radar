import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from app.core.logging import get_logger
from app.models.extraction import TaskStatus


WorkCallable = Callable[..., Awaitable[Any]]


class TaskConflictError(RuntimeError):
    def __init__(self, active_task: TaskStatus):
        self.active_task = active_task
        super().__init__(
            f"{active_task.name.capitalize()} task {active_task.task_id} is already "
            f"{active_task.status}"
        )


class TaskManager:
    def __init__(self):
        self._statuses: dict[str, TaskStatus] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self.logger = get_logger("tasks")

    def start(
        self,
        name: str,
        work: WorkCallable,
        *,
        exclusive_with: set[str] | frozenset[str] | None = None,
    ) -> TaskStatus:
        exclusive_names = exclusive_with or set()
        for active_status in self._statuses.values():
            if active_status.name in exclusive_names and active_status.status in {
                "queued",
                "running",
            }:
                raise TaskConflictError(active_status)

        task_id = str(uuid.uuid4())
        status = TaskStatus(
            task_id=task_id,
            name=name,
            status="queued",
            created_at=datetime.now(UTC),
        )
        self._statuses[task_id] = status
        self._tasks[task_id] = asyncio.create_task(self._run(task_id, work))
        self.logger.info(
            "task_queued",
            extra={"event": "task", "workflow_step": name, "task_id": task_id},
        )
        return status

    def get(self, task_id: str) -> TaskStatus | None:
        return self._statuses.get(task_id)

    async def _run(self, task_id: str, work: WorkCallable) -> None:
        status = self._statuses[task_id]
        status.status = "running"
        status.started_at = datetime.now(UTC)
        self.logger.info(
            "task_started",
            extra={"event": "task", "workflow_step": status.name, "task_id": task_id},
        )
        try:
            result = await _invoke_work(work, task_id)
            status.status = "succeeded"
            status.result = _serialize_result(result)
            extra = {
                "event": "task",
                "workflow_step": status.name,
                "task_id": task_id,
            }
            if status.name != "ingest":
                extra["detail"] = _result_summary(status.result)
            self.logger.info(
                "task_succeeded",
                extra=extra,
            )
        except Exception as exc:
            status.status = "failed"
            status.error = str(exc)
            self.logger.exception(
                "task_failed",
                extra={
                    "event": "task",
                    "workflow_step": status.name,
                    "task_id": task_id,
                    "error": str(exc),
                },
            )
        finally:
            status.completed_at = datetime.now(UTC)


async def _invoke_work(work: WorkCallable, task_id: str) -> Any:
    """Call work with task_id when its signature accepts it, else without."""
    try:
        sig = inspect.signature(work)
        accepts_task_id = "task_id" in sig.parameters
    except (ValueError, TypeError):
        accepts_task_id = True
    if accepts_task_id:
        return await work(task_id=task_id)
    return await work()


def _serialize_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, dict):
        return result
    return {"value": result}


def _result_summary(result: dict[str, Any] | None) -> str:
    if not result:
        return "no result payload"
    keys = [
        "articles_found",
        "articles_cached",
        "articles_scraped",
        "articles_skipped",
        "articles_processed",
        "articles_failed",
        "entities_extracted",
        "relationships_created",
        "candidate_entities",
        "review_calls",
        "rewrite_calls",
        "cost_usd",
        "duration_ms",
    ]
    parts = [f"{key}={result[key]}" for key in keys if key in result]
    return ", ".join(parts) if parts else "result ready"
