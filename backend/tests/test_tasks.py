import asyncio

import pytest

from app.services.tasks import TaskConflictError, TaskManager


@pytest.mark.asyncio
async def test_ingestion_and_curation_tasks_are_mutually_exclusive() -> None:
    manager = TaskManager()
    release = asyncio.Event()

    async def blocked_work(task_id: str) -> dict:
        await release.wait()
        return {"task_id": task_id}

    ingest = manager.start(
        "ingest",
        blocked_work,
        exclusive_with={"ingest", "curation"},
    )

    with pytest.raises(TaskConflictError) as conflict:
        manager.start(
            "curation",
            blocked_work,
            exclusive_with={"ingest", "curation"},
        )

    assert conflict.value.active_task.task_id == ingest.task_id
    release.set()
    await manager._tasks[ingest.task_id]

    curation = manager.start(
        "curation",
        blocked_work,
        exclusive_with={"ingest", "curation"},
    )
    await manager._tasks[curation.task_id]

    assert manager.get(ingest.task_id).status == "succeeded"
    assert manager.get(curation.task_id).status == "succeeded"
