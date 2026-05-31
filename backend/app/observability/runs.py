"""Per-ingestion MLflow Run wrapper.

One HTTP `POST /ingest` opens one MLflow Run via `IngestionRun(...)`.
Per-article traces (created by `@mlflow.trace` decorators inside the
service code) carry a `job_run_id` tag pointing back here so a single
filter — `tags.job_run_id = '<id>'` — returns every trace for the batch.

The `RunHandle` exposes a tiny API the orchestrator uses to record:
- aggregate metrics at end of run (`record_metric`, `add_token_usage`)
- JSON / JSONL / markdown artifacts (`add_artifact`)
- arbitrary tags / params (`add_tag`, `add_param`)

When MLflow is disabled, `IngestionRun` yields a `_NullHandle` so call
sites never branch.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow import MlflowClient

from app.core.logging import get_logger
from app.observability.setup import is_enabled

_logger = get_logger("observability.runs")


@dataclass
class RunHandle:
    """Stateful handle returned by `IngestionRun`.

    Metrics and artifacts are buffered locally and flushed once at Run close
    (single `log_metrics` call, one `log_text` per artifact). This avoids
    network chatter under high concurrency and gives us a clean failure mode:
    if the Run crashes, the partial buffer is still flushed in the `finally`.
    """

    run_id: str
    job_run_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[tuple[str, str]] = field(default_factory=list)

    # Cumulative token / cost rollup across all article traces in this job.
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def record_metric(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)

    def record_metrics(self, values: dict[str, float]) -> None:
        for k, v in values.items():
            self.record_metric(k, v)

    def add_artifact(self, path: str, body: str) -> None:
        self.artifacts.append((path, body))

    def add_json_artifact(self, path: str, payload: Any) -> None:
        self.artifacts.append(
            (path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        )

    def add_jsonl_artifact(self, path: str, rows: list[dict[str, Any]]) -> None:
        body = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows)
        self.artifacts.append((path, body))

    def add_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.total_input_tokens += int(input_tokens or 0)
        self.total_output_tokens += int(output_tokens or 0)
        self.total_cost_usd += float(cost_usd or 0.0)

    def add_tag(self, key: str, value: str) -> None:
        try:
            mlflow.set_tag(key, value)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.debug(
                "mlflow_set_tag_failed",
                extra={
                    "event": "observability",
                    "workflow_step": "set_tag",
                    "error": str(exc),
                    "detail": f"{key}={value}",
                },
            )

    def add_param(self, key: str, value: Any) -> None:
        try:
            mlflow.log_param(key, value)
        except Exception as exc:  # pragma: no cover
            _logger.debug(
                "mlflow_log_param_failed",
                extra={
                    "event": "observability",
                    "workflow_step": "log_param",
                    "error": str(exc),
                    "detail": f"{key}={value}",
                },
            )


class _NullHandle(RunHandle):
    """No-op handle returned when MLflow is disabled.

    Implements the same surface as `RunHandle` so call sites stay branch-free.
    """

    def __init__(self) -> None:
        super().__init__(run_id="disabled", job_run_id="disabled")

    def add_tag(self, key: str, value: str) -> None:  # pragma: no cover
        return

    def add_param(self, key: str, value: Any) -> None:  # pragma: no cover
        return


@asynccontextmanager
async def IngestionRun(
    *,
    job_run_id: str,
    source_name: str,
    params: dict[str, Any],
    tags: dict[str, str] | None = None,
) -> AsyncIterator[RunHandle]:
    """Open one MLflow Run for one ingestion job.

    Args:
        job_run_id: stable id propagated to each child trace via the
            `job_run_id` tag (we deliberately don't use `mlflow.parentRunId`
            since traces aren't runs).
        source_name: e.g. ``deutsche-startups.de``; used to build the Run name.
        params: immutable Run params (max_pages, model, thresholds, ...).
        tags: extra Run tags (env, git_sha, app version, ...).
    """
    if not is_enabled():
        yield _NullHandle()
        return

    run_name = f"ingest-{source_name}-{job_run_id[:8]}"
    merged_tags = {"job_run_id": job_run_id, "source_name": source_name}
    if tags:
        merged_tags.update({k: str(v) for k, v in tags.items() if v is not None})

    try:
        # Call start_run directly so thread-local active-run is set on this thread,
        # making all subsequent mlflow.log_* calls with run_id= work correctly.
        run_ctx = mlflow.start_run(run_name=run_name, tags=merged_tags)
    except Exception as exc:
        _logger.warning(
            "mlflow_start_run_failed",
            extra={
                "event": "observability",
                "workflow_step": "ingestion_run",
                "error": str(exc),
                "detail": f"run_name={run_name}",
            },
        )
        yield _NullHandle()
        return

    with run_ctx as run:
        run_id = run.info.run_id
        try:
            await asyncio.to_thread(
                mlflow.log_params,
                {k: _stringify_param(v) for k, v in params.items()},
                run_id=run_id,
            )
        except Exception as exc:
            _logger.debug(
                "mlflow_log_params_failed",
                extra={
                    "event": "observability",
                    "workflow_step": "ingestion_run",
                    "error": str(exc),
                },
            )

        handle = RunHandle(run_id=run_id, job_run_id=job_run_id)
        status_tag = "FINISHED"
        try:
            yield handle
        except Exception:
            status_tag = "FAILED"
            await asyncio.to_thread(MlflowClient().set_tag, run_id, "ingestion_status", "error")
            raise
        finally:
            # Roll up the token / cost summary collected from per-article traces.
            if handle.total_input_tokens or handle.total_output_tokens:
                handle.record_metric("total_input_tokens", handle.total_input_tokens)
                handle.record_metric("total_output_tokens", handle.total_output_tokens)
                handle.record_metric(
                    "total_tokens",
                    handle.total_input_tokens + handle.total_output_tokens,
                )
            if handle.total_cost_usd:
                handle.record_metric("total_cost_usd", round(handle.total_cost_usd, 6))
            if handle.metrics:
                try:
                    await asyncio.to_thread(mlflow.log_metrics, handle.metrics, run_id=run_id)
                except Exception as exc:  # pragma: no cover
                    _logger.warning(
                        "mlflow_log_metrics_failed",
                        extra={
                            "event": "observability",
                            "workflow_step": "ingestion_run_close",
                            "error": str(exc),
                        },
                    )
            for path, body in handle.artifacts:
                try:
                    await asyncio.to_thread(mlflow.log_text, body, path, run_id=run_id)
                except Exception as exc:  # pragma: no cover
                    _logger.warning(
                        "mlflow_log_text_failed",
                        extra={
                            "event": "observability",
                            "workflow_step": "ingestion_run_close",
                            "error": str(exc),
                            "detail": f"path={path}",
                        },
                    )
            if status_tag == "FINISHED":
                await asyncio.to_thread(MlflowClient().set_tag, run_id, "ingestion_status", "ok")


def _stringify_param(value: Any) -> str:
    """MLflow params must be strings; trim to the documented 500-char ceiling."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)[:500]
    return str(value)[:500]
