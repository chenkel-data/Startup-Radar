from __future__ import annotations

from urllib.parse import quote

import mlflow

from app.core.config import Settings


def mlflow_trace_link(settings: Settings, trace_id: str | None) -> tuple[str | None, str | None]:
    if not trace_id:
        return None, None
    experiment_id = mlflow_experiment_id(settings)
    if not experiment_id:
        return None, None
    base_url = (settings.mlflow_public_url or settings.mlflow_tracking_uri).rstrip("/")
    return (
        f"{base_url}/#/experiments/{quote(experiment_id, safe='')}/traces/{quote(trace_id, safe='')}",
        experiment_id,
    )


def mlflow_experiment_id(settings: Settings) -> str | None:
    try:
        experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    except Exception:  # pragma: no cover - depends on the configured MLflow service
        return None
    return str(experiment.experiment_id) if experiment else None
