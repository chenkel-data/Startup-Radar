from __future__ import annotations

import mlflow


def current_trace_id() -> str | None:
    """Best-effort lookup of the currently active MLflow trace id."""
    try:
        span = mlflow.get_current_active_span()
        return span.trace_id if span is not None else None
    except Exception:
        return None
