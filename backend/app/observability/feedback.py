"""Attach human feedback to a trace.

Wraps `mlflow.log_feedback` so the API route stays a one-liner. Any
failure here is non-fatal — feedback is operator metadata, not data we
return to the user.
"""

from __future__ import annotations

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from app.core.logging import get_logger
from app.observability.setup import is_enabled

_logger = get_logger("observability.feedback")


def log_extraction_feedback(
    *,
    trace_id: str,
    label: str,
    target: str = "overall",
    comment: str | None = None,
    reviewer: str | None = None,
) -> bool:
    """Record one human assessment on a trace.

    Args:
        trace_id: MLflow trace id (the one shown in the UI URL).
        label: short verdict, e.g. ``good``, ``bad``, ``wrong_merge``.
        target: which part of the pipeline this verdict is about
            (``extraction`` | ``resolution`` | ``overall``).
        comment: optional free-text rationale.
        reviewer: optional human identifier; defaults to ``anonymous``.

    Returns:
        True iff the feedback reached MLflow.
    """
    if not is_enabled():
        _logger.info(
            "feedback_skipped_mlflow_disabled",
            extra={
                "event": "observability",
                "workflow_step": "feedback",
                "detail": f"trace_id={trace_id} label={label}",
            },
        )
        return False
    try:
        mlflow.log_feedback(
            trace_id=trace_id,
            name=f"human_review_{target}",
            value=label,
            rationale=comment,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id=reviewer or "anonymous",
            ),
        )
        return True
    except Exception as exc:
        _logger.warning(
            "mlflow_log_feedback_failed",
            extra={
                "event": "observability",
                "workflow_step": "feedback",
                "error": str(exc),
                "detail": f"trace_id={trace_id} target={target}",
            },
        )
        return False
