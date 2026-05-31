"""MLflow-backed observability helpers for the Startup Radar pipeline.

This package exposes the app-level observability API: MLflow setup, ingestion
run tracking, feedback logging, and enabled/disabled checks. Lower-level spans,
traces, and prompt operations use MLflow's native API directly to avoid adding
wrapper indirection.
"""

from app.observability.feedback import log_extraction_feedback
from app.observability.runs import IngestionRun, RunHandle
from app.observability.setup import init_mlflow, is_enabled

__all__ = [
    "IngestionRun",
    "RunHandle",
    "init_mlflow",
    "is_enabled",
    "log_extraction_feedback",
]
