"""Deterministic scorers attached to per-article traces as feedback.

These run synchronously inside the extraction pipeline (no LLM cost) and
land on the trace's **Assessments** tab in the MLflow UI alongside any
human feedback. They cover the "did the LLM produce *anything sane*"
basics; deeper correctness checks live in
`scripts/run_evaluation.py` via `mlflow.genai.evaluate()`.

Each helper logs a single `mlflow.log_feedback` call. Failures are
swallowed — observability never breaks ingestion.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from app.core.logging import get_logger
from app.models.extraction import ExtractionResult
from app.observability.setup import is_enabled

_logger = get_logger("observability.scorers")
_CODE_SOURCE = AssessmentSource(
    source_type=AssessmentSourceType.CODE,
    source_id="online_deterministic_scorer",
)


def attach_extraction_scores(
    *,
    trace_id: str,
    extraction: ExtractionResult,
    duration_ms: float,
    raw_status_counts: dict[str, int] | None = None,
    latency_budget_ms: float = 30_000.0,
) -> None:
    """Score one extraction and attach four feedback rows to the trace.

    Scores:
    - ``extraction_schema_valid``  — the parsed Pydantic round-tripped
    - ``unsure_candidates_quarantined`` — graph-bound output contains no unsure facts
    - ``admitted_fact_nonzero`` — evidence gate admitted at least one entity or relation
    - ``latency_under_budget`` — root span ran in under the budget
    """
    if not is_enabled() or not trace_id:
        return

    all_entities = [
        entity
        for bucket in (
            extraction.startups,
            extraction.investors,
            extraction.people,
            extraction.topics,
            extraction.companies,
        )
        for entity in bucket
    ]
    facts = [*all_entities, *extraction.relationships]
    unsure_in_admitted = sum(item.evidence_status == "unsure" for item in facts)
    raw_unsure = sum(
        (raw_status_counts or {}).get(key, 0) for key in ("entity_unsure", "relationship_unsure")
    )
    raw_defaulted = sum(
        (raw_status_counts or {}).get(key, 0)
        for key in ("entity_status_defaulted", "relationship_status_defaulted")
    )

    scores: dict[str, tuple[Any, str]] = {  # type: ignore[type-arg]
        "extraction_schema_valid": (True, "Pydantic model_validate succeeded"),
        "unsure_candidates_quarantined": (
            unsure_in_admitted == 0,
            f"raw_unsure={raw_unsure} raw_defaulted={raw_defaulted} admitted_unsure={unsure_in_admitted}",
        ),
        "admitted_fact_nonzero": (
            bool(facts),
            f"admitted_entities={extraction.entity_count()} admitted_relationships={len(extraction.relationships)}",
        ),
        "latency_under_budget": (
            duration_ms <= latency_budget_ms,
            f"duration_ms={duration_ms:.1f} budget_ms={latency_budget_ms:.0f}",
        ),
    }

    for name, (value, rationale) in scores.items():
        try:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=name,
                value=value,
                rationale=rationale,
                source=_CODE_SOURCE,
            )
        except Exception as exc:
            _logger.debug(
                "scorer_attach_failed",
                extra={
                    "event": "observability",
                    "workflow_step": "scorer",
                    "error": str(exc),
                    "detail": f"trace_id={trace_id} name={name}",
                },
            )
