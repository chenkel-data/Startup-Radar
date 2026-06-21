from __future__ import annotations

from collections import defaultdict
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.tracing.constant import SpanAttributeKey

from app.core.config import Settings
from app.observability.llm_steps import (
    ORIGINAL_MODEL_ATTR,
    ORIGINAL_PROVIDER_ATTR,
    WORKFLOW_STEP_ATTR,
)
from app.observability.setup import is_enabled
from app.observability.traces import mlflow_experiment_id


_WORKFLOW_SPANS = {
    "extract_entities": "llm_extraction",
    "gleaning_pass": "llm_gleaning",
    "review_entity_profile": "llm_profile_review",
    "curate_entity_profile": "llm_profile_curation",
}
_CURATION_STEPS = {"llm_profile_review", "llm_profile_curation"}
_HISTORICAL_SYNTHETIC_TOTAL_SPAN_NAME = "llm_cost_total"
_HISTORICAL_SYNTHETIC_TOTAL_ATTR = "startup_radar.synthetic_cost_total"
_TRACE_TAG_NAMES = {
    "llm_cost_usd": "cost.llm_total_usd",
    "llm_extraction_cost_usd": "cost.llm_extraction_usd",
    "llm_gleaning_cost_usd": "cost.llm_gleaning_usd",
    "llm_curation_cost_usd": "cost.llm_curation_usd",
    "llm_profile_review_cost_usd": "cost.llm_profile_review_usd",
    "llm_profile_curation_cost_usd": "cost.llm_profile_curation_usd",
}


def aggregate_llm_costs_for_job(
    *,
    settings: Settings,
    job_run_id: str,
    max_results: int = 5000,
) -> list[dict[str, Any]]:
    """Read MLflow's native LLM span costs and bucket them by workflow step."""
    if not is_enabled():
        return []
    experiment_id = mlflow_experiment_id(settings)
    if not experiment_id:
        return []
    try:
        mlflow.flush_trace_async_logging()
        traces = mlflow.search_traces(
            experiment_ids=[experiment_id],
            filter_string=f"tags.job_run_id = '{_escape_filter_value(job_run_id)}'",
            max_results=max_results,
            include_spans=True,
            return_type="list",
            flush=True,
        )
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for trace in traces:
        rows.extend(_llm_cost_rows_from_trace(trace))
    _tag_traces_with_costs(rows)
    return rows


def llm_cost_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not rows:
        return metrics

    by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[str(row.get("workflow_step") or "unknown")].append(row)

    _add_group_metrics(metrics, "llm", rows)
    for step, step_rows in by_step.items():
        _add_group_metrics(metrics, step, step_rows)

    curation_rows = [row for row in rows if str(row.get("workflow_step") or "") in _CURATION_STEPS]
    if curation_rows:
        _add_group_metrics(metrics, "llm_curation", curation_rows)
    return metrics


def _llm_cost_rows_from_trace(trace: Any) -> list[dict[str, Any]]:
    spans = list(getattr(getattr(trace, "data", None), "spans", []) or [])
    spans_by_id = {span.span_id: span for span in spans}
    rows: list[dict[str, Any]] = []

    for span in spans:
        if _is_synthetic_total_span(span):
            continue
        cost = getattr(span, "llm_cost", None)
        if not cost:
            continue
        workflow_step = _workflow_step_for_span(span, spans_by_id)
        if workflow_step is None:
            continue
        usage = _token_usage(span)
        model_display = getattr(span, "model_name", None) or _span_attribute(
            span,
            SpanAttributeKey.MODEL,
        )
        provider_display = getattr(span, "model_provider", None) or _span_attribute(
            span,
            SpanAttributeKey.MODEL_PROVIDER,
        )
        rows.append(
            {
                **_trace_context(trace),
                **_ancestor_context(span, spans_by_id),
                "span_id": span.span_id,
                "span_name": span.name,
                "workflow_step": workflow_step,
                "workflow_step_label": _span_attribute(span, WORKFLOW_STEP_ATTR),
                "model": _span_attribute(span, ORIGINAL_MODEL_ATTR) or model_display,
                "model_display": model_display,
                "provider": _span_attribute(span, ORIGINAL_PROVIDER_ATTR) or provider_display,
                "provider_display": provider_display,
                **usage,
                "input_cost_usd": _round_usd(cost.get("input_cost")),
                "output_cost_usd": _round_usd(cost.get("output_cost")),
                "total_cost_usd": _round_usd(cost.get("total_cost")),
                "cost_source": "mlflow_autolog_span",
            }
        )
    return rows


def _workflow_step_for_span(span: Any, spans_by_id: dict[str, Any]) -> str | None:
    current = span
    while current is not None:
        if workflow_step := _WORKFLOW_SPANS.get(current.name):
            return workflow_step
        parent_id = current.parent_id
        current = spans_by_id.get(parent_id) if parent_id else None
    return None


def _is_synthetic_total_span(span: Any) -> bool:
    if getattr(span, "name", None) == _HISTORICAL_SYNTHETIC_TOTAL_SPAN_NAME:
        return True
    return bool(_span_attribute(span, _HISTORICAL_SYNTHETIC_TOTAL_ATTR))


def _ancestor_context(span: Any, spans_by_id: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    current = span
    while current is not None:
        inputs = getattr(current, "inputs", None)
        if isinstance(inputs, dict):
            for key in (
                "article_url",
                "article_title",
                "source_name",
                "entity_id",
                "entity_name",
                "entity_type",
            ):
                if key in inputs and key not in context:
                    context[key] = inputs[key]
        parent_id = current.parent_id
        current = spans_by_id.get(parent_id) if parent_id else None
    return context


def _trace_context(trace: Any) -> dict[str, Any]:
    info = getattr(trace, "info", None)
    tags = getattr(info, "tags", {}) or {}
    return {
        "trace_id": getattr(info, "trace_id", None),
        "article_url": tags.get("article_url"),
        "source_name": tags.get("source_name"),
        "job_run_id": tags.get("job_run_id"),
    }


def _token_usage(span: Any) -> dict[str, int]:
    usage = _span_attribute(span, SpanAttributeKey.CHAT_USAGE)
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    row = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        value = int(usage.get(key) or 0)
        if value:
            row[key] = value
    return row


def _tag_traces_with_costs(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if trace_id := row.get("trace_id"):
            by_trace[str(trace_id)].append(row)

    client = MlflowClient()
    for trace_id, trace_rows in by_trace.items():
        metrics = llm_cost_metrics(trace_rows)
        for metric_key, tag_name in _TRACE_TAG_NAMES.items():
            if metric_key not in metrics:
                continue
            try:
                client.set_trace_tag(trace_id, tag_name, f"{metrics[metric_key]:.6f}")
            except Exception:
                pass


def _add_group_metrics(
    metrics: dict[str, float],
    prefix: str,
    rows: list[dict[str, Any]],
) -> None:
    metrics[f"{prefix}_calls"] = float(len(rows))
    metrics[f"{prefix}_input_tokens"] = float(sum(_int(row, "input_tokens") for row in rows))
    metrics[f"{prefix}_output_tokens"] = float(sum(_int(row, "output_tokens") for row in rows))
    metrics[f"{prefix}_total_tokens"] = float(sum(_int(row, "total_tokens") for row in rows))
    metrics[f"{prefix}_cost_usd"] = round(
        sum(float(row.get("total_cost_usd") or 0.0) for row in rows),
        10,
    )


def _int(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def _span_attribute(span: Any, key: str) -> Any:
    try:
        return span.get_attribute(key)
    except Exception:
        return None


def _round_usd(value: float | None) -> float:
    return round(float(value or 0.0), 10)


def _escape_filter_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")
