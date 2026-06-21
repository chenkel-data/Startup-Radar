from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import calculate_cost_by_model_and_token_usage


LLM_STEP_EXTRACTION = "extraction"
LLM_STEP_GLEANING = "gleaning"
LLM_STEP_PROFILE_REVIEW = "profile_review"
LLM_STEP_PROFILE_CURATION = "profile_curation"

WORKFLOW_STEP_ATTR = "startup_radar.workflow_step"
ORIGINAL_MODEL_ATTR = "startup_radar.openai_model"
ORIGINAL_PROVIDER_ATTR = "startup_radar.openai_provider"

_current_llm_step: ContextVar[str | None] = ContextVar(
    "startup_radar_llm_workflow_step",
    default=None,
)


@contextmanager
def llm_workflow_step(step: str) -> Iterator[None]:
    token = _current_llm_step.set(step)
    try:
        yield
    finally:
        _current_llm_step.reset(token)


def label_llm_span_with_workflow_step(span: Any) -> None:
    """Expose the current workflow step through MLflow's model dimension.

    MLflow's GenAI Overview can group cost over time by model/provider, but not
    by arbitrary span attributes. For local tracking servers, MLflow computes
    cost server-side after span processors run, so this processor first attaches
    MLflow's own cost value using the real model name. It then relabels the
    model dimension for step charts and the provider dimension for a per-model
    total chart. Both groupings use the same real OpenAI span cost rows.
    """
    step = _current_llm_step.get()
    if not step:
        return

    model = _span_attr(span, SpanAttributeKey.MODEL)
    if not isinstance(model, str) or not model.strip():
        return

    original_model = _span_attr(span, ORIGINAL_MODEL_ATTR)
    base_model = original_model if isinstance(original_model, str) else model
    base_model = base_model.strip()
    if not base_model:
        return

    provider = _provider_for_span(span)
    _set_cost_if_missing(span, base_model, provider)
    span.set_attribute(WORKFLOW_STEP_ATTR, step)
    span.set_attribute(ORIGINAL_MODEL_ATTR, base_model)
    span.set_attribute(ORIGINAL_PROVIDER_ATTR, provider)
    span.set_attribute(SpanAttributeKey.MODEL, f"{base_model} / {step}")
    span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, f"{base_model} / total")


def _set_cost_if_missing(span: Any, model: str, provider: str) -> None:
    if _span_attr(span, SpanAttributeKey.LLM_COST):
        return
    usage = _span_attr(span, SpanAttributeKey.CHAT_USAGE)
    if not isinstance(usage, dict):
        return
    cost = calculate_cost_by_model_and_token_usage(model, usage, provider)
    if cost:
        span.set_attribute(SpanAttributeKey.LLM_COST, cost)


def _provider_for_span(span: Any) -> str:
    provider = _span_attr(span, ORIGINAL_PROVIDER_ATTR) or _span_attr(
        span,
        SpanAttributeKey.MODEL_PROVIDER,
    )
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    return "openai"


def _span_attr(span: Any, key: str) -> Any:
    try:
        return span.get_attribute(key)
    except Exception:
        return None
