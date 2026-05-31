"""LLM-driven extraction.

Tracing model:
  - The OpenAI Chat Completions call is auto-instrumented as a
    `CHAT_MODEL` span by `mlflow.openai.autolog()` (set up in
    `app.observability.setup`). That span captures the full prompt,
    response, tokens, and (when supported) cost.
  - `_extract_article_traced` is decorated with `@mlflow.trace` so its
    retry loop, the LLM call, and the response parsing all hang off a
    single `extract_entities` span inside the parent `process_article` trace.
  - `_parse_extraction_response` is its own PARSER span. We
    `set_outputs(extraction.model_dump(...))` there so the trace's
    detail view shows the full structured `ExtractionResult` JSON —
    not just the raw model text.
"""

import asyncio
from time import perf_counter
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import SpanType
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.extraction import (
    ArticleIn,
    EntityType,
    EvidenceStatus,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    RawEntityRecord,
    RawExtractionResult,
    RawRelationshipRecord,
    SourceAttribution,
)
from app.prompts.extraction import (
    ARTICLE_TEXT_CHAR_LIMIT,
    COMPLETION_DELIMITER,
    TUPLE_DELIMITER,
    build_extraction_system_prompt,
    build_extraction_user_prompt,
    build_gleaning_prompt,
)
from app.services.extraction_utils import (
    _normalize_description,
    _normalize_entity_name,
    _normalize_entity_type,
    _normalize_keywords,
)
from app.observability.context import current_trace_id as _current_trace_id
from app.services.progress import (
    article_fields,
    extraction_summary,
)


MISSING_OPENAI_API_KEY_MESSAGE = (
    "OPENAI_API_KEY is required for article extraction. "
    "Set OPENAI_API_KEY in .env and restart the backend."
)


class LLMExtractionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("extractor")
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)
        self._client = (
            AsyncOpenAI(
                api_key=settings.openai_api_key,
                max_retries=settings.openai_max_retries,
            )
            if settings.openai_api_key
            else None
        )
        self._extraction_prompt = None
        self._gleaning_prompt = None
        if settings.mlflow_use_prompt_registry:
            self._extraction_prompt = self._load_prompt_at_startup(
                settings.mlflow_prompt_extraction_uri
            )
            self._gleaning_prompt = self._load_prompt_at_startup(
                settings.mlflow_prompt_gleaning_uri
            )

    def _load_prompt_at_startup(self, uri: str) -> Any:
        """Load a prompt from the MLflow registry once at startup and cache it.

        Returns the prompt object on success, None on failure (the caller falls
        back to the local builder so the app starts regardless).
        """
        try:
            prompt = mlflow.genai.load_prompt(uri)
            self.logger.info(
                "mlflow_prompt_loaded",
                extra={
                    "event": "startup",
                    "workflow_step": "prompt_cache",
                    "detail": f"uri={uri}; version={getattr(prompt, 'version', '?')}",
                },
            )
            return prompt
        except Exception as exc:
            self.logger.warning(
                "mlflow_prompt_load_failed_at_startup",
                extra={
                    "event": "startup",
                    "workflow_step": "prompt_cache",
                    "error": str(exc),
                    "detail": f"uri={uri}; will use local builder as fallback",
                },
            )
            return None

    def _render_prompt_from_cache(
        self, cached_prompt: Any, variables: dict[str, str], uri: str
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Format a cached MLflow prompt object with the given variables.

        Returns (system_prompt, user_prompt, metadata) on success, None on any
        failure so the caller can fall through to the local builder.
        """
        if cached_prompt is None:
            return None
        try:
            messages = cached_prompt.format(**variables)
            system_prompt, user_prompt = _messages_to_system_user(messages)
            _link_prompt_to_current_trace(cached_prompt)
            return system_prompt, user_prompt, _registry_prompt_metadata(cached_prompt, uri)
        except Exception as exc:
            self.logger.warning(
                "mlflow_prompt_render_failed",
                extra={
                    "event": "observability",
                    "workflow_step": "prompt_registry",
                    "error": str(exc),
                    "detail": f"uri={uri}; falling back to local builder",
                },
            )
            return None

    async def extract_article(
        self,
        article: ArticleIn,
        *,
        article_index: int | None = None,
        article_total: int | None = None,
    ) -> tuple[ExtractionResult, dict[str, Any]]:
        """Extract entities for one article.

        Returns ``(extraction, metadata)`` where ``metadata`` carries the
        MLflow ``trace_id``, attempts, latency, and model — handy for
        downstream artifact rows.
        """
        async with self._semaphore:
            return await self._extract_article_traced(
                article,
                article_index=article_index,
                article_total=article_total,
            )

    @mlflow.trace(name="extract_entities", span_type=SpanType.CHAIN)
    async def _extract_article_traced(
        self,
        article: ArticleIn,
        *,
        article_index: int | None = None,
        article_total: int | None = None,
    ) -> tuple[ExtractionResult, dict[str, Any]]:
        article_log_fields = article_fields(
            article,
            article_index=article_index,
            article_total=article_total,
        )
        self.logger.debug(
            "article_extraction_requested",
            extra={
                "event": "extraction",
                "workflow_step": "llm",
                "mode": "openai",
                "model": self.settings.openai_model,
                "detail": f"text_chars={len(article.text)}, tags={len(article.tags)}",
                **article_log_fields,
            },
        )
        if not self._client:
            self.logger.error(
                "llm_extraction_missing_api_key",
                extra={
                    "event": "extraction",
                    "workflow_step": "llm_configuration",
                    "error": MISSING_OPENAI_API_KEY_MESSAGE,
                    **article_log_fields,
                },
            )
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)

        system_prompt, user_prompt, prompt_metadata = self._render_extraction_prompts(article)
        _annotate_extract_span(article, prompt_metadata)

        started = perf_counter()
        trace_id = _current_trace_id()
        metadata: dict[str, Any] = {
            "trace_id": trace_id,
            "model": self.settings.openai_model,
            "max_attempts": self.settings.llm_retry_attempts,
            **prompt_metadata,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.settings.llm_retry_attempts + 1):
            self.logger.info(
                "llm_request_started",
                extra={
                    "event": "extraction",
                    "workflow_step": "llm_request",
                    "mode": "openai",
                    "model": self.settings.openai_model,
                    "attempt_index": attempt,
                    "attempt_total": self.settings.llm_retry_attempts,
                    "detail": f"text_chars={len(article.text)}, tags={len(article.tags)}",
                    **article_log_fields,
                },
            )
            try:
                result, raw_content, usage, extraction_audit = await self._extract_with_openai(
                    article,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                last_error = exc
                if attempt == self.settings.llm_retry_attempts:
                    message = f"LLM extraction failed after {attempt} attempts: {exc}"
                    self.logger.error(
                        "llm_extraction_failed",
                        extra={
                            "event": "extraction",
                            "workflow_step": "llm_failure",
                            "url": article.url,
                            "error": message,
                            "attempt_index": attempt,
                            "attempt_total": self.settings.llm_retry_attempts,
                            "detail": "no retry left for this article",
                            **article_log_fields,
                        },
                    )
                    raise RuntimeError(message) from exc
                wait_seconds = min(2**attempt, 12)
                self.logger.warning(
                    "llm_extraction_retry",
                    extra={
                        "event": "extraction",
                        "workflow_step": "llm_retry",
                        "url": article.url,
                        "error": str(exc),
                        "attempt_index": attempt,
                        "attempt_total": self.settings.llm_retry_attempts,
                        "retry_delay_seconds": wait_seconds,
                        "detail": "LLM call failed; article will be retried",
                        **article_log_fields,
                    },
                )
                await asyncio.sleep(wait_seconds)
                continue

            latency_ms = round((perf_counter() - started) * 1000, 2)
            metadata.update(
                {
                    "attempts": attempt,
                    "latency_ms": latency_ms,
                    "raw_response_chars": len(raw_content),
                    "gleaning_passes_run": extraction_audit["gleaning_passes_run"],
                    "gleaning_total_changes": extraction_audit["total_changes"],
                    "gleaning_entities_added": extraction_audit["entities_added"],
                    "gleaning_entities_corrected": extraction_audit["entities_corrected"],
                    "gleaning_relationships_added": extraction_audit["relationships_added"],
                    "gleaning_relationships_corrected": extraction_audit["relationships_corrected"],
                }
            )
            if usage:
                metadata["input_tokens"] = usage.get("input_tokens", 0)
                metadata["output_tokens"] = usage.get("output_tokens", 0)
                metadata["total_tokens"] = usage.get("total_tokens", 0)
            self.logger.debug(
                "llm_extraction_completed",
                extra={
                    "event": "extraction",
                    "workflow_step": "llm_output",
                    "count": result.entity_count(),
                    "detail": extraction_summary(result),
                    "mode": "openai",
                    "model": self.settings.openai_model,
                    **article_log_fields,
                },
            )
            _attach_extraction_to_span(result, extraction_audit)
            return result, metadata

        # Defensive — loop should either return or raise above.
        raise RuntimeError(f"LLM extraction terminated unexpectedly: last error={last_error!r}")

    async def _call_llm_raw(
        self,
        system_prompt: str | None,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Call OpenAI and return the raw text content (no JSON parsing)."""
        if self._client is None:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)
        content, _ = await self._call_llm_raw_with_usage(
            system_prompt,
            user_prompt,
            history=history,
        )
        return content

    async def _call_llm_raw_with_usage(
        self,
        system_prompt: str | None,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, dict[str, int] | None]:
        """Call OpenAI and return the raw text content plus token usage."""
        if self._client is None:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        response = await self._client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            timeout=self.settings.llm_timeout_seconds,
        )
        return response.choices[0].message.content or "", _openai_usage(response)

    async def extract_with_gleaning(
        self,
        article: ArticleIn,
        *,
        max_gleaning: int | None = None,
    ) -> RawExtractionResult:
        """LightRAG-style extraction with optional gleaning passes."""
        if self._client is None:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)
        passes = max_gleaning if max_gleaning is not None else self.settings.llm_gleaning_passes
        system_prompt, user_prompt, _ = self._render_extraction_prompts(article)

        raw_content = await self._call_llm_raw(system_prompt, user_prompt)
        result = parse_extraction_output(raw_content)
        history: list[dict[str, str]] = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_content},
        ]

        for pass_index in range(1, passes + 1):
            gleaning_prompt, _ = self._render_gleaning_prompt()
            continuation = await self._call_llm_raw(system_prompt, gleaning_prompt, history=history)
            extra = parse_extraction_output(continuation)
            _merge_gleaning_pass(
                result,
                extra,
                pass_index=pass_index,
                raw_response_chars=len(continuation),
            )

            history.extend(
                [
                    {"role": "user", "content": gleaning_prompt},
                    {"role": "assistant", "content": continuation},
                ]
            )

            if COMPLETION_DELIMITER in continuation:
                break

        return result

    def _render_extraction_prompts(self, article: ArticleIn) -> tuple[str, str, dict[str, Any]]:
        """Render the system and user prompts sent to OpenAI."""
        vars_ = _article_prompt_vars(article)
        self.logger.debug(
            "extraction_prompt_vars",
            extra={
                "event": "extraction",
                "workflow_step": "prompt_render",
                "detail": f"text_chars_truncated={len(vars_['input_text'])}",
            },
        )
        cached = self._render_prompt_from_cache(
            self._extraction_prompt,
            vars_,
            self.settings.mlflow_prompt_extraction_uri,
        )
        if cached is not None:
            return cached
        return (
            build_extraction_system_prompt(),
            build_extraction_user_prompt(article),
            {"prompt_uri": None, "prompt_source": "local_fallback"},
        )

    def _render_gleaning_prompt(self) -> tuple[str, dict[str, Any]]:
        cached = self._render_prompt_from_cache(
            self._gleaning_prompt,
            {},
            self.settings.mlflow_prompt_gleaning_uri,
        )
        if cached is not None:
            _system_prompt, user_prompt, metadata = cached
            return user_prompt, metadata
        return build_gleaning_prompt(), {"prompt_uri": None, "prompt_source": "local_fallback"}

    async def _extract_with_openai(
        self,
        article: ArticleIn,
        *,
        user_prompt: str,
        system_prompt: str,
    ) -> tuple[ExtractionResult, str, dict[str, int] | None, dict[str, Any]]:
        """Run the extraction LLM call.

        Returns ``(result, raw_content, usage, audit)``. Token usage on the OpenAI
        span is captured automatically by ``mlflow.openai.autolog()``; we
        ALSO return it here so the orchestrator can roll it up into the
        per-job Run's ``total_*_tokens`` metrics.
        """
        if self._client is None:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)
        response = await self._client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self.settings.llm_timeout_seconds,
        )
        content = response.choices[0].message.content or ""
        usage = _openai_usage(response)
        raw = parse_extraction_output(content)
        raw_texts = [content]
        audit = _initial_extraction_audit(raw, configured_passes=self.settings.llm_gleaning_passes)

        if self.settings.llm_gleaning_passes > 0:
            (
                gleaned_raw,
                gleaning_texts,
                gleaning_usage,
                gleaning_audit,
            ) = await self._run_gleaning_passes(
                article=article,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                initial_raw_content=content,
                raw=raw,
            )
            raw = gleaned_raw
            raw_texts.extend(gleaning_texts)
            usage = _combine_usage(usage, gleaning_usage)
            audit = gleaning_audit

        combined_raw_text = "\n".join(raw_texts)
        result = _parse_extraction_response(raw, article, raw_text=combined_raw_text)
        self.logger.debug(
            "llm_output_received",
            extra={
                "event": "extraction",
                "workflow_step": "llm_output",
                "detail": extraction_summary(result),
                "gleaning_passes": audit["gleaning_passes_run"],
                "gleaning_corrections": audit["total_changes"],
                **article_fields(article),
            },
        )
        if self.settings.log_llm_raw_output:
            self.logger.info(
                "llm_raw_output_preview",
                extra={
                    "event": "extraction",
                    "workflow_step": "llm_output_raw",
                    "detail": _clip_text(
                        combined_raw_text,
                        self.settings.log_llm_preview_chars,
                    ),
                    **article_fields(article),
                },
            )
        return result, combined_raw_text, usage, audit

    async def _run_gleaning_passes(
        self,
        *,
        article: ArticleIn,
        system_prompt: str,
        user_prompt: str,
        initial_raw_content: str,
        raw: RawExtractionResult,
    ) -> tuple[RawExtractionResult, list[str], dict[str, int] | None, dict[str, Any]]:
        pass_total = self.settings.llm_gleaning_passes
        history: list[dict[str, str]] = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": initial_raw_content},
        ]
        raw_texts: list[str] = []
        usage_total: dict[str, int] | None = None
        pass_reports: list[dict[str, Any]] = []
        article_log_fields = article_fields(article)

        for pass_index in range(1, pass_total + 1):
            gleaning_prompt, prompt_metadata = self._render_gleaning_prompt()
            self.logger.info(
                "llm_gleaning_started",
                extra={
                    "event": "extraction",
                    "workflow_step": "llm_gleaning",
                    "mode": "openai",
                    "model": self.settings.openai_model,
                    "attempt_index": pass_index,
                    "attempt_total": pass_total,
                    "detail": (
                        f"reviewing extraction; entities={len(raw.entities)}, "
                        f"relationships={len(raw.relationships)}"
                    ),
                    **article_log_fields,
                },
            )
            with mlflow.start_span(name="gleaning_pass", span_type=SpanType.CHAIN) as span:
                try:
                    span.set_inputs(
                        {
                            "pass_index": pass_index,
                            "pass_total": pass_total,
                            "entities_before": len(raw.entities),
                            "relationships_before": len(raw.relationships),
                            **prompt_metadata,
                        }
                    )
                    span.set_attributes(
                        {
                            "pass_index": pass_index,
                            "pass_total": pass_total,
                            "entities_before": len(raw.entities),
                            "relationships_before": len(raw.relationships),
                            "article_url": article.url,
                            **prompt_metadata,
                        }
                    )
                except Exception:  # pragma: no cover
                    pass

                continuation, pass_usage = await self._call_llm_raw_with_usage(
                    system_prompt,
                    gleaning_prompt,
                    history=history,
                )
                raw_texts.append(continuation)
                usage_total = _combine_usage(usage_total, pass_usage)
                extra = parse_extraction_output(continuation)
                report = _merge_gleaning_pass(
                    raw,
                    extra,
                    pass_index=pass_index,
                    raw_response_chars=len(continuation),
                )
                pass_reports.append(report)
                try:
                    span.set_outputs(report)
                    span.set_attributes(
                        {
                            "raw_entities": report["raw_entities"],
                            "raw_relationships": report["raw_relationships"],
                            "added_entities": report["added_entities_count"],
                            "corrected_entities": report["corrected_entities_count"],
                            "added_relationships": report["added_relationships_count"],
                            "corrected_relationships": report["corrected_relationships_count"],
                            "total_changes": report["total_changes"],
                            "entities_after": report["entities_after"],
                            "relationships_after": report["relationships_after"],
                        }
                    )
                except Exception:  # pragma: no cover
                    pass

            self.logger.info(
                "llm_gleaning_reviewed",
                extra={
                    "event": "extraction",
                    "workflow_step": "llm_gleaning",
                    "mode": "openai",
                    "model": self.settings.openai_model,
                    "attempt_index": pass_index,
                    "attempt_total": pass_total,
                    "count": report["total_changes"],
                    "detail": _gleaning_report_detail(report),
                    **article_log_fields,
                },
            )

            history.extend(
                [
                    {"role": "user", "content": gleaning_prompt},
                    {"role": "assistant", "content": continuation},
                ]
            )
            if COMPLETION_DELIMITER in continuation:
                break

        audit = _summarize_gleaning_reports(
            pass_reports,
            configured_passes=pass_total,
            final_raw=raw,
        )
        span = mlflow.get_current_active_span()
        if span is not None:
            try:
                span.set_attributes(
                    {
                        "gleaning_configured_passes": pass_total,
                        "gleaning_passes_run": audit["gleaning_passes_run"],
                        "gleaning_total_changes": audit["total_changes"],
                        "gleaning_entities_added": audit["entities_added"],
                        "gleaning_entities_corrected": audit["entities_corrected"],
                        "gleaning_relationships_added": audit["relationships_added"],
                        "gleaning_relationships_corrected": audit["relationships_corrected"],
                    }
                )
            except Exception:  # pragma: no cover
                pass
        return raw, raw_texts, usage_total, audit


def _openai_usage(response: Any) -> dict[str, int] | None:
    """Extract token usage from an OpenAI ChatCompletion response.

    Returns a normalized ``{"input_tokens", "output_tokens", "total_tokens"}``
    dict, or ``None`` if the response did not carry usage info. The
    `mlflow.openai.autolog()` hook captures the same numbers on the span
    automatically; this is for the IngestionRun-level metric rollup.
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if prompt is None and completion is None and total is None:
        return None
    return {
        "input_tokens": int(prompt or 0),
        "output_tokens": int(completion or 0),
        "total_tokens": int(total or 0),
    }


def _combine_usage(
    left: dict[str, int] | None,
    right: dict[str, int] | None,
) -> dict[str, int] | None:
    if not left and not right:
        return None
    merged = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for usage in (left, right):
        if not usage:
            continue
        for key in merged:
            merged[key] += int(usage.get(key, 0) or 0)
    return merged


def _initial_extraction_audit(
    raw: RawExtractionResult,
    *,
    configured_passes: int,
) -> dict[str, Any]:
    return {
        "gleaning_configured_passes": configured_passes,
        "gleaning_passes_run": 0,
        "total_changes": 0,
        "entities_added": 0,
        "entities_corrected": 0,
        "relationships_added": 0,
        "relationships_corrected": 0,
        "final_raw_entities": len(raw.entities),
        "final_raw_relationships": len(raw.relationships),
        "passes": [],
    }


def _merge_gleaning_pass(
    base: RawExtractionResult,
    extra: RawExtractionResult,
    *,
    pass_index: int,
    raw_response_chars: int,
) -> dict[str, Any]:
    """Apply one gleaning response and return an auditable before/after report."""
    entities_before = len(base.entities)
    relationships_before = len(base.relationships)

    entity_by_name: dict[str, RawEntityRecord] = {
        _raw_entity_key(entity): entity for entity in base.entities
    }
    added_entities: list[dict[str, Any]] = []
    corrected_entities: list[dict[str, Any]] = []
    for entity in extra.entities:
        key = _raw_entity_key(entity)
        previous = entity_by_name.get(key)
        if previous is None:
            entity_by_name[key] = entity
            added_entities.append(_raw_entity_summary(entity))
        elif previous.model_dump() != entity.model_dump():
            entity_by_name[key] = entity
            corrected_entities.append(
                {
                    "name": entity.name,
                    "before": _raw_entity_summary(previous),
                    "after": _raw_entity_summary(entity),
                }
            )
    base.entities = list(entity_by_name.values())

    relationship_by_key: dict[tuple[str, str, str], RawRelationshipRecord] = {
        _raw_relationship_key(relationship): relationship for relationship in base.relationships
    }
    added_relationships: list[dict[str, Any]] = []
    corrected_relationships: list[dict[str, Any]] = []
    for relationship in extra.relationships:
        key = _raw_relationship_key(relationship)
        previous = relationship_by_key.get(key)
        if previous is None:
            relationship_by_key[key] = relationship
            added_relationships.append(_raw_relationship_summary(relationship))
        elif previous.model_dump() != relationship.model_dump():
            relationship_by_key[key] = relationship
            corrected_relationships.append(
                {
                    "relationship": _relationship_label(relationship),
                    "before": _raw_relationship_summary(previous),
                    "after": _raw_relationship_summary(relationship),
                }
            )
    base.relationships = list(relationship_by_key.values())

    total_changes = (
        len(added_entities)
        + len(corrected_entities)
        + len(added_relationships)
        + len(corrected_relationships)
    )
    return {
        "pass_index": pass_index,
        "raw_response_chars": raw_response_chars,
        "raw_entities": len(extra.entities),
        "raw_relationships": len(extra.relationships),
        "entities_before": entities_before,
        "entities_after": len(base.entities),
        "relationships_before": relationships_before,
        "relationships_after": len(base.relationships),
        "added_entities_count": len(added_entities),
        "corrected_entities_count": len(corrected_entities),
        "added_relationships_count": len(added_relationships),
        "corrected_relationships_count": len(corrected_relationships),
        "total_changes": total_changes,
        "added_entities": added_entities,
        "corrected_entities": corrected_entities,
        "added_relationships": added_relationships,
        "corrected_relationships": corrected_relationships,
    }


def _summarize_gleaning_reports(
    reports: list[dict[str, Any]],
    *,
    configured_passes: int,
    final_raw: RawExtractionResult,
) -> dict[str, Any]:
    return {
        "gleaning_configured_passes": configured_passes,
        "gleaning_passes_run": len(reports),
        "total_changes": sum(int(report["total_changes"]) for report in reports),
        "entities_added": sum(int(report["added_entities_count"]) for report in reports),
        "entities_corrected": sum(int(report["corrected_entities_count"]) for report in reports),
        "relationships_added": sum(int(report["added_relationships_count"]) for report in reports),
        "relationships_corrected": sum(
            int(report["corrected_relationships_count"]) for report in reports
        ),
        "final_raw_entities": len(final_raw.entities),
        "final_raw_relationships": len(final_raw.relationships),
        "passes": reports,
    }


def _gleaning_report_detail(report: dict[str, Any]) -> str:
    return (
        f"raw_entities={report['raw_entities']}, "
        f"raw_relationships={report['raw_relationships']}, "
        f"added_entities={report['added_entities_count']}, "
        f"corrected_entities={report['corrected_entities_count']}, "
        f"added_relationships={report['added_relationships_count']}, "
        f"corrected_relationships={report['corrected_relationships_count']}"
    )


def _raw_entity_key(entity: RawEntityRecord) -> str:
    return entity.name.casefold().strip()


def _raw_relationship_key(
    relationship: RawRelationshipRecord,
) -> tuple[str, str, str]:
    return (
        relationship.source.casefold().strip(),
        relationship.target.casefold().strip(),
        relationship.rel_type.strip().upper(),
    )


def _raw_entity_summary(entity: RawEntityRecord) -> dict[str, Any]:
    return {
        "name": entity.name,
        "entity_type": entity.entity_type,
        "evidence_status": entity.evidence_status,
        "evidence_status_defaulted": entity.evidence_status_defaulted,
        "description": _clip_text(entity.description, 240),
    }


def _raw_relationship_summary(relationship: RawRelationshipRecord) -> dict[str, Any]:
    return {
        "source": relationship.source,
        "target": relationship.target,
        "rel_type": relationship.rel_type,
        "evidence_status": relationship.evidence_status,
        "evidence_status_defaulted": relationship.evidence_status_defaulted,
        "keywords": relationship.keywords,
        "description": _clip_text(relationship.description, 240),
    }


def _relationship_label(relationship: RawRelationshipRecord) -> str:
    return f"{relationship.source} -> {relationship.target} ({relationship.rel_type})"


# ---------------------------------------------------------------------------
# Helpers (module-level so the @mlflow.trace decorator captures them as spans)
# ---------------------------------------------------------------------------


def _article_prompt_vars(article: ArticleIn) -> dict[str, str]:
    return {
        "article_title": article.title,
        "source_name": article.source_name,
        "published_at": article.published_at.isoformat() if article.published_at else "unknown",
        "article_tags": ", ".join(article.tags[:20]) if article.tags else "none",
        "input_text": article.text[:ARTICLE_TEXT_CHAR_LIMIT],
    }


def _messages_to_system_user(messages: Any) -> tuple[str, str]:
    if isinstance(messages, str):
        return "", messages
    if not isinstance(messages, list):
        raise TypeError(
            f"Expected prompt registry to format to chat messages, got {type(messages)!r}"
        )

    system_parts: list[str] = []
    user_parts: list[str] = []
    other_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        else:
            other_parts.append(content)

    system_prompt = "\n\n".join(part for part in system_parts if part)
    user_prompt = "\n\n".join(part for part in user_parts if part)
    if not user_prompt and other_parts:
        user_prompt = "\n\n".join(part for part in other_parts if part)
    if not user_prompt:
        raise ValueError("Prompt registry formatted no user message content")
    return system_prompt, user_prompt


def _registry_prompt_metadata(prompt: Any, uri: str) -> dict[str, Any]:
    version = getattr(prompt, "version", None)
    return {
        "prompt_name": getattr(prompt, "name", None),
        "prompt_version": str(version) if version is not None else None,
        "prompt_uri": getattr(prompt, "uri", uri),
        "prompt_source": "mlflow_prompt_registry",
    }


def _link_prompt_to_current_trace(prompt: Any) -> None:
    """Create MLflow's prompt-version association for the active trace."""
    trace_id = _current_trace_id()
    name = getattr(prompt, "name", None)
    version = getattr(prompt, "version", None)
    if not trace_id or not name or version is None:
        return
    try:
        MlflowClient().link_prompt_versions_to_trace([prompt], trace_id)
        mlflow.update_current_trace(
            tags={
                f"prompt.{name}.version": str(version),
                f"prompt.{name}.uri": getattr(prompt, "uri", f"prompts:/{name}/{version}"),
            }
        )
    except Exception:  # pragma: no cover - observability must never block extraction
        pass


def _annotate_extract_span(article: ArticleIn, prompt_metadata: dict[str, Any]) -> None:
    """Attach inputs/attributes to the current `extract_entities` span."""
    span = mlflow.get_current_active_span()
    if span is None:
        return
    try:
        span.set_inputs(
            {
                "article_url": article.url,
                "article_title": article.title,
                "source_name": article.source_name,
                "text_chars": len(article.text),
                "tags": list(article.tags),
            }
        )
        attrs: dict[str, Any] = {
            "article_url": article.url,
            "source_name": article.source_name,
            "text_chars": len(article.text),
            **prompt_metadata,
        }
        span.set_attributes(attrs)
    except Exception:  # pragma: no cover - never let observability break extraction
        pass


def _attach_extraction_to_span(
    extraction: ExtractionResult,
    extraction_audit: dict[str, Any] | None = None,
) -> None:
    """Set the parent span's outputs to the full structured ExtractionResult.

    This is what gives the trace detail view a Langfuse-style "Output" panel
    with every extracted entity, funding round, and relationship expanded —
    not just a stringified blob inside the OpenAI response.
    """
    span = mlflow.get_current_active_span()
    if span is None:
        return
    try:
        payload = extraction.model_dump(mode="json")
        if extraction_audit:
            span.set_outputs({"extraction": payload, "gleaning": extraction_audit})
        else:
            span.set_outputs(payload)
        attributes = {
            "entity_count": extraction.entity_count(),
            "startups_count": len(extraction.startups),
            "investors_count": len(extraction.investors),
            "people_count": len(extraction.people),
            "topics_count": len(extraction.topics),
            "companies_count": len(extraction.companies),
            "relationships_count": len(extraction.relationships),
        }
        if extraction_audit:
            attributes.update(
                {
                    "gleaning_configured_passes": extraction_audit["gleaning_configured_passes"],
                    "gleaning_passes_run": extraction_audit["gleaning_passes_run"],
                    "gleaning_total_changes": extraction_audit["total_changes"],
                    "gleaning_entities_added": extraction_audit["entities_added"],
                    "gleaning_entities_corrected": extraction_audit["entities_corrected"],
                    "gleaning_relationships_added": extraction_audit["relationships_added"],
                    "gleaning_relationships_corrected": extraction_audit["relationships_corrected"],
                }
            )
        span.set_attributes(attributes)
    except Exception:  # pragma: no cover
        pass


@mlflow.trace(name="parse_extraction_response", span_type=SpanType.PARSER)
def _parse_extraction_response(
    raw: RawExtractionResult,
    article: ArticleIn,
    *,
    raw_text: str,
) -> ExtractionResult:
    """Convert LightRAG records into the typed `ExtractionResult`.

    Lives as its own span so the trace's PARSER node carries the
    structured output independently of the LLM span.
    """
    result = _raw_to_extraction_result(raw, article)
    span = mlflow.get_current_active_span()
    if span is not None:
        try:
            span.set_inputs(
                {
                    "raw_text_chars": len(raw_text),
                    "raw_entities": len(raw.entities),
                    "raw_relationships": len(raw.relationships),
                }
            )
            span.set_outputs(result.model_dump(mode="json"))
            span.set_attributes(
                {
                    "entity_count": result.entity_count(),
                    "raw_text_chars": len(raw_text),
                }
            )
        except Exception:  # pragma: no cover
            pass
    return result


def _clip_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


# ---------------------------------------------------------------------------
# LightRAG delimiter parsing
# ---------------------------------------------------------------------------


def parse_extraction_output(raw: str) -> RawExtractionResult:
    """Parse delimiter output, failing closed to ``unsure`` evidence status."""
    entities: list[RawEntityRecord] = []
    relationships: list[RawRelationshipRecord] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if COMPLETION_DELIMITER in line:
            break
        parts = [p.strip() for p in line.split(TUPLE_DELIMITER, maxsplit=6)]
        if len(parts) < 4:
            continue
        record_type = parts[0].lower()
        if record_type == "entity":
            name = _normalize_entity_name(parts[1])
            etype = _normalize_entity_type(parts[2])
            if len(parts) >= 5:
                evidence_status, status_defaulted = _normalize_evidence_status(parts[3])
                desc = _normalize_description(parts[4])
            else:
                evidence_status = "unsure"
                status_defaulted = True
                desc = _normalize_description(parts[3])
            if name and etype and desc:
                entities.append(
                    RawEntityRecord(
                        name=name,
                        entity_type=etype,
                        evidence_status=evidence_status,
                        evidence_status_defaulted=status_defaulted,
                        description=desc,
                    )
                )
        elif record_type == "relation" and len(parts) >= 6:
            src = _normalize_entity_name(parts[1])
            tgt = _normalize_entity_name(parts[2])
            rel_type = parts[3].strip().upper()
            if len(parts) >= 7:
                evidence_status, status_defaulted = _normalize_evidence_status(parts[4])
                kw = _normalize_keywords(parts[5])
                desc = _normalize_description(parts[6])
            else:
                evidence_status = "unsure"
                status_defaulted = True
                kw = _normalize_keywords(parts[4])
                desc = _normalize_description(parts[5])
            if src and tgt and rel_type and desc:
                relationships.append(
                    RawRelationshipRecord(
                        source=src,
                        target=tgt,
                        rel_type=rel_type,
                        evidence_status=evidence_status,
                        evidence_status_defaulted=status_defaulted,
                        keywords=kw,
                        description=desc,
                    )
                )

    return RawExtractionResult(entities=entities, relationships=relationships)


_TYPE_MAP: dict[str, EntityType] = {
    "startup": "Startup",
    "investor": "Investor",
    "person": "Person",
    "topic": "Topic",
    "company": "Company",
}


def _normalize_evidence_status(value: str) -> tuple[EvidenceStatus, bool]:
    status = value.strip().casefold()
    if status in {"stated", "attributed", "unsure"}:
        return status, False  # type: ignore[return-value]
    return "unsure", True


def _raw_to_extraction_result(raw: RawExtractionResult, article: ArticleIn) -> ExtractionResult:
    startups: list[ExtractedEntity] = []
    investors: list[ExtractedEntity] = []
    people: list[ExtractedEntity] = []
    topics: list[ExtractedEntity] = []
    companies: list[ExtractedEntity] = []
    source = SourceAttribution(article_url=article.url, article_title=article.title)

    entity_index: dict[str, EntityType] = {}
    for rec in raw.entities:
        label = _TYPE_MAP.get(rec.entity_type)
        if label is None:
            continue
        entity = ExtractedEntity(
            name=rec.name,
            evidence_status=rec.evidence_status,
            evidence_status_defaulted=rec.evidence_status_defaulted,
            description=rec.description,
            source=source,
        )
        if label == "Startup":
            startups.append(entity)
        elif label == "Investor":
            investors.append(entity)
        elif label == "Person":
            people.append(entity)
        elif label == "Company":
            companies.append(entity)
        else:
            topics.append(entity)
        entity_index[rec.name.lower()] = label

    relationships: list[ExtractedRelationship] = []
    for rec in raw.relationships:
        src_type = entity_index.get(rec.source.lower())
        tgt_type = entity_index.get(rec.target.lower())
        if src_type is None or tgt_type is None:
            continue
        try:
            relationships.append(
                ExtractedRelationship(
                    type=rec.rel_type,
                    source_name=rec.source,
                    source_type=src_type,
                    target_name=rec.target,
                    target_type=tgt_type,
                    evidence_status=rec.evidence_status,
                    evidence_status_defaulted=rec.evidence_status_defaulted,
                    keywords=rec.keywords or None,
                    evidence=rec.description,
                )
            )
        except ValidationError:
            continue

    return ExtractionResult(
        startups=startups,
        investors=investors,
        people=people,
        topics=topics,
        companies=companies,
        relationships=relationships,
    )
