"""LLM-driven extraction.

Tracing model:
  - The OpenAI Chat Completions call is auto-instrumented as a
    `CHAT_MODEL` span by `mlflow.openai.autolog()` (set up in
    `app.observability.setup`). That span captures the full prompt,
    response, tokens, and (when supported) cost.
  - `_extract_article_traced` is decorated with `@mlflow.trace` so its
    retry loop, the LLM call, and the response parsing all hang off a
    single `extract_entities` span inside the parent `process_article` trace.
    Its Input panel shows the exact parsed scraper output next to the final
    rendered LLM article input and messages.
  - `_parse_extraction_response` is its own PARSER span. We
    `set_outputs(extraction.model_dump(...))` there so the trace's
    detail view shows the full structured `ExtractionResult` JSON —
    not just the raw model text.
"""

import asyncio
import hashlib
import json
import re
import unicodedata
from time import perf_counter
from typing import Any, cast

import mlflow
from mlflow import MlflowClient
from mlflow.entities import SpanType
from openai import AsyncOpenAI, RateLimitError
from pydantic import ValidationError

from app.core.config import Settings
from app.core.logging import get_logger
from app.evidence import EvidenceCatalog, build_evidence_catalog
from app.models.extraction import (
    ADMITTED_EVIDENCE_STATUSES,
    ArticleIn,
    EntityType,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    RawEntityRecord,
    RawExtractionResult,
    RawRelationshipRecord,
    SourceAttribution,
    StructuredExtractionResponse,
    strongest_evidence_status,
)
from app.observability.context import current_trace_id as _current_trace_id
from app.observability.llm_steps import (
    LLM_STEP_EXTRACTION,
    LLM_STEP_GLEANING,
    llm_workflow_step,
)
from app.prompts.extraction import (
    STRUCTURED_OUTPUT_CONTRACT,
    article_prompt_audit,
    build_article_prompt_input,
    build_extraction_system_prompt,
    build_extraction_prompt_registry_template,
    build_extraction_user_prompt,
    build_gleaning_prompt,
    build_gleaning_prompt_registry_template,
)
from app.relationship_contract import (
    RELATIONSHIP_SIGNATURES,
    has_valid_relationship_signature,
)
from app.services.progress import (
    article_fields,
    extraction_summary,
)
from app.topic_ontology import get_topic_ontology


MISSING_OPENAI_API_KEY_MESSAGE = (
    "OPENAI_API_KEY is required for article extraction. "
    "Set OPENAI_API_KEY in .env and restart the backend."
)
_EXTRACTION_PROMPT_CONTRACT_MARKERS = (
    "CLOSED TOPIC ONTOLOGY",
    STRUCTURED_OUTPUT_CONTRACT,
    "evidence_ref",
    "Both top-level arrays are required",
)
_GLEANING_PROMPT_CONTRACT_MARKERS = (
    STRUCTURED_OUTPUT_CONTRACT,
    "smallest contiguous same-block range",
    "entities",
    "relationships",
)
_EXTRACTION_MAX_COMPLETION_TOKENS = 12_000
_RATE_LIMIT_FALLBACK_SECONDS = 1.0
_RATE_LIMIT_SAFETY_BUFFER_SECONDS = 0.25
_RATE_LIMIT_MAX_DELAY_SECONDS = 60.0
_RATE_LIMIT_MESSAGE_RE = re.compile(
    r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?)",
    re.IGNORECASE,
)


class NonRetryableLLMResponseError(RuntimeError):
    """A completed API response that must not trigger another paid attempt."""


class LLMExtractionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("extractor")
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)
        self._rate_limit_lock = asyncio.Lock()
        self._rate_limit_resume_at = 0.0
        self._rate_limit_stats = _empty_rate_limit_stats()
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
            prompt_kind = (
                "extraction" if uri == self.settings.mlflow_prompt_extraction_uri else "gleaning"
            )
            if not _registered_prompt_contract_compatible(prompt, prompt_kind):
                self.logger.warning(
                    "mlflow_prompt_contract_incompatible",
                    extra={
                        "event": "startup",
                        "workflow_step": "prompt_cache",
                        "detail": (
                            f"uri={uri}; version={getattr(prompt, 'version', '?')}; "
                            "using local prompt because the registered prompt does not "
                            "contain the current ontology/evidence contract"
                        ),
                    },
                )
                return None
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
            prompt_variables = set(getattr(cached_prompt, "variables", set()) or set())
            missing_variables = prompt_variables - set(variables)
            if missing_variables:
                raise ValueError(
                    "Prompt registry variables were not supplied: "
                    + ", ".join(sorted(missing_variables))
                )
            format_variables = {name: variables[name] for name in prompt_variables}
            messages = cached_prompt.format(**format_variables)
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

    def pop_openai_rate_limit_stats(self) -> dict[str, float]:
        stats = dict(self._rate_limit_stats)
        self._rate_limit_stats = _empty_rate_limit_stats()
        return stats

    async def _create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        structured_output: bool = False,
    ) -> Any:
        if self._client is None:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)

        attempts = max(1, int(getattr(self.settings, "llm_retry_attempts", 1) or 1))
        last_error: RateLimitError | None = None
        for attempt in range(1, attempts + 1):
            async with self._rate_limit_lock:
                wait_seconds = self._rate_limit_resume_at - asyncio.get_running_loop().time()
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            try:
                options = self._chat_completion_options()
                if structured_output:
                    options.update(
                        {
                            "response_format": _structured_extraction_response_format(),
                            "max_completion_tokens": _EXTRACTION_MAX_COMPLETION_TOKENS,
                        }
                    )
                return await self._client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=messages,
                    timeout=self.settings.llm_timeout_seconds,
                    **options,
                )
            except RateLimitError as exc:
                last_error = exc
                exhausted = attempt == attempts
                wait_seconds = await self._set_rate_limit_pause(
                    _rate_limit_delay_seconds(exc, attempt)
                )
                self._record_rate_limit_pause(wait_seconds, exhausted=exhausted)
                self.logger.warning(
                    "openai_rate_limit_pause",
                    extra={
                        "event": "extraction",
                        "workflow_step": "llm_retry",
                        "mode": "openai",
                        "model": self.settings.openai_model,
                        "attempt_index": attempt,
                        "attempt_total": attempts,
                        "retry_delay_seconds": round(wait_seconds, 3),
                        "error": str(exc),
                        "detail": "OpenAI rate limit reached; pausing chat requests",
                    },
                )
                if exhausted:
                    raise
                await asyncio.sleep(wait_seconds)

        raise last_error or RuntimeError("OpenAI chat completion retry loop ended unexpectedly")

    async def _set_rate_limit_pause(self, delay_seconds: float) -> float:
        async with self._rate_limit_lock:
            now = asyncio.get_running_loop().time()
            self._rate_limit_resume_at = max(self._rate_limit_resume_at, now + delay_seconds)
            return max(0.0, self._rate_limit_resume_at - now)

    def _record_rate_limit_pause(self, wait_seconds: float, *, exhausted: bool) -> None:
        self._rate_limit_stats["events"] += 1
        self._rate_limit_stats["wait_seconds"] = round(
            self._rate_limit_stats["wait_seconds"] + wait_seconds,
            3,
        )
        if exhausted:
            self._rate_limit_stats["failures"] += 1
        else:
            self._rate_limit_stats["retries"] += 1

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
        _annotate_extract_span(
            article,
            prompt_metadata,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        started = perf_counter()
        trace_id = _current_trace_id()
        metadata: dict[str, Any] = {
            "trace_id": trace_id,
            "model": self.settings.openai_model,
            "max_attempts": self.settings.llm_retry_attempts,
            "prompt_input": build_article_prompt_input(article),
            "user_prompt": user_prompt,
            **article_prompt_audit(article),
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
                (
                    result,
                    raw_content,
                    usage,
                    extraction_audit,
                    evidence_validation,
                ) = await self._extract_with_openai(
                    article,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                last_error = exc
                if isinstance(exc, NonRetryableLLMResponseError):
                    message = f"LLM extraction stopped without retry: {exc}"
                    self.logger.error(
                        "llm_extraction_failed",
                        extra={
                            "event": "extraction",
                            "workflow_step": "llm_failure",
                            "url": article.url,
                            "error": message,
                            "attempt_index": attempt,
                            "attempt_total": self.settings.llm_retry_attempts,
                            "detail": "response was truncated or refused; retry disabled",
                            **article_log_fields,
                        },
                    )
                    raise RuntimeError(message) from exc
                if isinstance(exc, RateLimitError):
                    message = f"LLM extraction failed after rate-limit retries: {exc}"
                    self.logger.error(
                        "llm_extraction_failed",
                        extra={
                            "event": "extraction",
                            "workflow_step": "llm_failure",
                            "url": article.url,
                            "error": message,
                            "attempt_index": attempt,
                            "attempt_total": self.settings.llm_retry_attempts,
                            "detail": "OpenAI rate-limit retry budget exhausted",
                            **article_log_fields,
                        },
                    )
                    raise RuntimeError(message) from exc
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
                    "gleaning_relationships_added": extraction_audit["relationships_added"],
                    "evidence_validation": evidence_validation,
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
            _attach_extraction_to_span(result, extraction_audit, evidence_validation)
            return result, metadata

        # Defensive — loop should either return or raise above.
        raise RuntimeError(f"LLM extraction terminated unexpectedly: last error={last_error!r}")

    async def _call_llm_raw(
        self,
        system_prompt: str | None,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        *,
        structured_output: bool = False,
    ) -> str:
        """Call OpenAI and return the raw text content (no JSON parsing)."""
        if self._client is None:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)
        content, _ = await self._call_llm_raw_with_usage(
            system_prompt,
            user_prompt,
            history=history,
            structured_output=structured_output,
        )
        return content

    async def _call_llm_raw_with_usage(
        self,
        system_prompt: str | None,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        *,
        structured_output: bool = False,
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
        response = await self._create_chat_completion(
            messages=messages,
            structured_output=structured_output,
        )
        usage = _openai_usage(response)
        return _chat_completion_content(response), usage

    def _chat_completion_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.settings.openai_temperature}
        if self.settings.openai_seed is not None:
            options["seed"] = self.settings.openai_seed
        return options

    def article_extraction_policy(self) -> dict[str, Any]:
        ontology = get_topic_ontology()
        return {
            "cache_version": "article_extraction_cache_v7_structured_evidence_refs",
            "model": self.settings.openai_model,
            "gleaning_passes": self.settings.llm_gleaning_passes,
            "admission_policy": "stated|attributed+resolved_evidence_ref",
            "topic_ontology": {
                "version": ontology.ontology_version,
                "sha256": ontology.content_hash,
                "mode": "closed_exact_aliases",
            },
            "extraction": _prompt_policy_payload(
                self._extraction_prompt,
                self.settings.mlflow_prompt_extraction_uri,
                build_extraction_prompt_registry_template(),
            ),
            "gleaning": _prompt_policy_payload(
                self._gleaning_prompt,
                self.settings.mlflow_prompt_gleaning_uri,
                build_gleaning_prompt_registry_template(),
            ),
        }

    def article_extraction_policy_hash(self) -> str:
        return _hash_policy(self.article_extraction_policy())

    async def extract_with_gleaning(
        self,
        article: ArticleIn,
        *,
        max_gleaning: int | None = None,
    ) -> RawExtractionResult:
        """Structured extraction with optional gleaning passes."""
        if self._client is None:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)
        passes = max_gleaning if max_gleaning is not None else self.settings.llm_gleaning_passes
        system_prompt, user_prompt, _ = self._render_extraction_prompts(article)

        with llm_workflow_step(LLM_STEP_EXTRACTION):
            raw_content = await self._call_llm_raw(
                system_prompt,
                user_prompt,
                structured_output=True,
            )
        result = parse_extraction_output(raw_content)
        history: list[dict[str, str]] = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_content},
        ]

        for pass_index in range(1, passes + 1):
            gleaning_prompt, _ = self._render_gleaning_prompt()
            gleaning_prompt = _with_gleaning_block_coverage(
                gleaning_prompt,
                article,
                result,
            )
            with llm_workflow_step(LLM_STEP_GLEANING):
                continuation = await self._call_llm_raw(
                    system_prompt,
                    gleaning_prompt,
                    history=history,
                    structured_output=True,
                )
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

            if not extra.entities and not extra.relationships:
                break

        return result

    def _render_extraction_prompts(self, article: ArticleIn) -> tuple[str, str, dict[str, Any]]:
        """Render the system and user prompts sent to OpenAI."""
        ontology = get_topic_ontology()
        ontology_metadata = {
            "topic_ontology_version": ontology.ontology_version,
            "topic_ontology_hash": ontology.content_hash,
        }
        vars_ = _article_prompt_vars(article)
        self.logger.debug(
            "extraction_prompt_vars",
            extra={
                "event": "extraction",
                "workflow_step": "prompt_render",
                "detail": f"prompt_input_chars={len(vars_['input_text'])}",
                **article_prompt_audit(article),
            },
        )
        cached = self._render_prompt_from_cache(
            self._extraction_prompt,
            vars_,
            self.settings.mlflow_prompt_extraction_uri,
        )
        if cached is not None:
            system_prompt, user_prompt, metadata = cached
            return system_prompt, user_prompt, {**metadata, **ontology_metadata}
        return (
            build_extraction_system_prompt(),
            build_extraction_user_prompt(article),
            {
                "prompt_uri": None,
                "prompt_source": "local_fallback",
                **ontology_metadata,
            },
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
    ) -> tuple[
        ExtractionResult,
        str,
        dict[str, int] | None,
        dict[str, Any],
        dict[str, Any],
    ]:
        """Run the extraction LLM call.

        Returns the accepted result, raw output, usage, gleaning audit, and
        per-fact evidence validation. Token usage on the OpenAI span is captured
        automatically by ``mlflow.openai.autolog()``; we also return it here so
        the orchestrator can roll it up into per-job metrics.
        """
        if self._client is None:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)
        with llm_workflow_step(LLM_STEP_EXTRACTION):
            response = await self._create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                structured_output=True,
            )
            usage = _openai_usage(response)
        content = _chat_completion_content(response)
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
        parsed = _parse_extraction_response(
            raw,
            article,
            raw_text=combined_raw_text,
        )
        result: ExtractionResult = parsed["extraction"]
        evidence_validation: dict[str, Any] = parsed["evidence_validation"]
        self.logger.debug(
            "llm_output_received",
            extra={
                "event": "extraction",
                "workflow_step": "llm_output",
                "detail": extraction_summary(result),
                "gleaning_passes": audit["gleaning_passes_run"],
                "gleaning_additions": audit["total_changes"],
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
        return result, combined_raw_text, usage, audit, evidence_validation

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
            gleaning_prompt = _with_gleaning_block_coverage(
                gleaning_prompt,
                article,
                raw,
            )
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
                        f"relationships={len(raw.relationships)} "
                        f"by_type={_format_relationship_type_counts(raw.relationships)}"
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

                with llm_workflow_step(LLM_STEP_GLEANING):
                    continuation, pass_usage = await self._call_llm_raw_with_usage(
                        system_prompt,
                        gleaning_prompt,
                        history=history,
                        structured_output=True,
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
                            "added_relationships": report["added_relationships_count"],
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
            if not extra.entities and not extra.relationships:
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
                        "gleaning_relationships_added": audit["relationships_added"],
                    }
                )
            except Exception:  # pragma: no cover
                pass
        return raw, raw_texts, usage_total, audit


def _structured_extraction_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "startup_graph_extraction",
            "strict": True,
            "schema": StructuredExtractionResponse.model_json_schema(),
        },
    }


def _chat_completion_content(response: Any) -> str:
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise NonRetryableLLMResponseError(
            f"response reached {_EXTRACTION_MAX_COMPLETION_TOKENS} completion tokens"
        )
    message = choice.message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise NonRetryableLLMResponseError(f"model refusal: {refusal}")
    return message.content or ""


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
        **_openai_cache_usage(usage),
    }


def _openai_cache_usage(usage: Any) -> dict[str, int]:
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached = _usage_detail(prompt_details, "cached_tokens")
    return {"cache_read_input_tokens": cached} if cached else {}


def _usage_detail(value: Any, key: str) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return int(value.get(key) or 0)
    return int(getattr(value, key, 0) or 0)


def _empty_rate_limit_stats() -> dict[str, float]:
    return {
        "events": 0.0,
        "retries": 0.0,
        "failures": 0.0,
        "wait_seconds": 0.0,
    }


def _rate_limit_delay_seconds(exc: RateLimitError, attempt_index: int) -> float:
    delay: float | None = None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        retry_after_ms = _float_or_none(headers.get("retry-after-ms"))
        retry_after = _float_or_none(headers.get("retry-after"))
        if retry_after_ms is not None:
            delay = retry_after_ms / 1000.0
        elif retry_after is not None:
            delay = retry_after

    match = _RATE_LIMIT_MESSAGE_RE.search(str(exc))
    value = _float_or_none(match.group(1)) if match else None
    if delay is None and value is not None:
        delay = value / 1000.0 if match.group(2).lower().startswith("m") else value

    if delay is None:
        delay = min(_RATE_LIMIT_FALLBACK_SECONDS * (2 ** max(0, attempt_index - 1)), 12.0)
    return min(delay + _RATE_LIMIT_SAFETY_BUFFER_SECONDS, _RATE_LIMIT_MAX_DELAY_SECONDS)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _combine_usage(
    left: dict[str, int] | None,
    right: dict[str, int] | None,
) -> dict[str, int] | None:
    if not left and not right:
        return None
    merged = {key: 0 for key in _usage_keys(left, right)}
    for usage in (left, right):
        if not usage:
            continue
        for key in merged:
            merged[key] += int(usage.get(key, 0) or 0)
    return merged


def _usage_keys(*usages: dict[str, int] | None) -> list[str]:
    keys = {"input_tokens", "output_tokens", "total_tokens"}
    for usage in usages:
        if usage:
            keys.update(usage)
    return sorted(keys)


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
        "relationships_added": 0,
        "final_raw_entities": len(raw.entities),
        "final_raw_relationships": len(raw.relationships),
        "initial_raw_relationships_by_type": _relationship_type_counts(raw.relationships),
        "passes": [],
    }


def _with_gleaning_block_coverage(
    prompt: str,
    article: ArticleIn,
    raw: RawExtractionResult,
) -> str:
    catalog = build_evidence_catalog(article.text)
    all_blocks = {sentence.block_number for sentence in catalog.sentences.values()}
    if not all_blocks:
        return prompt
    referenced_blocks: set[int] = set()
    for record in [*raw.entities, *raw.relationships]:
        match = re.match(r"^B(\d{3,})\.S\d{3,}", record.evidence_ref.strip())
        if match:
            referenced_blocks.add(int(match.group(1)))
    unreferenced_blocks = sorted(all_blocks - referenced_blocks)
    if not unreferenced_blocks:
        coverage = "Every EvidenceBlock already has at least one cited fact."
    else:
        coverage = (
            "EvidenceBlocks with no cited fact yet: "
            + ", ".join(f"B{block:03d}" for block in unreferenced_blocks)
            + "."
        )
    return (
        f"{prompt}\n\nBLOCK COVERAGE CHECK\n"
        f"The article runs from B001 through B{max(all_blocks):03d}. "
        f"{coverage} Inspect every listed block, especially the final blocks, before returning. "
        "An unreferenced block may be ineligible; add nothing from it unless it contains a "
        "supported fact."
    )


def _merge_gleaning_pass(
    base: RawExtractionResult,
    extra: RawExtractionResult,
    *,
    pass_index: int,
    raw_response_chars: int,
) -> dict[str, Any]:
    """Append genuinely new audit facts without changing the initial extraction."""
    entities_before = len(base.entities)
    relationships_before = len(base.relationships)
    relationships_before_by_type = _relationship_type_counts(base.relationships)

    entity_keys = {_raw_entity_key(entity) for entity in base.entities}
    relationship_keys = {_raw_relationship_key(relationship) for relationship in base.relationships}
    added_entities: list[dict[str, Any]] = []
    for entity in extra.entities:
        key = _raw_entity_key(entity)
        if entity.evidence_status_defaulted:
            continue
        if key in entity_keys:
            continue
        entity_keys.add(key)
        base.entities.append(entity)
        summary = _raw_entity_summary(entity)
        added_entities.append(summary)

    added_relationships: list[dict[str, Any]] = []
    for relationship in extra.relationships:
        key = _raw_relationship_key(relationship)
        if relationship.evidence_status_defaulted:
            continue
        if key[0] not in entity_keys or key[1] not in entity_keys:
            continue
        if key in relationship_keys:
            continue
        relationship_keys.add(key)
        base.relationships.append(relationship)
        summary = _raw_relationship_summary(relationship)
        added_relationships.append(summary)
    relationships_after_by_type = _relationship_type_counts(base.relationships)

    total_changes = len(added_entities) + len(added_relationships)
    return {
        "pass_index": pass_index,
        "raw_response_chars": raw_response_chars,
        "raw_entities": len(extra.entities),
        "raw_relationships": len(extra.relationships),
        "raw_relationships_by_type": _relationship_type_counts(extra.relationships),
        "entities_before": entities_before,
        "entities_after": len(base.entities),
        "relationships_before": relationships_before,
        "relationships_after": len(base.relationships),
        "relationships_before_by_type": relationships_before_by_type,
        "relationships_after_by_type": relationships_after_by_type,
        "added_entities_count": len(added_entities),
        "added_relationships_count": len(added_relationships),
        "total_changes": total_changes,
        "added_entities": added_entities,
        "added_relationships": added_relationships,
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
        "relationships_added": sum(int(report["added_relationships_count"]) for report in reports),
        "final_raw_entities": len(final_raw.entities),
        "final_raw_relationships": len(final_raw.relationships),
        "passes": reports,
    }


def _gleaning_report_detail(report: dict[str, Any]) -> str:
    return (
        f"raw_entities={report['raw_entities']}, "
        f"raw_relationships={report['raw_relationships']}, "
        f"added_entities={report['added_entities_count']}, "
        f"added_relationships={report['added_relationships_count']}, "
        f"relationships_before_by_type="
        f"{_format_relationship_type_counts(report['relationships_before_by_type'])}, "
        f"relationships_after_by_type="
        f"{_format_relationship_type_counts(report['relationships_after_by_type'])}"
    )


def _relationship_type_counts(
    relationships: list[RawRelationshipRecord] | dict[str, int],
) -> dict[str, int]:
    if isinstance(relationships, dict):
        return {key: int(value) for key, value in sorted(relationships.items())}
    counts: dict[str, int] = {}
    for relationship in relationships:
        rel_type = relationship.rel_type.strip().upper() or "UNKNOWN"
        counts[rel_type] = counts.get(rel_type, 0) + 1
    return dict(sorted(counts.items()))


def _format_relationship_type_counts(
    relationships: list[RawRelationshipRecord] | dict[str, int],
) -> str:
    counts = _relationship_type_counts(relationships)
    if not counts:
        return "none"
    return ", ".join(f"{rel_type}={count}" for rel_type, count in counts.items())


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
        "type_basis": entity.type_basis,
        "evidence_status": entity.evidence_status,
        "evidence_status_defaulted": entity.evidence_status_defaulted,
        "description": _clip_text(entity.description, 240),
        "evidence_ref": entity.evidence_ref,
        "evidence": _clip_text(entity.evidence, 240),
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
        "evidence_ref": relationship.evidence_ref,
        "evidence": _clip_text(relationship.evidence, 240),
    }


# ---------------------------------------------------------------------------
# Helpers (module-level so the @mlflow.trace decorator captures them as spans)
# ---------------------------------------------------------------------------


def _article_prompt_vars(article: ArticleIn) -> dict[str, str]:
    return {"input_text": build_article_prompt_input(article)}


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
    except Exception as exc:  # pragma: no cover - observability must never block extraction
        get_logger("extractor").warning(
            "mlflow_prompt_trace_link_failed",
            extra={
                "event": "observability",
                "workflow_step": "prompt_registry",
                "error": str(exc),
                "detail": f"trace_id={trace_id}; prompt={name}; version={version}",
            },
        )


def _annotate_extract_span(
    article: ArticleIn,
    prompt_metadata: dict[str, Any],
    *,
    system_prompt: str,
    user_prompt: str,
) -> None:
    """Attach the exact scraper output and rendered LLM input to extraction."""
    span = mlflow.get_current_active_span()
    if span is None:
        return
    try:
        article_input = build_article_prompt_input(article)
        span.set_inputs(
            {
                "scraper_output": article.model_dump(mode="json"),
                "llm_input": {
                    "article_input": article_input,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            }
        )
        attrs: dict[str, Any] = {
            "article_url": article.url,
            "source_name": article.source_name,
            "primary_type": article.primary_type or "unknown",
            "text_chars": len(article.text),
            **article_prompt_audit(article),
            **prompt_metadata,
        }
        if article.cleaning:
            attrs.update(
                {
                    "content_container": article.cleaning.selected_container,
                    "content_chars_before": article.cleaning.text_chars_before,
                    "content_chars_after": article.cleaning.text_chars_after,
                    "content_removed_blocks": len(article.cleaning.removed_blocks),
                    "content_remaining_promotion_markers": len(
                        article.cleaning.remaining_promotion_markers
                    ),
                }
            )
        span.set_attributes(attrs)
    except Exception:  # pragma: no cover - never let observability break extraction
        pass


def _attach_extraction_to_span(
    extraction: ExtractionResult,
    extraction_audit: dict[str, Any] | None = None,
    evidence_validation: dict[str, Any] | None = None,
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
        if extraction_audit or evidence_validation:
            outputs: dict[str, Any] = {"extraction": payload}
            if extraction_audit:
                outputs["gleaning"] = extraction_audit
            if evidence_validation:
                outputs["evidence_validation"] = evidence_validation
            span.set_outputs(outputs)
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
                    "gleaning_relationships_added": extraction_audit["relationships_added"],
                }
            )
        if evidence_validation:
            summary = evidence_validation["summary"]
            attributes.update(
                {
                    "evidence_facts_checked": summary["facts_checked"],
                    "evidence_facts_accepted": summary["facts_accepted"],
                    "evidence_facts_rejected": summary["facts_rejected"],
                    "evidence_matched": summary["evidence_matched"],
                    "evidence_missing": summary["evidence_missing"],
                    "evidence_not_in_article": summary["evidence_not_in_article"],
                    "evidence_match_rate": summary["evidence_match_rate"],
                    "entities_recovered_from_relationship": summary[
                        "entities_recovered_from_relationship"
                    ],
                    "entities_restored_from_relationship": summary[
                        "entities_restored_from_relationship"
                    ],
                    "entities_synthesized_from_relationship": summary[
                        "entities_synthesized_from_relationship"
                    ],
                    "evidence_normalization": evidence_validation["normalization"],
                    "evidence_rejections_by_reason": json.dumps(
                        summary["rejections_by_reason"],
                        sort_keys=True,
                    ),
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
) -> dict[str, Any]:
    """Convert structured records into the typed `ExtractionResult`.

    Lives as its own span so the trace's PARSER node carries the
    structured output independently of the LLM span.
    """
    result, diagnostics = _raw_to_extraction_result_with_diagnostics(raw, article)
    evidence_validation = _evidence_validation_trace_payload(diagnostics)
    if (
        diagnostics["entities_missing_evidence"]
        or diagnostics["entities_unmatched_evidence"]
        or diagnostics["relationships_dropped"]
    ):
        get_logger("extractor").info(
            "llm_grounding_diagnostics",
            extra={
                "event": "extraction",
                "workflow_step": "llm_output",
                "count": diagnostics["raw_entities"] + diagnostics["raw_relationships"],
                "skipped_count": (
                    diagnostics["entities_missing_evidence"]
                    + diagnostics["entities_unmatched_evidence"]
                    + diagnostics["relationships_dropped"]
                ),
                "detail": _raw_conversion_diagnostics_detail(diagnostics),
            },
        )
    span = mlflow.get_current_active_span()
    if span is not None:
        try:
            span.set_inputs(
                {
                    "raw_text_chars": len(raw_text),
                    "raw_entities": len(raw.entities),
                    "raw_relationships": len(raw.relationships),
                    "raw_relationships_by_type": diagnostics["raw_relationships_by_type"],
                }
            )
            evidence_summary = evidence_validation["summary"]
            span.set_attributes(
                {
                    "entity_count": result.entity_count(),
                    "raw_text_chars": len(raw_text),
                    "evidence_facts_checked": evidence_summary["facts_checked"],
                    "evidence_facts_accepted": evidence_summary["facts_accepted"],
                    "evidence_facts_rejected": evidence_summary["facts_rejected"],
                    "evidence_matched": evidence_summary["evidence_matched"],
                    "evidence_missing": evidence_summary["evidence_missing"],
                    "evidence_not_in_article": evidence_summary["evidence_not_in_article"],
                    "evidence_match_rate": evidence_summary["evidence_match_rate"],
                    "entities_recovered_from_relationship": evidence_summary[
                        "entities_recovered_from_relationship"
                    ],
                    "entities_restored_from_relationship": evidence_summary[
                        "entities_restored_from_relationship"
                    ],
                    "entities_synthesized_from_relationship": evidence_summary[
                        "entities_synthesized_from_relationship"
                    ],
                    "evidence_normalization": evidence_validation["normalization"],
                    "evidence_rejections_by_reason": json.dumps(
                        evidence_summary["rejections_by_reason"],
                        sort_keys=True,
                    ),
                    "raw_entity_invalid_type_count": diagnostics["entity_invalid_type_count"],
                    "raw_entities_missing_evidence": diagnostics["entities_missing_evidence"],
                    "raw_entities_unmatched_evidence": diagnostics["entities_unmatched_evidence"],
                    "raw_entity_type_conflicts": diagnostics["entity_type_conflicts"],
                    "raw_relationships": diagnostics["raw_relationships"],
                    "typed_relationships": diagnostics["relationships_kept"],
                    "raw_relationships_dropped": diagnostics["relationships_dropped"],
                    "raw_relationships_missing_evidence": diagnostics[
                        "relationships_missing_evidence"
                    ],
                    "raw_relationships_unmatched_evidence": diagnostics[
                        "relationships_unmatched_evidence"
                    ],
                    "raw_relationships_missing_source_endpoint": diagnostics[
                        "relationships_missing_source_endpoint"
                    ],
                    "raw_relationships_missing_target_endpoint": diagnostics[
                        "relationships_missing_target_endpoint"
                    ],
                    "raw_relationship_validation_errors": diagnostics[
                        "relationship_validation_errors"
                    ],
                    "raw_relationship_invalid_signatures": diagnostics[
                        "relationship_invalid_signatures"
                    ],
                    "raw_relationship_ambiguous_endpoint_types": diagnostics[
                        "relationship_ambiguous_endpoint_types"
                    ],
                    "raw_relationships_by_type": json.dumps(
                        diagnostics["raw_relationships_by_type"],
                        sort_keys=True,
                    ),
                    "typed_relationships_by_type": json.dumps(
                        diagnostics["relationships_kept_by_type"],
                        sort_keys=True,
                    ),
                }
            )
        except Exception:  # pragma: no cover
            pass
    return {
        "extraction": result,
        "evidence_validation": evidence_validation,
    }


def _clip_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _normalize_grounding_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def _verbatim_article_evidence(
    article_text: str,
    evidence: str,
) -> tuple[str | None, str | None]:
    candidate = evidence.strip()
    if not candidate:
        return None, "missing"
    if _normalize_grounding_text(candidate) not in _normalize_grounding_text(article_text):
        return None, "not_in_article"
    return candidate, None


def _resolve_article_evidence(
    catalog: EvidenceCatalog,
    *,
    evidence_ref: str,
    legacy_evidence: str,
) -> tuple[str | None, str | None]:
    if evidence_ref.strip():
        return catalog.resolve(evidence_ref)
    return _verbatim_article_evidence(catalog.article_text, legacy_evidence)


def parse_extraction_output(raw: str) -> RawExtractionResult:
    """Validate the strict JSON response and convert it to internal records."""
    return StructuredExtractionResponse.model_validate_json(raw).to_raw()


_TYPE_MAP: dict[str, EntityType] = {
    "startup": "Startup",
    "investor": "Investor",
    "person": "Person",
    "topic": "Topic",
    "company": "Company",
}


def _prompt_policy_payload(
    prompt: Any,
    uri: str,
    local_template: list[dict[str, str]],
) -> dict[str, Any]:
    if prompt is None:
        return {
            "source": "local_fallback",
            "template": local_template,
        }
    version = getattr(prompt, "version", None)
    return {
        "source": "mlflow_prompt_registry",
        "name": getattr(prompt, "name", None),
        "version": str(version) if version is not None else None,
        "uri": getattr(prompt, "uri", uri),
        "template": getattr(prompt, "template", None),
    }


def _registered_prompt_contract_compatible(prompt: Any, prompt_kind: str) -> bool:
    template = json.dumps(
        getattr(prompt, "template", ""),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    markers = (
        _EXTRACTION_PROMPT_CONTRACT_MARKERS
        if prompt_kind == "extraction"
        else _GLEANING_PROMPT_CONTRACT_MARKERS
    )
    return all(marker in template for marker in markers)


def _hash_policy(policy: dict[str, Any]) -> str:
    serialized = json.dumps(policy, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _raw_to_extraction_result(raw: RawExtractionResult, article: ArticleIn) -> ExtractionResult:
    result, _diagnostics = _raw_to_extraction_result_with_diagnostics(raw, article)
    return result


def _raw_to_extraction_result_with_diagnostics(
    raw: RawExtractionResult,
    article: ArticleIn,
) -> tuple[ExtractionResult, dict[str, Any]]:
    startups: list[ExtractedEntity] = []
    investors: list[ExtractedEntity] = []
    people: list[ExtractedEntity] = []
    topics: list[ExtractedEntity] = []
    companies: list[ExtractedEntity] = []
    diagnostics: dict[str, Any] = {
        "raw_entities": len(raw.entities),
        "typed_entities": 0,
        "entity_invalid_type_count": 0,
        "entities_missing_evidence": 0,
        "entities_unmatched_evidence": 0,
        "entity_type_conflicts": 0,
        "entity_type_basis_counts": {},
        "entity_recoveries": [],
        "raw_relationships": len(raw.relationships),
        "relationships_kept": 0,
        "relationships_dropped": 0,
        "relationships_missing_evidence": 0,
        "relationships_unmatched_evidence": 0,
        "relationships_missing_source_endpoint": 0,
        "relationships_missing_target_endpoint": 0,
        "relationship_validation_errors": 0,
        "relationship_invalid_signatures": 0,
        "relationship_ambiguous_endpoint_types": 0,
        "raw_relationships_by_type": _relationship_type_counts(raw.relationships),
        "relationships_kept_by_type": {},
        "fact_evidence_checks": [],
    }
    fact_evidence_checks: list[dict[str, Any]] = diagnostics["fact_evidence_checks"]
    evidence_catalog = build_evidence_catalog(article.text)

    declared_entity_types: dict[str, list[EntityType]] = {}
    for rec in raw.entities:
        label = _TYPE_MAP.get(rec.entity_type)
        if label is None:
            continue
        type_candidates = declared_entity_types.setdefault(rec.name.casefold(), [])
        if label not in type_candidates:
            type_candidates.append(label)

    accepted_entity_types: dict[str, list[EntityType]] = {}
    for rec in raw.entities:
        evidence, evidence_error = _resolve_article_evidence(
            evidence_catalog,
            evidence_ref=rec.evidence_ref,
            legacy_evidence=rec.evidence,
        )
        decision: dict[str, Any] = {
            "kind": "entity",
            "fact": {
                "name": rec.name,
                "entity_type": rec.entity_type,
                "type_basis": rec.type_basis,
                "description": rec.description,
            },
            "evidence_status": rec.evidence_status,
            "evidence_ref": rec.evidence_ref,
            "evidence": evidence or rec.evidence,
            "evidence_check": evidence_error or "matched",
            "accepted": False,
            "rejection_reasons": [],
        }
        if evidence_error in {"missing", "missing_ref"}:
            diagnostics["entities_missing_evidence"] += 1
            decision["rejection_reasons"].append("missing_evidence")
            fact_evidence_checks.append(decision)
            continue
        if evidence_error is not None:
            diagnostics["entities_unmatched_evidence"] += 1
            decision["rejection_reasons"].append(
                "evidence_not_in_article"
                if evidence_error == "not_in_article"
                else f"invalid_evidence_ref:{evidence_error}"
            )
            fact_evidence_checks.append(decision)
            continue
        label = _TYPE_MAP.get(rec.entity_type)
        if label is None:
            diagnostics["entity_invalid_type_count"] += 1
            decision["rejection_reasons"].append("invalid_entity_type")
            fact_evidence_checks.append(decision)
            continue
        entity = ExtractedEntity(
            name=rec.name,
            type_basis=rec.type_basis,
            evidence_status=rec.evidence_status,
            evidence_status_defaulted=rec.evidence_status_defaulted,
            description=rec.description,
            source=SourceAttribution(
                article_url=article.url,
                article_title=article.title,
                evidence=evidence,
            ),
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
        type_candidates = accepted_entity_types.setdefault(rec.name.casefold(), [])
        if label not in type_candidates:
            type_candidates.append(label)
        decision["accepted"] = True
        fact_evidence_checks.append(decision)
    diagnostics["entity_type_conflicts"] = sum(
        len(type_candidates) > 1 for type_candidates in accepted_entity_types.values()
    )
    diagnostics["entity_type_basis_counts"] = dict(
        sorted(
            {
                basis: sum(1 for entity in raw.entities if entity.type_basis == basis)
                for basis in {entity.type_basis for entity in raw.entities}
            }.items()
        )
    )

    relationships: list[ExtractedRelationship] = []
    for rec in raw.relationships:
        evidence, evidence_error = _resolve_article_evidence(
            evidence_catalog,
            evidence_ref=rec.evidence_ref,
            legacy_evidence=rec.evidence,
        )
        decision = {
            "kind": "relationship",
            "fact": {
                "source": rec.source,
                "relationship_type": rec.rel_type,
                "target": rec.target,
                "keywords": rec.keywords,
            },
            "evidence_status": rec.evidence_status,
            "evidence_ref": rec.evidence_ref,
            "evidence": evidence or rec.evidence,
            "evidence_check": evidence_error or "matched",
            "accepted": False,
            "rejection_reasons": [],
        }
        if evidence_error in {"missing", "missing_ref"}:
            diagnostics["relationships_dropped"] += 1
            diagnostics["relationships_missing_evidence"] += 1
            decision["rejection_reasons"].append("missing_evidence")
            fact_evidence_checks.append(decision)
            continue
        if evidence_error is not None:
            diagnostics["relationships_dropped"] += 1
            diagnostics["relationships_unmatched_evidence"] += 1
            decision["rejection_reasons"].append(
                "evidence_not_in_article"
                if evidence_error == "not_in_article"
                else f"invalid_evidence_ref:{evidence_error}"
            )
            fact_evidence_checks.append(decision)
            continue
        # Relationship evidence is validated independently. Prefer an accepted
        # entity row, but retain a valid declared type when only that row's
        # standalone evidence was rejected. Accepted relationships restore their
        # missing endpoints below.
        source_types = accepted_entity_types.get(
            rec.source.casefold()
        ) or declared_entity_types.get(rec.source.casefold(), [])
        target_types = accepted_entity_types.get(
            rec.target.casefold()
        ) or declared_entity_types.get(rec.target.casefold(), [])
        # If an endpoint row is absent, infer its type only when the relationship
        # contract leaves exactly one possibility (for example FOUNDED_BY targets).
        signature = RELATIONSHIP_SIGNATURES.get(rec.rel_type)
        if signature is not None:
            allowed_source_types, allowed_target_types = signature
            if not source_types and len(allowed_source_types) == 1:
                source_types = [cast(EntityType, next(iter(allowed_source_types)))]
            if not target_types and len(allowed_target_types) == 1:
                target_types = [cast(EntityType, next(iter(allowed_target_types)))]
        if not source_types or not target_types:
            diagnostics["relationships_dropped"] += 1
            diagnostics["relationships_missing_source_endpoint"] += int(not source_types)
            diagnostics["relationships_missing_target_endpoint"] += int(not target_types)
            if not source_types:
                decision["rejection_reasons"].append("source_entity_not_accepted")
            if not target_types:
                decision["rejection_reasons"].append("target_entity_not_accepted")
            fact_evidence_checks.append(decision)
            continue
        valid_signatures = [
            (source_type, target_type)
            for source_type in source_types
            for target_type in target_types
            if has_valid_relationship_signature(
                rec.rel_type,
                source_type,
                target_type,
            )
        ]
        if not valid_signatures:
            diagnostics["relationships_dropped"] += 1
            diagnostics["relationship_validation_errors"] += 1
            diagnostics["relationship_invalid_signatures"] += 1
            decision["rejection_reasons"].append("invalid_relationship_signature")
            fact_evidence_checks.append(decision)
            continue
        if len(valid_signatures) > 1:
            diagnostics["relationships_dropped"] += 1
            diagnostics["relationship_validation_errors"] += 1
            diagnostics["relationship_ambiguous_endpoint_types"] += 1
            decision["rejection_reasons"].append("ambiguous_endpoint_types")
            fact_evidence_checks.append(decision)
            continue
        source_type, target_type = valid_signatures[0]
        try:
            relationships.append(
                ExtractedRelationship(
                    type=rec.rel_type,
                    source_name=rec.source,
                    source_type=source_type,
                    target_name=rec.target,
                    target_type=target_type,
                    evidence_status=rec.evidence_status,
                    evidence_status_defaulted=rec.evidence_status_defaulted,
                    keywords=rec.keywords or None,
                    evidence=evidence,
                )
            )
        except ValidationError:
            diagnostics["relationships_dropped"] += 1
            diagnostics["relationship_validation_errors"] += 1
            decision["rejection_reasons"].append("invalid_relationship")
            fact_evidence_checks.append(decision)
            continue
        decision["accepted"] = True
        fact_evidence_checks.append(decision)

    entity_groups: dict[EntityType, list[ExtractedEntity]] = {
        "Startup": startups,
        "Investor": investors,
        "Person": people,
        "Topic": topics,
        "Company": companies,
    }
    diagnostics["entity_recoveries"] = _recover_relationship_backed_entities(
        raw_entities=raw.entities,
        relationships=relationships,
        article=article,
        entity_groups=entity_groups,
        fact_evidence_checks=fact_evidence_checks,
    )
    diagnostics["typed_entities"] = sum(len(entities) for entities in entity_groups.values())
    diagnostics["relationships_kept"] = len(relationships)
    diagnostics["relationships_kept_by_type"] = dict(
        sorted(
            {
                rel_type: sum(1 for relationship in relationships if relationship.type == rel_type)
                for rel_type in {relationship.type for relationship in relationships}
            }.items()
        )
    )

    return (
        ExtractionResult(
            startups=startups,
            investors=investors,
            people=people,
            topics=topics,
            companies=companies,
            relationships=relationships,
        ),
        diagnostics,
    )


def _recover_relationship_backed_entities(
    *,
    raw_entities: list[RawEntityRecord],
    relationships: list[ExtractedRelationship],
    article: ArticleIn,
    entity_groups: dict[EntityType, list[ExtractedEntity]],
    fact_evidence_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_by_key: dict[tuple[EntityType, str], RawEntityRecord] = {}
    for raw_entity in raw_entities:
        entity_type = _TYPE_MAP.get(raw_entity.entity_type)
        if entity_type is not None:
            raw_by_key.setdefault(
                (entity_type, raw_entity.name.casefold()),
                raw_entity,
            )

    support_by_key: dict[
        tuple[EntityType, str],
        tuple[ExtractedRelationship, str],
    ] = {}
    for relationship in relationships:
        evidence = (relationship.evidence or "").strip()
        if relationship.evidence_status not in ADMITTED_EVIDENCE_STATUSES or not evidence:
            continue
        for entity_type, name in (
            (relationship.source_type, relationship.source_name),
            (relationship.target_type, relationship.target_name),
        ):
            if entity_type == "Topic":
                continue
            key = (entity_type, name.casefold())
            previous = support_by_key.get(key)
            if (
                previous is None
                or strongest_evidence_status(
                    previous[0].evidence_status,
                    relationship.evidence_status,
                )
                != previous[0].evidence_status
            ):
                support_by_key[key] = (relationship, name)

    recoveries: list[dict[str, Any]] = []
    for (entity_type, normalized_name), (relationship, endpoint_name) in support_by_key.items():
        entities = entity_groups[entity_type]
        existing_index = next(
            (
                index
                for index, entity in enumerate(entities)
                if entity.name.casefold() == normalized_name
            ),
            None,
        )
        existing = entities[existing_index] if existing_index is not None else None
        if existing is not None and existing.evidence_status in ADMITTED_EVIDENCE_STATUSES:
            continue

        raw_entity = raw_by_key.get((entity_type, normalized_name))
        evidence = (relationship.evidence or "").strip()
        recovered = ExtractedEntity(
            name=(
                existing.name
                if existing is not None
                else raw_entity.name
                if raw_entity is not None
                else endpoint_name
            ),
            aliases=list(existing.aliases) if existing is not None else [],
            type_basis=(
                existing.type_basis
                if existing is not None
                else raw_entity.type_basis
                if raw_entity is not None
                else "contextual"
            ),
            evidence_status=relationship.evidence_status,
            evidence_status_defaulted=False,
            description=evidence,
            source=SourceAttribution(
                article_url=article.url,
                article_title=article.title,
                evidence=evidence,
            ),
        )
        mode = "restored" if existing is not None or raw_entity is not None else "synthesized"
        if existing_index is None:
            entities.append(recovered)
        else:
            entities[existing_index] = recovered

        recovery = {
            "name": recovered.name,
            "entity_type": entity_type,
            "mode": mode,
            "relationship": {
                "source": relationship.source_name,
                "relationship_type": relationship.type,
                "target": relationship.target_name,
            },
            "evidence_status": relationship.evidence_status,
            "evidence": evidence,
        }
        recoveries.append(recovery)
        for decision in fact_evidence_checks:
            fact = decision.get("fact", {})
            if (
                decision.get("kind") == "entity"
                and str(fact.get("name", "")).casefold() == normalized_name
                and _TYPE_MAP.get(str(fact.get("entity_type", ""))) == entity_type
            ):
                decision["recovery"] = recovery
                break

    return recoveries


def _evidence_validation_trace_payload(diagnostics: dict[str, Any]) -> dict[str, Any]:
    facts = diagnostics["fact_evidence_checks"]
    recoveries = diagnostics["entity_recoveries"]
    rejection_counts: dict[str, int] = {}
    for fact in facts:
        for reason in fact["rejection_reasons"]:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    checked = len(facts)
    matched = sum(fact["evidence_check"] == "matched" for fact in facts)
    accepted = sum(bool(fact["accepted"]) for fact in facts)
    return {
        "normalization": "exact character offsets from same-block sentence references",
        "summary": {
            "facts_checked": checked,
            "facts_accepted": accepted,
            "facts_rejected": checked - accepted,
            "evidence_matched": matched,
            "evidence_missing": sum(
                fact["evidence_check"] in {"missing", "missing_ref"} for fact in facts
            ),
            "evidence_not_in_article": sum(
                fact["evidence_check"] not in {"matched", "missing", "missing_ref"}
                for fact in facts
            ),
            "evidence_match_rate": round(matched / checked, 4) if checked else 0.0,
            "entities_recovered_from_relationship": len(recoveries),
            "entities_restored_from_relationship": sum(
                recovery["mode"] == "restored" for recovery in recoveries
            ),
            "entities_synthesized_from_relationship": sum(
                recovery["mode"] == "synthesized" for recovery in recoveries
            ),
            "rejections_by_reason": dict(sorted(rejection_counts.items())),
        },
        "facts": facts,
        "recoveries": recoveries,
    }


def _raw_conversion_diagnostics_detail(diagnostics: dict[str, Any]) -> str:
    return (
        f"raw_entities={diagnostics['raw_entities']}, "
        f"typed_entities={diagnostics['typed_entities']}, "
        f"entities_missing_evidence={diagnostics['entities_missing_evidence']}, "
        f"entities_unmatched_evidence={diagnostics['entities_unmatched_evidence']}, "
        f"entities_recovered_from_relationship={len(diagnostics['entity_recoveries'])}, "
        f"entity_type_conflicts={diagnostics['entity_type_conflicts']}, "
        f"raw_relationships={diagnostics['raw_relationships']}, "
        f"kept={diagnostics['relationships_kept']}, "
        f"dropped={diagnostics['relationships_dropped']}, "
        f"relationships_missing_evidence={diagnostics['relationships_missing_evidence']}, "
        f"relationships_unmatched_evidence={diagnostics['relationships_unmatched_evidence']}, "
        f"missing_source={diagnostics['relationships_missing_source_endpoint']}, "
        f"missing_target={diagnostics['relationships_missing_target_endpoint']}, "
        f"validation_errors={diagnostics['relationship_validation_errors']}, "
        f"invalid_signatures={diagnostics['relationship_invalid_signatures']}, "
        f"ambiguous_endpoint_types="
        f"{diagnostics['relationship_ambiguous_endpoint_types']}, "
        f"raw_by_type={_format_relationship_type_counts(diagnostics['raw_relationships_by_type'])}, "
        f"kept_by_type="
        f"{_format_relationship_type_counts(diagnostics['relationships_kept_by_type'])}"
    )
