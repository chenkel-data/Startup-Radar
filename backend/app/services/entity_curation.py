from __future__ import annotations

import asyncio
from contextvars import Context
import hashlib
import json
import unicodedata
from datetime import datetime
from time import perf_counter
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.provider import safe_set_span_in_context
from rapidfuzz.distance import Levenshtein

from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.entity_profile_store import EntityProfileStore
from app.observability.context import current_trace_id as _current_trace_id
from app.observability.llm_steps import (
    LLM_STEP_PROFILE_CURATION,
    LLM_STEP_PROFILE_REVIEW,
    llm_workflow_step,
)
from app.observability.traces import mlflow_trace_link
from app.prompts.entity_curation import (
    ENTITY_CURATION_SYSTEM_PROMPT,
    ENTITY_PROFILE_REVIEW_SYSTEM_PROMPT,
    build_entity_curation_prompt_registry_template,
    build_entity_curation_prompt,
    build_entity_profile_review_prompt_registry_template,
    build_entity_profile_review_prompt,
)
from app.services.embedding import EmbeddingService
from app.services.llm import LLMExtractionService


_REVIEW_KEYS = {
    "decision",
    "confidence",
    "reason",
    "evidence_refs_considered",
    "update_instructions",
}
_CURATION_KEYS = {"description", "confidence", "used_evidence_refs", "limitations"}
_REVIEW_DECISIONS = {
    "keep_profile",
    "update_profile",
    "insufficient_evidence",
    "possible_wrong_merge",
    "conflicting_evidence",
}
_CONFIDENCE_VALUES = {"high", "medium", "low"}
_KEEP_DECISIONS = {"keep_profile", "insufficient_evidence"}
_FLAG_DECISIONS = {"possible_wrong_merge", "conflicting_evidence"}
_MAX_PROFILE_EVIDENCE_ARTICLES = 5
_MAX_PROFILE_EVIDENCE_CHARS = 500
_MAX_PROFILE_MATCHES = 3
_MATCH_TEXT_PREVIEW_CHARS = 220
_MAX_PREVIOUS_CURATION_TRACES = 8
_MLFLOW_TRACE_NAME_TAG = "mlflow.traceName"
_MAX_TRANSLITERATION_EDIT_DISTANCE = 2


class EntityProfileCurationService:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMExtractionService,
        embedding: EmbeddingService | None,
        profile_store: EntityProfileStore,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.embedding = embedding
        self.profile_store = profile_store
        self.logger = get_logger("entity_curation")
        self._semaphore = asyncio.Semaphore(settings.entity_curation_max_concurrency)
        self._profile_review_prompt = None
        self._profile_curation_prompt = None
        self._profile_review_prompt_uri = getattr(
            settings,
            "mlflow_prompt_profile_review_uri",
            "prompts:/entity_profile_review@champion",
        )
        self._profile_curation_prompt_uri = getattr(
            settings,
            "mlflow_prompt_profile_curation_uri",
            "prompts:/entity_profile_curation@champion",
        )
        if getattr(settings, "mlflow_use_prompt_registry", False):
            self._profile_review_prompt = self.llm._load_prompt_at_startup(
                self._profile_review_prompt_uri
            )
            self._profile_curation_prompt = self.llm._load_prompt_at_startup(
                self._profile_curation_prompt_uri
            )
        self.profile_curation_policy_hash = _profile_curation_policy_hash(
            review_prompt=self._profile_review_prompt,
            review_uri=self._profile_review_prompt_uri,
            curation_prompt=self._profile_curation_prompt,
            curation_uri=self._profile_curation_prompt_uri,
        )

    async def curate_profiles(
        self,
        outcome_rows: list[dict[str, Any]],
        *,
        job_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        touched_ids = sorted({row["entity_id"] for row in outcome_rows if row.get("entity_id")})
        self.logger.info(
            "entity_profile_touched_entities_total",
            extra={
                "event": "curation",
                "workflow_step": "touched_entities",
                "count": len(touched_ids),
            },
        )
        with mlflow.start_span(name="load_profile_candidates", span_type=SpanType.TOOL) as span:
            _set_span_inputs(
                span,
                {
                    "job_run_id": job_run_id,
                    "touched_entity_count": len(touched_ids),
                    "candidate_scope": "all_graph_entities",
                    "profile_curation_policy_hash": self.profile_curation_policy_hash,
                },
            )
            candidates = await self.profile_store.profile_review_inputs(
                profile_curation_policy_hash=self.profile_curation_policy_hash,
            )
            _set_span_outputs(
                span,
                {
                    "candidate_count": len(candidates),
                    "touched_candidates": sum(
                        1 for candidate in candidates if candidate.get("id") in touched_ids
                    ),
                    "candidate_ids": [candidate.get("id") for candidate in candidates],
                },
            )
        candidate_ids = {row.get("id") for row in candidates}
        self.logger.info(
            "entity_profile_candidates_loaded",
            extra={
                "event": "curation",
                "workflow_step": "profile_candidates",
                "count": len(candidates),
                "detail": (
                    f"touched={len(touched_ids)}, candidates={len(candidates)}, "
                    f"touched_candidates={len(candidate_ids.intersection(touched_ids))}"
                ),
            },
        )
        tasks = [self._process_entity(entity, job_run_id=job_run_id) for entity in candidates]
        return await asyncio.gather(*tasks) if tasks else []

    async def curate_entity_ids(
        self,
        entity_ids: list[str],
        *,
        job_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        scoped_ids = sorted({entity_id for entity_id in entity_ids if entity_id})
        if not scoped_ids:
            return []

        self.logger.info(
            "entity_profile_touched_entities_total",
            extra={
                "event": "curation",
                "workflow_step": "touched_entities",
                "count": len(scoped_ids),
            },
        )
        with mlflow.start_span(name="load_profile_candidates", span_type=SpanType.TOOL) as span:
            _set_span_inputs(
                span,
                {
                    "job_run_id": job_run_id,
                    "touched_entity_count": len(scoped_ids),
                    "candidate_scope": "resolved_entities",
                    "entity_ids": scoped_ids,
                    "profile_curation_policy_hash": self.profile_curation_policy_hash,
                },
            )
            candidates = await self.profile_store.profile_review_inputs(
                profile_curation_policy_hash=self.profile_curation_policy_hash,
                entity_ids=scoped_ids,
            )
            _set_span_outputs(
                span,
                {
                    "candidate_count": len(candidates),
                    "touched_candidates": len(candidates),
                    "candidate_ids": [candidate.get("id") for candidate in candidates],
                },
            )
        self.logger.info(
            "entity_profile_candidates_loaded",
            extra={
                "event": "curation",
                "workflow_step": "profile_candidates",
                "count": len(candidates),
                "detail": (
                    f"scope=resolved_entities, touched={len(scoped_ids)}, "
                    f"candidates={len(candidates)}"
                ),
            },
        )
        tasks = [self._process_entity(entity, job_run_id=job_run_id) for entity in candidates]
        return await asyncio.gather(*tasks) if tasks else []

    async def _process_entity(
        self,
        entity: dict[str, Any],
        *,
        job_run_id: str | None,
    ) -> dict[str, Any]:
        async with self._semaphore:
            try:
                return await self._process_entity_traced(entity, job_run_id=job_run_id)
            except Exception as exc:
                self.logger.warning(
                    "entity_profile_curation_failed",
                    extra={
                        "event": "curation",
                        "workflow_step": "entity_profile_curation",
                        "error": str(exc),
                        **_entity_log_fields(entity),
                        "detail": str(exc)[:300],
                    },
                )
                return {
                    "entity_id": entity.get("id"),
                    "name": entity.get("name"),
                    "label": entity.get("label"),
                    "status": "failed",
                    "error": str(exc)[:300],
                }

    async def _process_entity_traced(
        self,
        entity: dict[str, Any],
        *,
        job_run_id: str | None,
    ) -> dict[str, Any]:
        root_span = Context().run(
            lambda: mlflow.start_span_no_context(
                name="process_entity_profile",
                span_type=SpanType.CHAIN,
            )
        )
        try:
            with safe_set_span_in_context(root_span):
                return await self._process_entity_profile_span(
                    entity,
                    job_run_id=job_run_id,
                )
        except Exception as exc:
            try:
                root_span.record_exception(exc)
            except Exception:  # pragma: no cover - tracing must not break ingestion
                pass
            raise
        finally:
            try:
                root_span.end()
            except Exception:  # pragma: no cover - tracing must not break ingestion
                pass

    async def _process_entity_profile_span(
        self,
        entity: dict[str, Any],
        *,
        job_run_id: str | None,
    ) -> dict[str, Any]:
        trace_id = _current_trace_id()
        mlflow_trace_url, mlflow_experiment_id = mlflow_trace_link(self.settings, trace_id)
        trace_context = {
            "trace_id": trace_id,
            "mlflow_trace_url": mlflow_trace_url,
            "mlflow_experiment_id": mlflow_experiment_id,
        }
        previous_curation_traces = _previous_curation_traces(entity)
        entity_name, entity_type, request_label = _entity_trace_labels(entity)
        _update_current_trace(
            tags={
                _MLFLOW_TRACE_NAME_TAG: entity_name,
                "job_run_id": job_run_id or "",
                "workflow": "entity_profile_curation",
                "entity_id": str(entity.get("id") or ""),
                "entity_name": entity_name,
                "entity_type": entity_type,
                "model": self.settings.openai_model,
                "profile_curation_policy_hash": self.profile_curation_policy_hash,
            },
            request_preview=request_label,
        )

        match_summary = build_profile_evidence_match_summary(entity)
        prompt_entity = {
            **entity,
            "new_evidence": match_summary["new_evidence"],
        }
        (
            profile_context,
            evidence_ref_map,
            article_ref_map,
            evidence_sources,
        ) = build_profile_prompt_context(prompt_entity)
        span = mlflow.get_current_active_span()
        evidence_ids = _evidence_ids_for_refs(evidence_ref_map.keys(), evidence_ref_map)
        current_description = str(entity.get("current_description") or "").strip()
        _set_span_inputs(
            span,
            {
                "job_run_id": job_run_id,
                "entity_id": entity.get("id"),
                "entity_name": entity.get("name"),
                "entity_type": entity.get("label"),
                "has_current_description": bool(current_description),
                "raw_new_evidence_count": len(entity.get("new_evidence") or []),
                "exact_duplicate_evidence_count": match_summary["exact_duplicate_count"],
                "existing_considered_evidence_count": match_summary["existing_evidence_count"],
                "prompt_evidence_count": len(profile_context["new_evidence"]),
                "prompt_evidence_article_count": len(article_ref_map),
                "best_existing_evidence_similarity": match_summary["best_similarity_score"],
                "has_embedding": bool(entity.get("has_embedding")),
                "previous_curation_trace_count": len(previous_curation_traces),
            },
        )
        if previous_curation_traces:
            with mlflow.start_span(
                name="previous_curation_traces",
                span_type=SpanType.TOOL,
            ) as history_span:
                _set_span_inputs(
                    history_span,
                    {
                        "entity_id": entity.get("id"),
                        "entity_name": entity.get("name"),
                    },
                )
                _set_span_outputs(
                    history_span,
                    {
                        "count": len(previous_curation_traces),
                        "traces": previous_curation_traces,
                    },
                )
        with mlflow.start_span(
            name="profile_evidence_context", span_type=SpanType.TOOL
        ) as context_span:
            _set_span_outputs(
                context_span,
                {
                    "profile_context": profile_context,
                    "evidence_sources": evidence_sources,
                    "evidence_ids": evidence_ids,
                    "duplicate_evidence_ids": match_summary["exact_duplicate_evidence_ids"],
                    "duplicate_article_ids": match_summary["exact_duplicate_article_ids"],
                    "match_metrics": {
                        "raw_new_evidence_count": len(entity.get("new_evidence") or []),
                        "existing_considered_evidence_count": match_summary[
                            "existing_evidence_count"
                        ],
                        "exact_duplicate_evidence_count": match_summary["exact_duplicate_count"],
                        "prompt_evidence_count": len(profile_context["new_evidence"]),
                        "best_existing_evidence_similarity": match_summary["best_similarity_score"],
                    },
                    "top_existing_evidence_matches": match_summary["top_matches"],
                },
            )

        result = await self._process_entity_inner(
            entity,
            profile_context=profile_context,
            evidence_ref_map=evidence_ref_map,
            article_ref_map=article_ref_map,
            evidence_sources=evidence_sources,
            duplicate_evidence_ids=match_summary["exact_duplicate_evidence_ids"],
            duplicate_article_ids=match_summary["exact_duplicate_article_ids"],
            match_summary=match_summary,
            trace_context=trace_context,
        )
        result["trace_id"] = trace_id
        if mlflow_trace_url:
            result["trace_url"] = mlflow_trace_url
        _set_span_outputs(span, _trace_result_summary(result))
        _update_current_trace(response_preview=_trace_response_preview(result))
        return result

    async def _process_entity_inner(
        self,
        entity: dict[str, Any],
        *,
        profile_context: dict[str, Any],
        evidence_ref_map: dict[str, list[str]],
        article_ref_map: dict[str, str],
        evidence_sources: list[dict[str, str]],
        duplicate_evidence_ids: list[str],
        duplicate_article_ids: list[str],
        match_summary: dict[str, Any],
        trace_context: dict[str, str | None],
    ) -> dict[str, Any]:
        evidence_refs = list(evidence_ref_map)
        evidence_ids = _evidence_ids_for_refs(evidence_refs, evidence_ref_map)
        article_ids = _article_ids_for_refs(evidence_refs, article_ref_map)
        current_description = str(entity.get("current_description") or "").strip()
        self.logger.info(
            "entity_profile_processing_started",
            extra={
                "event": "curation",
                "workflow_step": "entity_profile_curation",
                **_entity_log_fields(entity),
                "detail": (
                    f"prompt_evidence={len(evidence_ids)}, "
                    f"exact_duplicates={len(duplicate_evidence_ids)}, "
                    f"articles={len(article_ids)}, "
                    f"duplicate_articles={len(duplicate_article_ids)}, "
                    f"best_similarity={match_summary['best_similarity_score']}, "
                    f"has_description={bool(current_description)}, "
                    f"has_embedding={bool(entity.get('has_embedding'))}"
                ),
            },
        )

        if not evidence_ids:
            if duplicate_evidence_ids:
                return await self._save_exact_duplicate_skip(
                    entity,
                    duplicate_evidence_ids=duplicate_evidence_ids,
                    duplicate_article_ids=duplicate_article_ids,
                    current_description=current_description,
                    trace_context=trace_context,
                )
            return await self._refresh_embedding_or_skip(entity, current_description)

        if not current_description:
            self.logger.info(
                "entity_profile_review_completed",
                extra={
                    "event": "curation",
                    "workflow_step": "profile_review",
                    **_entity_log_fields(entity),
                    "decision": "update_profile",
                    "confidence": "high",
                    "detail": "profile missing; curation required without review LLM",
                },
            )
            review = _direct_update_review(evidence_refs)
            return await self._curate_and_save(
                entity,
                review,
                profile_context=profile_context,
                evidence_ref_map=evidence_ref_map,
                article_ref_map=article_ref_map,
                duplicate_evidence_ids=duplicate_evidence_ids,
                duplicate_article_ids=duplicate_article_ids,
                trace_context=trace_context,
            )

        review = await self._review_profile(
            entity,
            profile_context,
            evidence_sources=evidence_sources,
        )
        considered_refs = _known_refs(
            review.get("evidence_refs_considered"),
            list(evidence_ref_map),
            fallback=list(evidence_ref_map),
        )
        considered_ids = _merge_unique(
            duplicate_evidence_ids,
            _evidence_ids_for_refs(considered_refs, evidence_ref_map),
        )
        considered_article_ids = _merge_unique(
            duplicate_article_ids,
            _article_ids_for_refs(considered_refs, article_ref_map),
        )
        review["evidence_refs_considered"] = considered_refs

        if review["decision"] in _KEEP_DECISIONS:
            with mlflow.start_span(
                name="update_profile_embedding", span_type=SpanType.TOOL
            ) as span:
                _set_span_inputs(
                    span,
                    {
                        "entity_id": entity["id"],
                        "old_embedding_input_hash": entity.get("embedding_input_hash"),
                    },
                )
                embedding, embedding_hash = await self._embedding_if_needed(
                    entity, current_description
                )
                _set_span_outputs(
                    span,
                    {
                        "embedding_updated": embedding is not None,
                        "new_embedding_input_hash": embedding_hash,
                    },
                )
            with mlflow.start_span(
                name="save_profile_review_keep", span_type=SpanType.TOOL
            ) as span:
                _set_span_inputs(
                    span,
                    {
                        "entity_id": entity["id"],
                        "evidence_ids": considered_ids,
                        "decision": review["decision"],
                    },
                )
                await self.profile_store.save_profile_review_keep(
                    entity_id=entity["id"],
                    evidence_ids=considered_ids,
                    article_ids=considered_article_ids,
                    profile_curation_policy_hash=self.profile_curation_policy_hash,
                    review=review,
                    model=self.settings.openai_model,
                    embedding=embedding,
                    embedding_input_hash=embedding_hash,
                    **trace_context,
                )
                _set_span_outputs(span, {"status": "kept"})
            self.logger.info(
                "entity_profile_kept",
                extra={
                    "event": "curation",
                    "workflow_step": "entity_profile_curation",
                    **_entity_log_fields(entity),
                    "decision": review["decision"],
                    "confidence": review["confidence"],
                    "detail": _review_detail(review, considered_ids, embedding is not None),
                },
            )
            return {
                "entity_id": entity["id"],
                "name": entity.get("name"),
                "label": entity.get("label"),
                "status": "kept",
                "review_decision": review["decision"],
                "confidence": review["confidence"],
                "reason": review["reason"],
                "evidence_ids": considered_ids,
                "embedding_updated": embedding is not None,
            }

        if review["decision"] == "update_profile":
            return await self._curate_and_save(
                entity,
                review,
                profile_context=profile_context,
                evidence_ref_map=evidence_ref_map,
                article_ref_map=article_ref_map,
                duplicate_evidence_ids=duplicate_evidence_ids,
                duplicate_article_ids=duplicate_article_ids,
                trace_context=trace_context,
            )

        if review["decision"] in _FLAG_DECISIONS:
            with mlflow.start_span(name="flag_profile_for_review", span_type=SpanType.TOOL) as span:
                _set_span_inputs(
                    span,
                    {
                        "entity_id": entity["id"],
                        "evidence_ids": considered_ids,
                        "decision": review["decision"],
                    },
                )
                await self.profile_store.flag_profile_for_review(
                    entity_id=entity["id"],
                    evidence_ids=considered_ids,
                    article_ids=considered_article_ids,
                    profile_curation_policy_hash=self.profile_curation_policy_hash,
                    review=review,
                    model=self.settings.openai_model,
                    **trace_context,
                )
                _set_span_outputs(span, {"status": "needs_human_review"})
            self.logger.warning(
                "entity_profile_flagged",
                extra={
                    "event": "curation",
                    "workflow_step": "entity_profile_curation",
                    **_entity_log_fields(entity),
                    "decision": review["decision"],
                    "confidence": review["confidence"],
                    "detail": _review_detail(review, considered_ids, embedding_updated=False),
                },
            )
            return {
                "entity_id": entity["id"],
                "name": entity.get("name"),
                "label": entity.get("label"),
                "status": "needs_human_review",
                "review_decision": review["decision"],
                "confidence": review["confidence"],
                "reason": review["reason"],
                "evidence_ids": considered_ids,
            }

        raise ValueError(f"Unsupported review decision: {review['decision']}")

    async def _save_exact_duplicate_skip(
        self,
        entity: dict[str, Any],
        *,
        duplicate_evidence_ids: list[str],
        duplicate_article_ids: list[str],
        current_description: str,
        trace_context: dict[str, str | None],
    ) -> dict[str, Any]:
        decision = "exact_duplicate_evidence"
        review = {
            "decision": decision,
            "confidence": "high",
            "reason": (
                "All new profile evidence exactly duplicates the current profile "
                "or evidence that was already considered for this entity."
            ),
            "evidence_refs_considered": [],
            "update_instructions": [],
        }
        embedding = None
        embedding_hash = None
        if current_description:
            with mlflow.start_span(
                name="update_profile_embedding", span_type=SpanType.TOOL
            ) as span:
                _set_span_inputs(
                    span,
                    {
                        "entity_id": entity["id"],
                        "old_embedding_input_hash": entity.get("embedding_input_hash"),
                    },
                )
                embedding, embedding_hash = await self._embedding_if_needed(
                    entity, current_description
                )
                _set_span_outputs(
                    span,
                    {
                        "embedding_updated": embedding is not None,
                        "new_embedding_input_hash": embedding_hash,
                    },
                )
        with mlflow.start_span(
            name="save_exact_duplicate_profile_evidence", span_type=SpanType.TOOL
        ) as span:
            _set_span_inputs(
                span,
                {
                    "entity_id": entity["id"],
                    "evidence_ids": duplicate_evidence_ids,
                    "article_ids": duplicate_article_ids,
                    "decision": decision,
                },
            )
            await self.profile_store.save_profile_review_keep(
                entity_id=entity["id"],
                evidence_ids=duplicate_evidence_ids,
                article_ids=duplicate_article_ids,
                profile_curation_policy_hash=self.profile_curation_policy_hash,
                review=review,
                model=self.settings.openai_model,
                embedding=embedding,
                embedding_input_hash=embedding_hash,
                **trace_context,
            )
            _set_span_outputs(span, {"status": "skipped_exact_duplicate_evidence"})
        self.logger.info(
            "entity_profile_exact_duplicate_evidence_skipped",
            extra={
                "event": "curation",
                "workflow_step": "entity_profile_curation",
                **_entity_log_fields(entity),
                "decision": decision,
                "confidence": "high",
                "detail": (
                    f"exact_duplicates={len(duplicate_evidence_ids)}, "
                    f"duplicate_articles={len(duplicate_article_ids)}, "
                    f"embedding_updated={embedding is not None}"
                ),
            },
        )
        return {
            "entity_id": entity["id"],
            "name": entity.get("name"),
            "label": entity.get("label"),
            "status": "skipped_exact_duplicate_evidence",
            "review_decision": decision,
            "confidence": "high",
            "reason": review["reason"],
            "evidence_ids": duplicate_evidence_ids,
            "article_ids": duplicate_article_ids,
            "embedding_updated": embedding is not None,
        }

    def _render_profile_review_prompt(
        self,
        profile_context: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        variables = {
            "entity_json": json.dumps(
                profile_context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        }
        if self._profile_review_prompt is not None:
            cached = self.llm._render_prompt_from_cache(
                self._profile_review_prompt,
                variables,
                self._profile_review_prompt_uri,
            )
            if cached is not None:
                return cached
        return (
            ENTITY_PROFILE_REVIEW_SYSTEM_PROMPT,
            build_entity_profile_review_prompt(profile_context),
            {"prompt_uri": None, "prompt_source": "local_fallback"},
        )

    def _render_profile_curation_prompt(
        self,
        profile_context: dict[str, Any],
        review: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        variables = {
            "entity_json": json.dumps(
                profile_context,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "review_json": json.dumps(
                review,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        }
        if self._profile_curation_prompt is not None:
            cached = self.llm._render_prompt_from_cache(
                self._profile_curation_prompt,
                variables,
                self._profile_curation_prompt_uri,
            )
            if cached is not None:
                return cached
        return (
            ENTITY_CURATION_SYSTEM_PROMPT,
            build_entity_curation_prompt(profile_context, review),
            {"prompt_uri": None, "prompt_source": "local_fallback"},
        )

    async def _review_profile(
        self,
        entity: dict[str, Any],
        profile_context: dict[str, Any],
        *,
        evidence_sources: list[dict[str, str]],
    ) -> dict[str, Any]:
        started = perf_counter()
        self.logger.info(
            "entity_profile_review_requested",
            extra={
                "event": "curation",
                "workflow_step": "profile_review",
                "mode": "openai",
                "model": self.settings.openai_model,
                **_entity_log_fields(entity),
                "detail": f"prompt_evidence={len(profile_context['new_evidence'])}",
            },
        )
        system_prompt, prompt, prompt_metadata = self._render_profile_review_prompt(profile_context)
        with mlflow.start_span(name="review_entity_profile", span_type=SpanType.CHAIN) as span:
            _set_span_inputs(
                span,
                {
                    "entity_id": entity.get("id"),
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("label"),
                    "profile_context": profile_context,
                    "evidence_sources": evidence_sources,
                },
            )
            _set_span_attributes(span, prompt_metadata)
            with llm_workflow_step(LLM_STEP_PROFILE_REVIEW):
                raw = await self.llm._call_llm_raw(
                    system_prompt,
                    prompt,
                )
            payload = _parse_json_object(raw)
            if not _valid_review_payload(payload):
                raise ValueError(f"Invalid profile review payload: {raw[:300]}")
            _set_span_outputs(
                span,
                {
                    "decision": payload["decision"],
                    "confidence": payload["confidence"],
                    "reason": payload["reason"],
                    "evidence_refs_considered": payload["evidence_refs_considered"],
                    "update_instructions": payload["update_instructions"],
                },
            )
        duration_ms = round((perf_counter() - started) * 1000, 2)
        self.logger.info(
            "entity_profile_review_completed",
            extra={
                "event": "curation",
                "workflow_step": "profile_review",
                "mode": "openai",
                "model": self.settings.openai_model,
                **_entity_log_fields(entity),
                "decision": payload["decision"],
                "confidence": payload["confidence"],
                "duration_ms": duration_ms,
                "detail": _truncate_for_log(str(payload.get("reason") or ""), 260),
            },
        )
        return payload

    async def _curate_and_save(
        self,
        entity: dict[str, Any],
        review: dict[str, Any],
        *,
        profile_context: dict[str, Any],
        evidence_ref_map: dict[str, list[str]],
        article_ref_map: dict[str, str],
        duplicate_evidence_ids: list[str],
        duplicate_article_ids: list[str],
        trace_context: dict[str, str | None],
    ) -> dict[str, Any]:
        started = perf_counter()
        self.logger.info(
            "entity_profile_curation_requested",
            extra={
                "event": "curation",
                "workflow_step": "profile_curation",
                "mode": "openai",
                "model": self.settings.openai_model,
                **_entity_log_fields(entity),
                "decision": review.get("decision"),
                "confidence": review.get("confidence"),
                "detail": f"prompt_evidence={len(profile_context['new_evidence'])}",
            },
        )
        system_prompt, prompt, prompt_metadata = self._render_profile_curation_prompt(
            profile_context,
            review,
        )
        with mlflow.start_span(name="curate_entity_profile", span_type=SpanType.CHAIN) as span:
            _set_span_inputs(
                span,
                {
                    "entity_id": entity.get("id"),
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("label"),
                    "review": review,
                    "profile_context": profile_context,
                },
            )
            _set_span_attributes(span, prompt_metadata)
            with llm_workflow_step(LLM_STEP_PROFILE_CURATION):
                raw = await self.llm._call_llm_raw(
                    system_prompt,
                    prompt,
                )
            curation = _parse_json_object(raw)
            if not _valid_curation_payload(curation):
                raise ValueError(f"Invalid profile curation payload: {raw[:300]}")
            _set_span_outputs(
                span,
                {
                    "description": curation["description"],
                    "confidence": curation["confidence"],
                    "used_evidence_refs": curation["used_evidence_refs"],
                    "limitations": curation["limitations"],
                },
            )

        description = curation["description"].strip()
        all_refs = list(evidence_ref_map)
        considered_refs = _known_refs(
            review.get("evidence_refs_considered"),
            all_refs,
            fallback=all_refs,
        )
        used_refs = _known_refs(
            curation.get("used_evidence_refs"),
            all_refs,
            fallback=considered_refs,
        )
        review["evidence_refs_considered"] = considered_refs
        curation["used_evidence_refs"] = used_refs
        considered_ids = _merge_unique(
            duplicate_evidence_ids,
            _evidence_ids_for_refs(considered_refs, evidence_ref_map),
        )
        used_ids = _evidence_ids_for_refs(used_refs, evidence_ref_map)
        considered_article_ids = _merge_unique(
            duplicate_article_ids,
            _article_ids_for_refs(considered_refs, article_ref_map),
        )
        with mlflow.start_span(name="update_profile_embedding", span_type=SpanType.TOOL) as span:
            _set_span_inputs(
                span,
                {
                    "entity_id": entity["id"],
                    "old_embedding_input_hash": entity.get("embedding_input_hash"),
                },
            )
            embedding, embedding_hash = await self._embedding_for_description(entity, description)
            _set_span_outputs(
                span,
                {
                    "embedding_updated": embedding is not None,
                    "new_embedding_input_hash": embedding_hash,
                },
            )
        before = entity.get("current_description")
        with mlflow.start_span(name="save_profile_revision", span_type=SpanType.TOOL) as span:
            _set_span_inputs(
                span,
                {
                    "entity_id": entity["id"],
                    "evidence_ids_considered": considered_ids,
                    "used_evidence_ids": used_ids,
                    "description": description,
                },
            )
            await self.profile_store.save_profile_revision(
                entity_id=entity["id"],
                description=description,
                evidence_ids_considered=considered_ids,
                used_evidence_ids=used_ids,
                article_ids_considered=considered_article_ids,
                profile_curation_policy_hash=self.profile_curation_policy_hash,
                review=review,
                curation=curation,
                model=self.settings.openai_model,
                embedding=embedding,
                embedding_input_hash=embedding_hash,
                **trace_context,
            )
            _set_span_outputs(
                span,
                {
                    "status": "updated",
                    "used_evidence_count": len(used_ids),
                    "embedding_updated": embedding is not None,
                },
            )
        duration_ms = round((perf_counter() - started) * 1000, 2)
        self.logger.info(
            "entity_profile_curation_completed",
            extra={
                "event": "curation",
                "workflow_step": "profile_curation",
                "mode": "openai",
                "model": self.settings.openai_model,
                **_entity_log_fields(entity),
                "status": "updated",
                "decision": review.get("decision"),
                "confidence": curation.get("confidence"),
                "duration_ms": duration_ms,
                "detail": (
                    f"used_evidence={len(used_ids)}, embedding_updated={embedding is not None}"
                ),
            },
        )
        return {
            "entity_id": entity["id"],
            "name": entity.get("name"),
            "label": entity.get("label"),
            "status": "updated",
            "review_decision": review.get("decision"),
            "before": before,
            "after": description,
            "confidence": curation.get("confidence"),
            "reason": review.get("reason"),
            "evidence_ids": considered_ids,
            "used_evidence_ids": used_ids,
            "embedding_updated": embedding is not None,
        }

    async def _refresh_embedding_or_skip(
        self,
        entity: dict[str, Any],
        current_description: str,
    ) -> dict[str, Any]:
        if not current_description:
            self.logger.info(
                "entity_profile_skipped",
                extra={
                    "event": "curation",
                    "workflow_step": "entity_profile_curation",
                    **_entity_log_fields(entity),
                    "status": "skipped_no_profile_or_evidence",
                    "detail": "no profile description and no new evidence",
                },
            )
            return {
                "entity_id": entity.get("id"),
                "name": entity.get("name"),
                "label": entity.get("label"),
                "status": "skipped_no_profile_or_evidence",
            }
        with mlflow.start_span(name="update_profile_embedding", span_type=SpanType.TOOL) as span:
            _set_span_inputs(
                span,
                {
                    "entity_id": entity["id"],
                    "old_embedding_input_hash": entity.get("embedding_input_hash"),
                },
            )
            embedding, embedding_hash = await self._embedding_if_needed(entity, current_description)
            _set_span_outputs(
                span,
                {
                    "embedding_updated": embedding is not None,
                    "new_embedding_input_hash": embedding_hash,
                },
            )
        if embedding is None:
            self.logger.info(
                "entity_profile_embedding_skipped",
                extra={
                    "event": "curation",
                    "workflow_step": "profile_embedding",
                    **_entity_log_fields(entity),
                    "status": "skipped_no_new_profile_evidence",
                    "detail": "embedding input unchanged",
                },
            )
            return {
                "entity_id": entity.get("id"),
                "name": entity.get("name"),
                "label": entity.get("label"),
                "status": "skipped_no_new_profile_evidence",
            }
        with mlflow.start_span(name="save_profile_embedding", span_type=SpanType.TOOL) as span:
            _set_span_inputs(
                span,
                {
                    "entity_id": entity["id"],
                    "embedding_input_hash": embedding_hash,
                },
            )
            await self.profile_store.save_profile_embedding(
                entity_id=entity["id"],
                embedding=embedding,
                embedding_input_hash=embedding_hash or "",
                model=self.settings.openai_model,
            )
            _set_span_outputs(span, {"status": "embedding_refreshed"})
        self.logger.info(
            "entity_profile_embedding_updated",
            extra={
                "event": "curation",
                "workflow_step": "profile_embedding",
                **_entity_log_fields(entity),
                "status": "embedding_refreshed",
                "detail": "profile embedding refreshed without LLM review",
            },
        )
        return {
            "entity_id": entity["id"],
            "name": entity.get("name"),
            "label": entity.get("label"),
            "status": "embedding_refreshed",
            "embedding_updated": True,
        }

    async def _embedding_if_needed(
        self,
        entity: dict[str, Any],
        description: str,
    ) -> tuple[list[float] | None, str | None]:
        text = profile_embedding_text(entity, description)
        input_hash = _hash_text(text)
        if entity.get("has_embedding") and entity.get("embedding_input_hash") == input_hash:
            return None, None
        return await self._embedding_for_text(text), input_hash

    async def _embedding_for_description(
        self,
        entity: dict[str, Any],
        description: str,
    ) -> tuple[list[float] | None, str | None]:
        text = profile_embedding_text(entity, description)
        input_hash = _hash_text(text)
        return await self._embedding_for_text(text), input_hash

    async def _embedding_for_text(self, text: str) -> list[float] | None:
        if self.embedding is None:
            return None
        return await self.embedding.embed_one(text)


def profile_embedding_text(entity: dict[str, Any], description: str) -> str:
    aliases = ", ".join(str(alias) for alias in entity.get("aliases") or [])
    return (
        f"Entity type: {entity.get('label')}\n"
        f"Name: {entity.get('name')}\n"
        f"Aliases: {aliases}\n"
        f"Public profile: {description}"
    ).strip()


def _entity_trace_labels(entity: dict[str, Any]) -> tuple[str, str, str]:
    entity_name = str(
        entity.get("name") or entity.get("canonical_name") or entity.get("id") or "Unknown entity"
    )
    entity_type = str(entity.get("label") or "Entity")
    return entity_name, entity_type, f"{entity_type} | {entity_name}"


def _previous_curation_traces(entity: dict[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw_event in entity.get("description_curation_history") or []:
        event = _history_event(raw_event)
        if event is None:
            continue
        record = _curation_trace_record(
            status=event.get("status"),
            decision=event.get("decision"),
            confidence=event.get("confidence"),
            checked_at=event.get("checked_at") or event.get("checkedAt"),
            trace_id=event.get("trace_id") or event.get("traceId"),
            trace_url=(
                event.get("mlflow_trace_url") or event.get("trace_url") or event.get("traceUrl")
            ),
            experiment_id=(
                event.get("mlflow_experiment_id")
                or event.get("experiment_id")
                or event.get("experimentId")
            ),
        )
        if record is not None:
            records.append(record)

    fallback = _curation_trace_record(
        status=entity.get("description_curation_status"),
        decision=entity.get("description_review_decision"),
        confidence=entity.get("description_confidence"),
        checked_at=entity.get("description_traced_at"),
        trace_id=entity.get("description_trace_id"),
        trace_url=entity.get("description_trace_url"),
        experiment_id=entity.get("description_mlflow_experiment_id"),
    )
    if fallback is not None:
        records.append(fallback)

    deduped: dict[str, dict[str, str]] = {}
    for record in sorted(
        records,
        key=lambda item: _timestamp_sort_value(item.get("checked_at")),
        reverse=True,
    ):
        key = record.get("trace_id") or record.get("trace_url")
        if key is None:
            continue
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = record
            continue
        for field, value in record.items():
            existing.setdefault(field, value)

    return list(deduped.values())[:_MAX_PREVIOUS_CURATION_TRACES]


def _history_event(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return value if isinstance(value, dict) else None


def _curation_trace_record(
    *,
    status: Any,
    decision: Any,
    confidence: Any,
    checked_at: Any,
    trace_id: Any,
    trace_url: Any,
    experiment_id: Any,
) -> dict[str, str] | None:
    record = {
        "status": _string_or_none(status),
        "decision": _string_or_none(decision),
        "confidence": _string_or_none(confidence),
        "checked_at": _string_or_none(checked_at),
        "trace_id": _string_or_none(trace_id),
        "trace_url": _string_or_none(trace_url),
        "mlflow_experiment_id": _string_or_none(experiment_id),
    }
    if not record["trace_id"] and not record["trace_url"]:
        return None
    return {key: value for key, value in record.items() if value}


def _string_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _timestamp_sort_value(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return 0.0


def _update_current_trace(
    *,
    tags: dict[str, str] | None = None,
    request_preview: str | None = None,
    response_preview: str | None = None,
) -> None:
    try:
        kwargs: dict[str, Any] = {}
        if tags is not None:
            kwargs["tags"] = tags
        if request_preview is not None:
            kwargs["request_preview"] = request_preview
        if response_preview is not None:
            kwargs["response_preview"] = response_preview
        if kwargs:
            mlflow.update_current_trace(**kwargs)
    except Exception:  # pragma: no cover - tracing must not break ingestion
        pass


def _set_span_attributes(span: Any, payload: dict[str, Any]) -> None:
    if span is None:
        return
    attrs = {key: value for key, value in payload.items() if value is not None}
    if not attrs:
        return
    try:
        span.set_attributes(attrs)
    except Exception:  # pragma: no cover - tracing must not break ingestion
        pass


def _set_span_inputs(span: Any, payload: dict[str, Any]) -> None:
    if span is None:
        return
    try:
        span.set_inputs(payload)
    except Exception:  # pragma: no cover - tracing must not break ingestion
        pass


def _set_span_outputs(span: Any, payload: dict[str, Any]) -> None:
    if span is None:
        return
    try:
        span.set_outputs(payload)
    except Exception:  # pragma: no cover - tracing must not break ingestion
        pass


def _trace_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": result.get("entity_id"),
        "name": result.get("name"),
        "label": result.get("label"),
        "status": result.get("status"),
        "review_decision": result.get("review_decision"),
        "confidence": result.get("confidence"),
        "embedding_updated": result.get("embedding_updated"),
        "trace_id": result.get("trace_id"),
        "trace_url": result.get("trace_url"),
    }


def _trace_response_preview(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "unknown")
    decision = str(result.get("review_decision") or "")
    confidence = str(result.get("confidence") or "")
    if decision == "insufficient_evidence":
        return f"{decision} | decision_confidence={confidence}" if confidence else decision
    return f"{status} | {confidence}" if confidence else status


def _entity_log_fields(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_name": entity.get("name") or entity.get("canonical_name") or entity.get("id"),
        "entity_type": entity.get("label"),
    }


def _review_detail(
    review: dict[str, Any],
    evidence_ids: list[str],
    embedding_updated: bool,
) -> str:
    reason = _truncate_for_log(str(review.get("reason") or ""), 260)
    return f"evidence={len(evidence_ids)}, embedding_updated={embedding_updated}, reason={reason}"


def _truncate_for_log(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return f"{value[: max(limit - 3, 0)]}..."


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload


def _valid_review_payload(payload: dict[str, Any]) -> bool:
    return (
        set(payload) == _REVIEW_KEYS
        and payload.get("decision") in _REVIEW_DECISIONS
        and payload.get("confidence") in _CONFIDENCE_VALUES
        and isinstance(payload.get("reason"), str)
        and isinstance(payload.get("evidence_refs_considered"), list)
        and isinstance(payload.get("update_instructions"), list)
    )


def _valid_curation_payload(payload: dict[str, Any]) -> bool:
    return (
        set(payload) == _CURATION_KEYS
        and isinstance(payload.get("description"), str)
        and bool(payload["description"].strip())
        and payload.get("confidence") in _CONFIDENCE_VALUES
        and isinstance(payload.get("used_evidence_refs"), list)
        and isinstance(payload.get("limitations"), list)
    )


def build_profile_evidence_match_summary(entity: dict[str, Any]) -> dict[str, Any]:
    existing_items = []
    current_description_normalized = _normalized_evidence_text(entity.get("current_description"))
    for evidence in entity.get("considered_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        normalized = _normalized_evidence_text(evidence.get("text"))
        if not normalized:
            continue
        existing_items.append(
            {
                "evidence": evidence,
                "normalized_text": normalized,
                "ngrams": _char_ngrams(normalized),
            }
        )

    exact_candidates = [item["normalized_text"] for item in existing_items]
    if current_description_normalized:
        exact_candidates.append(current_description_normalized)
    filtered_new_evidence: list[dict[str, Any]] = []
    duplicate_evidence_ids: list[str] = []
    duplicate_article_ids: list[str] = []
    match_records: list[dict[str, Any]] = []

    for evidence in entity.get("new_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        normalized = _normalized_evidence_text(evidence.get("text"))
        if not normalized:
            filtered_new_evidence.append(evidence)
            continue

        new_ngrams = _char_ngrams(normalized)
        exact_duplicate = any(
            _same_evidence_text(normalized, candidate) for candidate in exact_candidates
        )
        for existing_item in existing_items:
            existing_evidence = existing_item["evidence"]
            exact_match = _same_evidence_text(
                normalized,
                existing_item["normalized_text"],
            )
            score = 1.0 if exact_match else _ngram_jaccard(new_ngrams, existing_item["ngrams"])
            match_records.append(
                {
                    "new_evidence_id": evidence.get("id"),
                    "new_article_id": evidence.get("article_id"),
                    "new_text": _text_preview(evidence.get("text")),
                    "matching_evidence_id": existing_evidence.get("id"),
                    "matching_article_id": existing_evidence.get("article_id"),
                    "matching_text": _text_preview(existing_evidence.get("text")),
                    "score": round(score, 4),
                    "exact_match": exact_match,
                }
            )

        if exact_duplicate:
            evidence_id = str(evidence.get("id") or "").strip()
            article_id = str(evidence.get("article_id") or "").strip()
            if evidence_id:
                duplicate_evidence_ids.append(evidence_id)
            if article_id:
                duplicate_article_ids.append(article_id)
        else:
            filtered_new_evidence.append(evidence)

    top_matches = sorted(
        match_records,
        key=lambda item: (item["score"], bool(item["exact_match"])),
        reverse=True,
    )[:_MAX_PROFILE_MATCHES]
    return {
        "new_evidence": filtered_new_evidence,
        "existing_evidence_count": len(existing_items),
        "exact_duplicate_evidence_ids": _merge_unique(duplicate_evidence_ids),
        "exact_duplicate_article_ids": _merge_unique(duplicate_article_ids),
        "exact_duplicate_count": len(_merge_unique(duplicate_evidence_ids)),
        "best_similarity_score": top_matches[0]["score"] if top_matches else None,
        "top_matches": top_matches,
    }


def build_profile_prompt_context(
    entity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, str], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for evidence in entity.get("new_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        article_id = str(evidence.get("article_id") or "").strip()
        text = str(evidence.get("text") or "").strip()
        evidence_id = str(evidence.get("id") or "").strip()
        if not article_id or not text or not evidence_id:
            continue
        grouped.setdefault(article_id, []).append(evidence)

    groups = sorted(grouped.values(), key=_article_group_sort_key, reverse=True)[
        :_MAX_PROFILE_EVIDENCE_ARTICLES
    ]
    prompt_evidence: list[dict[str, str]] = []
    evidence_sources: list[dict[str, str]] = []
    evidence_ref_map: dict[str, list[str]] = {}
    article_ref_map: dict[str, str] = {}
    for index, group in enumerate(groups, start=1):
        selected = max(group, key=_evidence_revision_sort_key)
        ref = f"E{index}"
        article_url = str(selected.get("article_url") or "").strip()
        article_id = str(selected["article_id"])
        prompt_evidence.append(
            {
                "ref": ref,
                "text": str(selected["text"]).strip()[:_MAX_PROFILE_EVIDENCE_CHARS],
            }
        )
        evidence_sources.append(
            {
                "ref": ref,
                "article_id": article_id,
                "article_url": article_url,
            }
        )
        evidence_ref_map[ref] = [
            str(item["id"]) for item in group if isinstance(item, dict) and item.get("id")
        ]
        article_ref_map[ref] = article_id

    context = {
        "entity": {
            "type": entity.get("label"),
            "name": entity.get("name") or entity.get("canonical_name"),
            "aliases": entity.get("aliases") or [],
        },
        "current_description": str(entity.get("current_description") or "").strip(),
        "new_evidence": prompt_evidence,
    }
    return context, evidence_ref_map, article_ref_map, evidence_sources


def _article_group_sort_key(group: list[dict[str, Any]]) -> str:
    return max(_evidence_article_sort_key(evidence) for evidence in group)


def _evidence_article_sort_key(evidence: dict[str, Any]) -> str:
    return str(
        evidence.get("published_at")
        or evidence.get("last_seen_at")
        or evidence.get("created_at")
        or ""
    )


def _evidence_revision_sort_key(evidence: dict[str, Any]) -> str:
    return str(
        evidence.get("last_seen_at")
        or evidence.get("created_at")
        or evidence.get("published_at")
        or evidence.get("id")
        or ""
    )


def _known_refs(
    candidates: Any,
    allowed_refs: list[str],
    *,
    fallback: list[str],
) -> list[str]:
    allowed = set(allowed_refs)
    selected = [str(ref) for ref in candidates or [] if str(ref) in allowed]
    return selected or list(fallback)


def _evidence_ids_for_refs(
    refs: Any,
    evidence_ref_map: dict[str, list[str]],
) -> list[str]:
    evidence_ids: list[str] = []
    seen: set[str] = set()
    for ref in refs or []:
        for evidence_id in evidence_ref_map.get(str(ref), []):
            if evidence_id not in seen:
                seen.add(evidence_id)
                evidence_ids.append(evidence_id)
    return evidence_ids


def _article_ids_for_refs(
    refs: Any,
    article_ref_map: dict[str, str],
) -> list[str]:
    article_ids: list[str] = []
    seen: set[str] = set()
    for ref in refs or []:
        article_id = article_ref_map.get(str(ref))
        if article_id and article_id not in seen:
            seen.add(article_id)
            article_ids.append(article_id)
    return article_ids


def _merge_unique(*groups: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            value = str(item or "").strip()
            if value and value not in seen:
                seen.add(value)
                values.append(value)
    return values


def _normalized_evidence_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def _same_evidence_text(left: str, right: str) -> bool:
    """Compare evidence text conservatively for duplicate detection.

    Texts match only when they are identical, identical after removing
    accents/marks, or differ by a tiny non-ASCII transliteration in the same
    token positions. Added/removed words, changed numbers, and plain ASCII word
    changes are treated as different evidence (conservative).
    """
    if left == right:
        return True

    left_tokens = _evidence_text_tokens(left, strip_marks=True)
    right_tokens = _evidence_text_tokens(right, strip_marks=True)
    if left_tokens == right_tokens:
        return True
    if len(left_tokens) != len(right_tokens):
        return False

    left_original_tokens = _evidence_text_tokens(left, strip_marks=False)
    right_original_tokens = _evidence_text_tokens(right, strip_marks=False)
    total_distance = 0
    for left_token, right_token, left_original, right_original in zip(
        left_tokens,
        right_tokens,
        left_original_tokens,
        right_original_tokens,
        strict=True,
    ):
        if left_token == right_token:
            continue
        if _contains_decimal_digit(left_token) or _contains_decimal_digit(right_token):
            return False
        if left_original.isascii() and right_original.isascii():
            return False
        distance = Levenshtein.distance(left_token, right_token)
        if distance > _MAX_TRANSLITERATION_EDIT_DISTANCE:
            return False
        total_distance += distance
        if total_distance > _MAX_TRANSLITERATION_EDIT_DISTANCE:
            return False

    return True


def _contains_decimal_digit(value: str) -> bool:
    return any(char.isdecimal() for char in value)


def _evidence_text_tokens(value: str, *, strip_marks: bool) -> list[str]:
    """Return comparable word tokens, optionally removing accents/marks first."""
    form = "NFKD" if strip_marks else "NFKC"
    text = unicodedata.normalize(form, value.casefold())
    chars = []
    for char in text:
        if strip_marks and unicodedata.category(char).startswith("M"):
            continue
        chars.append(char if char.isalnum() else " ")
    return "".join(chars).split()


def _char_ngrams(text: str, size: int = 3) -> set[str]:
    if not text:
        return set()
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _ngram_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _text_preview(value: Any) -> str:
    return _truncate_for_log(str(value or ""), _MATCH_TEXT_PREVIEW_CHARS)


def _direct_update_review(evidence_refs: list[str]) -> dict[str, Any]:
    return {
        "decision": "update_profile",
        "confidence": "high",
        "reason": "The entity has no current public profile.",
        "evidence_refs_considered": evidence_refs,
        "update_instructions": ["Create a stable public profile from the provided evidence."],
    }


def _profile_curation_policy_hash(
    *,
    review_prompt: Any,
    review_uri: str,
    curation_prompt: Any,
    curation_uri: str,
) -> str:
    payload = {
        "profile_review": _prompt_policy_payload(
            review_prompt,
            review_uri,
            build_entity_profile_review_prompt_registry_template(),
        ),
        "profile_curation": _prompt_policy_payload(
            curation_prompt,
            curation_uri,
            build_entity_curation_prompt_registry_template(),
        ),
    }
    return _hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
