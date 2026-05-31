"""Per-article tracing orchestrator.

Pipeline shape
==============
1. `IngestionService.ingest` opens **one MLflow Run** via
   `IngestionRun(...)` for the batch.
2. Scrape and resolver-registry load run as plain stages — no traces.
3. Each article then flows through `_process_article`, decorated with
   `@mlflow.trace`. That is the trace **root** for the article, and the
   tree below it captures the whole pipeline (extract → filter → resolve
   → write).
4. Per-article OpenAI calls auto-instrument as `CHAT_MODEL` spans
   thanks to `mlflow.openai.autolog()`.

Concurrency
-----------
LLM concurrency is bounded by the semaphore inside
`LLMExtractionService`. Resolution + Neo4j writes mutate the shared
`EntityResolver` state and the shared graph, so we serialize them
behind a per-job `_resolver_lock`. Extraction still runs in parallel.
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any
from urllib.parse import quote

import mlflow
from mlflow.entities import SpanType

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.observability import timed_step
from app.models.extraction import (
    ADMITTED_EVIDENCE_STATUSES,
    ArticleIn,
    EntityType,
    EvidenceStatus,
    ExtractedEntity,
    ExtractionResult,
    IngestRequest,
    IngestStats,
    NormalizedEntity,
    strongest_evidence_status,
)
from app.observability import IngestionRun, RunHandle
from app.observability.context import current_trace_id as _current_trace_id
from app.observability.artifacts import (
    build_dedup_report,
    build_ingestion_summary_md,
)
from app.observability.scorers import attach_extraction_scores
from app.graph.graph_store import GraphStore
from app.services.embedding import EmbeddingService
from app.services.entity_resolution import EntityResolver, NameNormalizer, ResolutionOutcome
from app.services.llm import LLMExtractionService, MISSING_OPENAI_API_KEY_MESSAGE
from app.services.progress import article_fields
from app.services.scraper import ArticleScraper


_MAX_EVIDENCE_CHARS = 320


class _ArticleResult:
    """Per-article aggregate returned by `_process_article`."""

    __slots__ = (
        "success",
        "article",
        "extract_row",
        "graph_row",
        "outcome_rows",
        "extraction_dump",
        "failure_row",
        "input_tokens",
        "output_tokens",
        "duration_ms",
    )

    def __init__(self) -> None:
        self.success: bool = False
        self.article: ArticleIn | None = None
        self.extract_row: dict[str, Any] = {}
        self.graph_row: dict[str, Any] = {}
        self.outcome_rows: list[dict[str, Any]] = []
        self.extraction_dump: dict[str, Any] | None = None
        self.failure_row: dict[str, Any] | None = None
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.duration_ms: float = 0.0


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        scraper: ArticleScraper,
        llm: LLMExtractionService,
        graph: GraphStore,
        embedding: EmbeddingService | None = None,
    ):
        self.settings = settings
        self.scraper = scraper
        self.llm = llm
        self.graph = graph
        self.embedding = embedding
        self.logger = get_logger("ingestion")
        # Resolver state and Neo4j writes are not thread-safe; serialize the
        # post-extraction phase per ingestion. Extraction itself still runs
        # in parallel inside the LLMExtractionService semaphore.
        self._resolver_lock = asyncio.Lock()

    async def ingest(
        self,
        request: IngestRequest,
        *,
        ingest_run_id: str | None = None,
    ) -> IngestStats:
        if not self.settings.openai_api_key:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)

        started = perf_counter()
        stats = IngestStats(source_name=request.source_name)
        run_id = ingest_run_id or "no-task-id"

        run_params = self._build_run_params(request)
        run_tags = self._build_run_tags(request)

        async with timed_step(
            self.logger,
            "ingestion",
            workflow_step="ingest",
            url=str(request.source_url),
            count=request.max_pages,
            detail=(
                f"source={request.source_name}; include_feed={request.include_feed}; "
                f"paths={len(request.paths)}; ingest_run_id={run_id}"
            ),
        ):
            async with IngestionRun(
                job_run_id=run_id,
                source_name=request.source_name,
                params=run_params,
                tags=run_tags,
            ) as ingest_run:
                articles = await self._scrape_stage(request, stats, ingest_run)
                resolver = await self._registry_stage(ingest_run)
                results = await self._dispatch_articles(
                    articles=articles,
                    resolver=resolver,
                    job_run_id=run_id,
                )
                self._tally(results, stats)
                self._finalize_run(
                    ingest_run=ingest_run,
                    stats=stats,
                    results=results,
                    duration_ms=round((perf_counter() - started) * 1000, 2),
                )

        stats.duration_ms = round((perf_counter() - started) * 1000, 2)
        self.logger.info(
            "ingest_completed",
            extra={
                "event": "ingestion",
                "workflow_step": "ingest",
                "count": stats.articles_processed,
                "failed_count": stats.articles_failed,
                "duration_ms": stats.duration_ms,
                "detail": (
                    f"found={stats.articles_found}, processed={stats.articles_processed}, "
                    f"failed={stats.articles_failed}, entities={stats.entities_extracted}, "
                    f"relationships={stats.relationships_created}"
                ),
            },
        )
        return stats

    # ------------------------------------------------------------------
    # Run-level helpers
    # ------------------------------------------------------------------

    def _build_run_params(self, request: IngestRequest) -> dict[str, Any]:
        return {
            "max_pages": request.max_pages,
            "source_name": request.source_name,
            "source_url": str(request.source_url),
            "include_feed": request.include_feed,
            "paths_count": len(request.paths),
            "openai_model": self.settings.openai_model,
            "admission_policy": "evidence_status:stated|attributed",
            "embedding_model": self.settings.embedding_model,
            "embedding_similarity_threshold": self.settings.embedding_similarity_threshold,
            "enable_embedding_resolution": self.settings.enable_embedding_resolution,
            "llm_retry_attempts": self.settings.llm_retry_attempts,
            "llm_max_concurrency": self.settings.llm_max_concurrency,
            "llm_gleaning_passes": self.settings.llm_gleaning_passes,
            "prompt_extraction_uri": self.settings.mlflow_prompt_extraction_uri,
            "prompt_gleaning_uri": self.settings.mlflow_prompt_gleaning_uri,
            "use_prompt_registry": self.settings.mlflow_use_prompt_registry,
            "app_env": self.settings.app_env,
            "git_sha": self.settings.git_sha,
            "app_version": self.settings.app_version,
        }

    def _build_run_tags(self, request: IngestRequest) -> dict[str, str]:
        return {
            "env": self.settings.app_env,
            "git_sha": self.settings.git_sha,
            "app_version": self.settings.app_version,
            "source_name": request.source_name,
            "model": self.settings.openai_model,
        }

    async def _scrape_stage(
        self,
        request: IngestRequest,
        stats: IngestStats,
        ingest_run: RunHandle,
    ) -> list[ArticleIn]:
        stage_started = perf_counter()
        self.logger.info(
            "workflow_stage_started",
            extra={
                "event": "ingestion",
                "workflow_step": "scraping",
                "url": str(request.source_url),
                "detail": "collecting article links and article text",
            },
        )
        articles = await self.scraper.collect(
            source_url=str(request.source_url),
            source_name=request.source_name,
            max_pages=request.max_pages,
            include_feed=request.include_feed,
            paths=request.paths,
        )
        stats.articles_found = len(articles)
        duration_ms = round((perf_counter() - stage_started) * 1000, 2)
        ingest_run.record_metric("scrape_duration_ms", duration_ms)
        ingest_run.record_metric("articles_found", float(stats.articles_found))
        self.logger.info(
            "workflow_stage_completed",
            extra={
                "event": "ingestion",
                "workflow_step": "scraping",
                "url": str(request.source_url),
                "count": stats.articles_found,
                "detail": f"articles_ready={stats.articles_found}",
            },
        )
        return articles

    async def _registry_stage(self, ingest_run: RunHandle) -> EntityResolver:
        stage_started = perf_counter()
        self.logger.info(
            "workflow_stage_started",
            extra={
                "event": "ingestion",
                "workflow_step": "resolution_registry",
                "detail": "loading existing graph entities and aliases",
            },
        )
        resolver = EntityResolver(self.settings, self.embedding, self.graph)
        await resolver.load_from_graph(self.graph)
        duration_ms = round((perf_counter() - stage_started) * 1000, 2)
        registry_size = self._registry_size(resolver)
        ingest_run.record_metric("registry_load_duration_ms", duration_ms)
        ingest_run.record_metric("registry_size", float(registry_size))
        self.logger.info(
            "workflow_stage_completed",
            extra={
                "event": "ingestion",
                "workflow_step": "resolution_registry",
                "detail": f"entity registry ready (size={registry_size})",
            },
        )
        return resolver

    async def _dispatch_articles(
        self,
        *,
        articles: list[ArticleIn],
        resolver: EntityResolver,
        job_run_id: str,
    ) -> list[_ArticleResult]:
        if not articles:
            return []
        total = len(articles)
        tasks = [
            asyncio.create_task(
                self._process_article(
                    article=article,
                    resolver=resolver,
                    job_run_id=job_run_id,
                    article_index=index,
                    article_total=total,
                )
            )
            for index, article in enumerate(articles, start=1)
        ]
        return await asyncio.gather(*tasks)

    def _tally(self, results: list[_ArticleResult], stats: IngestStats) -> None:
        for result in results:
            if result.success:
                stats.articles_processed += 1
                stats.entities_extracted += int(result.graph_row.get("entity_count", 0) or 0)
                stats.relationships_created += int(result.graph_row.get("graph_operations", 0) or 0)
            else:
                stats.articles_failed += 1

    def _finalize_run(
        self,
        *,
        ingest_run: RunHandle,
        stats: IngestStats,
        results: list[_ArticleResult],
        duration_ms: float,
    ) -> None:
        latencies = [r.duration_ms for r in results if r.success and r.duration_ms]
        method_totals: dict[str, int] = {"exact": 0, "fuzzy": 0, "embedding": 0, "new": 0}
        outcome_rows: list[dict[str, Any]] = []
        extraction_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        graph_rows: list[dict[str, Any]] = []
        extraction_dumps: list[dict[str, Any]] = []

        for result in results:
            if result.success:
                for row in result.outcome_rows:
                    method_totals[row.get("method", "new")] = (
                        method_totals.get(row.get("method", "new"), 0) + 1
                    )
                    outcome_rows.append(row)
                extraction_rows.append(result.extract_row)
                graph_rows.append(result.graph_row)
                if result.extraction_dump:
                    extraction_dumps.append(result.extraction_dump)
                ingest_run.add_token_usage(
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
            elif result.failure_row:
                failure_rows.append(result.failure_row)

        attempts_list = [
            int(row.get("attempts") or 0) for row in extraction_rows if row.get("attempts")
        ]
        metrics: dict[str, float] = {
            "articles_processed": float(stats.articles_processed),
            "articles_failed": float(stats.articles_failed),
            "entities_extracted": float(stats.entities_extracted),
            "relationships_created": float(stats.relationships_created),
            "duration_ms": duration_ms,
            "resolved_exact": float(method_totals["exact"]),
            "resolved_fuzzy": float(method_totals["fuzzy"]),
            "resolved_embedding": float(method_totals["embedding"]),
            "resolved_new": float(method_totals["new"]),
        }
        for key in (
            "entity_stated",
            "entity_attributed",
            "entity_unsure",
            "entity_status_defaulted",
            "relationship_stated",
            "relationship_attributed",
            "relationship_unsure",
            "relationship_status_defaulted",
        ):
            metrics[key] = float(sum(int(row.get(key) or 0) for row in extraction_rows))
        entity_candidates = (
            metrics["entity_stated"] + metrics["entity_attributed"] + metrics["entity_unsure"]
        )
        relationship_candidates = (
            metrics["relationship_stated"]
            + metrics["relationship_attributed"]
            + metrics["relationship_unsure"]
        )
        if entity_candidates:
            metrics["entity_admitted_ratio"] = round(
                (metrics["entity_stated"] + metrics["entity_attributed"]) / entity_candidates,
                4,
            )
        if relationship_candidates:
            metrics["relationship_admitted_ratio"] = round(
                (metrics["relationship_stated"] + metrics["relationship_attributed"])
                / relationship_candidates,
                4,
            )
        if latencies:
            metrics["p50_article_latency_ms"] = _percentile(latencies, 50)
            metrics["p95_article_latency_ms"] = _percentile(latencies, 95)
        if attempts_list:
            metrics["avg_extraction_attempts"] = sum(attempts_list) / len(attempts_list)
        total_outcomes = sum(method_totals.values())
        if total_outcomes:
            metrics["dedup_rate"] = round(
                (method_totals["exact"] + method_totals["fuzzy"] + method_totals["embedding"])
                / total_outcomes,
                4,
            )
        ingest_run.record_metrics(metrics)

        if extraction_rows:
            ingest_run.add_jsonl_artifact("extraction_summary.jsonl", extraction_rows)
        if graph_rows:
            ingest_run.add_jsonl_artifact("graph_ops.jsonl", graph_rows)
        if extraction_dumps:
            ingest_run.add_jsonl_artifact("extraction_dump.jsonl", extraction_dumps)
        if failure_rows:
            ingest_run.add_jsonl_artifact("failed_articles.jsonl", failure_rows)
        if outcome_rows:
            ingest_run.add_json_artifact(
                "dedup_report.json",
                build_dedup_report(outcome_rows=outcome_rows),
            )

        ingest_run.add_artifact(
            "ingestion_summary.md",
            build_ingestion_summary_md(
                job_run_id=ingest_run.job_run_id,
                source_name=ingest_run.metrics.get("source_name", "") or self.settings.app_env,
                metrics=metrics,
                failures_sample=failure_rows,
            ),
        )

    # ------------------------------------------------------------------
    # Per-article trace root
    # ------------------------------------------------------------------

    @mlflow.trace(name="process_article", span_type=SpanType.CHAIN)
    async def _process_article(
        self,
        *,
        article: ArticleIn,
        resolver: EntityResolver,
        job_run_id: str,
        article_index: int,
        article_total: int,
    ) -> _ArticleResult:
        result = _ArticleResult()
        result.article = article
        stage_started = perf_counter()

        mlflow.update_current_trace(
            tags={
                "job_run_id": job_run_id,
                "article_url": article.url,
                "source_name": article.source_name,
                "model": self.settings.openai_model,
                "env": self.settings.app_env,
                "git_sha": self.settings.git_sha,
            },
            request_preview=f"{article.source_name} | {(article.title or article.url)[:120]}",
        )

        span = mlflow.get_current_active_span()
        if span is not None:
            try:
                span.set_inputs(
                    {
                        "article_url": article.url,
                        "article_title": article.title,
                        "source_name": article.source_name,
                        "published_at": (
                            article.published_at.isoformat() if article.published_at else None
                        ),
                        "tags": list(article.tags),
                    }
                )
                span.set_attributes(
                    {
                        "article_url": article.url,
                        "source_name": article.source_name,
                        "text_chars": len(article.text),
                        "tag_count": len(article.tags),
                    }
                )
            except Exception:  # pragma: no cover
                pass

        try:
            extraction, extraction_metadata = await self.llm.extract_article(
                article,
                article_index=article_index,
                article_total=article_total,
            )
        except Exception as exc:
            duration_ms = round((perf_counter() - stage_started) * 1000, 2)
            result.duration_ms = duration_ms
            result.failure_row = {
                "article_url": article.url,
                "title": (article.title or "")[:200],
                "source_name": article.source_name,
                "stage": "extraction",
                "error": str(exc)[:300],
                "trace_id": _current_trace_id(),
                "attempts": self.settings.llm_retry_attempts,
            }
            self.logger.error(
                "article_extraction_failed",
                extra={
                    "event": "extraction",
                    "workflow_step": "article_extraction",
                    "error": str(exc),
                    **article_fields(
                        article, article_index=article_index, article_total=article_total
                    ),
                },
            )
            if span is not None:
                try:
                    span.set_outputs({"status": "failed", "stage": "extraction"})
                    span.set_attribute("error", str(exc)[:200])
                except Exception:  # pragma: no cover
                    pass
            mlflow.update_current_trace(response_preview=f"FAILED at extraction: {str(exc)[:80]}")
            return result

        raw_snapshot = _raw_extracted_entities_snapshot(extraction)
        raw_status_counts = _evidence_status_counts(extraction)
        pre_filter_count = extraction.entity_count()
        cleaned = self._filter_by_evidence_traced(extraction)

        if span is not None:
            try:
                span.set_attribute("entity_count_pre_filter", pre_filter_count)
                span.set_attribute("entity_count_post_filter", cleaned.entity_count())
            except Exception:  # pragma: no cover
                pass

        try:
            async with self._resolver_lock:
                resolved, outcomes = await self._resolve_traced(resolver, article, cleaned)
                graph_ops = await self._write_traced(
                    article=article,
                    extraction=cleaned,
                    resolved=resolved,
                    raw_extracted=raw_snapshot,
                    job_run_id=job_run_id,
                )
        except Exception as exc:
            duration_ms = round((perf_counter() - stage_started) * 1000, 2)
            result.duration_ms = duration_ms
            result.failure_row = {
                "article_url": article.url,
                "title": (article.title or "")[:200],
                "source_name": article.source_name,
                "stage": "resolve_or_write",
                "error": str(exc)[:300],
                "trace_id": _current_trace_id(),
            }
            self.logger.exception(
                "article_resolve_or_write_failed",
                extra={
                    "event": "ingestion",
                    "workflow_step": "article_process",
                    "error": str(exc),
                    **article_fields(
                        article, article_index=article_index, article_total=article_total
                    ),
                },
            )
            if span is not None:
                try:
                    span.set_outputs({"status": "failed", "stage": "resolve_or_write"})
                    span.set_attribute("error", str(exc)[:200])
                except Exception:  # pragma: no cover
                    pass
            mlflow.update_current_trace(
                response_preview=f"FAILED at resolve/write: {str(exc)[:80]}"
            )
            return result

        duration_ms = round((perf_counter() - stage_started) * 1000, 2)
        method_counts = {m: 0 for m in ("exact", "fuzzy", "embedding", "new")}
        for outcome in outcomes:
            method_counts[outcome.method] = method_counts.get(outcome.method, 0) + 1

        result.success = True
        result.duration_ms = duration_ms
        result.input_tokens = int(extraction_metadata.get("input_tokens") or 0)
        result.output_tokens = int(extraction_metadata.get("output_tokens") or 0)
        result.extract_row = {
            "article_url": article.url,
            "title": (article.title or "")[:200],
            "source_name": article.source_name,
            "status": "succeeded",
            "entity_count": cleaned.entity_count(),
            "relationship_count": len(cleaned.relationships),
            **raw_status_counts,
            "attempts": extraction_metadata.get("attempts"),
            "latency_ms": extraction_metadata.get("latency_ms"),
            "trace_id": extraction_metadata.get("trace_id") or _current_trace_id(),
            "prompt_name": extraction_metadata.get("prompt_name"),
            "prompt_version": extraction_metadata.get("prompt_version"),
            "prompt_uri": extraction_metadata.get("prompt_uri"),
        }
        result.graph_row = {
            "article_url": article.url,
            "title": (article.title or "")[:200],
            "source_name": article.source_name,
            "status": "succeeded",
            "entity_count": cleaned.entity_count(),
            "relationship_count": len(cleaned.relationships),
            **raw_status_counts,
            "resolved_exact": method_counts["exact"],
            "resolved_fuzzy": method_counts["fuzzy"],
            "resolved_embedding": method_counts["embedding"],
            "resolved_new": method_counts["new"],
            "graph_operations": graph_ops,
            "trace_id": _current_trace_id(),
        }
        result.outcome_rows = [
            {
                "article_url": article.url,
                "entity_type": outcome.entity.label,
                **outcome.to_dict(),
            }
            for outcome in outcomes
        ]
        result.extraction_dump = {
            "article_url": article.url,
            "title": article.title,
            "source_name": article.source_name,
            "trace_id": _current_trace_id(),
            "extraction": cleaned.model_dump(mode="json"),
            "raw_extraction": raw_snapshot,
            "evidence_status_counts": raw_status_counts,
            "outcomes": [outcome.to_dict() for outcome in outcomes],
            "graph_operations": graph_ops,
        }

        if span is not None:
            try:
                span.set_outputs(
                    {
                        "status": "succeeded",
                        "entities_extracted": cleaned.entity_count(),
                        "entities_resolved": len(outcomes),
                        "graph_operations": graph_ops,
                        "resolution_methods": method_counts,
                        "duration_ms": duration_ms,
                    }
                )
            except Exception:  # pragma: no cover
                pass

        mlflow.update_current_trace(
            response_preview=(
                f"{cleaned.entity_count()} entities "
                f"(S={len(cleaned.startups)} I={len(cleaned.investors)} "
                f"P={len(cleaned.people)} T={len(cleaned.topics)} "
                f"C={len(cleaned.companies)}) | "
                f"{len(cleaned.relationships)} rels | "
                f"{graph_ops} graph ops | "
                f"{duration_ms:.0f}ms"
            )
        )

        attach_extraction_scores(
            trace_id=_current_trace_id() or "",
            extraction=cleaned,
            duration_ms=duration_ms,
            raw_status_counts=raw_status_counts,
        )

        self.logger.info(
            "article_processed",
            extra={
                "event": "ingestion",
                "workflow_step": "article_process",
                "url": article.url,
                "count": cleaned.entity_count(),
                "detail": (
                    f"entities={cleaned.entity_count()}, graph_ops={graph_ops}, "
                    f"duration_ms={duration_ms:.1f}, attempts={extraction_metadata.get('attempts')}"
                ),
                **article_fields(article, article_index=article_index, article_total=article_total),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Inner traced helpers
    # ------------------------------------------------------------------

    @mlflow.trace(name="evidence_gate", span_type=SpanType.CHAIN)
    def _filter_by_evidence_traced(self, extraction: ExtractionResult) -> ExtractionResult:
        status_counts = _evidence_status_counts(extraction)
        pre = extraction.entity_count()
        pre_rel = len(extraction.relationships)
        extraction.startups = [
            e for e in extraction.startups if e.evidence_status in ADMITTED_EVIDENCE_STATUSES
        ]
        extraction.investors = [
            e for e in extraction.investors if e.evidence_status in ADMITTED_EVIDENCE_STATUSES
        ]
        extraction.people = [
            e for e in extraction.people if e.evidence_status in ADMITTED_EVIDENCE_STATUSES
        ]
        extraction.topics = [
            e for e in extraction.topics if e.evidence_status in ADMITTED_EVIDENCE_STATUSES
        ]
        extraction.companies = [
            e for e in extraction.companies if e.evidence_status in ADMITTED_EVIDENCE_STATUSES
        ]
        extraction.relationships = [
            r for r in extraction.relationships if r.evidence_status in ADMITTED_EVIDENCE_STATUSES
        ]
        post = extraction.entity_count()
        post_rel = len(extraction.relationships)
        span = mlflow.get_current_active_span()
        if span is not None:
            try:
                span.set_attributes(
                    {
                        "admission_policy": "stated|attributed",
                        "entities_pre_filter": pre,
                        "entities_post_filter": post,
                        "entities_dropped": max(pre - post, 0),
                        "relationships_pre_filter": pre_rel,
                        "relationships_post_filter": post_rel,
                        "relationships_dropped": max(pre_rel - post_rel, 0),
                        **status_counts,
                    }
                )
                span.set_outputs(
                    {
                        "entities_kept": post,
                        "entities_dropped": max(pre - post, 0),
                        "relationships_kept": post_rel,
                        "relationships_dropped": max(pre_rel - post_rel, 0),
                    }
                )
            except Exception:  # pragma: no cover
                pass
        return extraction

    async def _resolve_traced(
        self,
        resolver: EntityResolver,
        article: ArticleIn,  # noqa: ARG002 — kept for symmetry with the call site
        extraction: ExtractionResult,
    ) -> tuple[dict[tuple[str, str], NormalizedEntity], list[ResolutionOutcome]]:
        # Manual span — the return value contains a dict with tuple keys
        # which the @mlflow.trace auto-capture cannot JSON-serialize.
        with mlflow.start_span(name="resolve_entities", span_type=SpanType.CHAIN) as span:
            return await self._do_resolve_entities(span, resolver, extraction)

    async def _do_resolve_entities(
        self,
        span,
        resolver: EntityResolver,
        extraction: ExtractionResult,
    ) -> tuple[dict[tuple[str, str], NormalizedEntity], list[ResolutionOutcome]]:
        self._dedupe_topics(extraction)
        self._ensure_relationship_entities(extraction)
        relationship_neighbors = _relationship_neighbor_keys(extraction)

        _entity_groups: list[tuple[str, list[ExtractedEntity]]] = [
            ("Startup", extraction.startups),
            ("Investor", extraction.investors),
            ("Person", extraction.people),
            ("Topic", extraction.topics),
            ("Company", extraction.companies),
        ]

        # Pre-compute all embeddings in one batched call before the resolution loop.
        # Resolution must remain sequential (each resolved entity is immediately added
        # to the in-memory registry), but embedding I/O can be front-loaded.
        precomputed: dict[tuple[str, str], list[float]] = {}
        if self.embedding and self.settings.enable_embedding_resolution:
            all_pairs = [(et, e) for et, entities in _entity_groups for e in entities]
            texts = [
                f"{NameNormalizer.display(e.name)}. {e.description or ''}".strip()
                for _, e in all_pairs
            ]
            try:
                vecs = await self.embedding.embed(texts)
                precomputed = {(et, e.name): vec for (et, e), vec in zip(all_pairs, vecs)}
            except Exception as exc:
                self.logger.warning(
                    "embedding_prefetch_failed",
                    extra={
                        "event": "resolution",
                        "workflow_step": "embedding",
                        "error": str(exc),
                    },
                )

        mapping: dict[tuple[str, str], NormalizedEntity] = {}
        outcomes: list[ResolutionOutcome] = []
        # Track (outcome, entity_type, raw_name) to attach pre-fetched embeddings after the loop
        _outcome_keys: list[tuple[ResolutionOutcome, str, str]] = []
        for entity_type, entities in _entity_groups:
            for entity in entities:
                entity_key = _entity_key(entity_type, entity.name)
                neighbor_keys = (
                    relationship_neighbors.get(entity_key, set()) if entity_key else set()
                )
                blocked_entity_ids = {
                    resolved.id
                    for neighbor_key in neighbor_keys
                    if (resolved := mapping.get(neighbor_key)) is not None
                }
                blocked_canonical_keys = {
                    key
                    for neighbor_type, key in neighbor_keys
                    if neighbor_type == entity_type and key
                }
                outcome = await self._resolve_one_traced(
                    resolver,
                    entity_type=entity_type,
                    entity=entity,
                    precomputed_embedding=precomputed.get((entity_type, entity.name)),
                    blocked_entity_ids=blocked_entity_ids,
                    blocked_canonical_keys=blocked_canonical_keys,
                )
                outcomes.append(outcome)
                _outcome_keys.append((outcome, entity_type, entity.name))
                self._add_mapping(mapping, entity_type, entity, outcome)

        # Attach pre-computed embeddings to new entities so they're stored in Neo4j
        # and available for vector search on the next ingestion run.
        for outcome, entity_type, raw_name in _outcome_keys:
            if outcome.method == "new":
                vec = precomputed.get((entity_type, raw_name))
                if vec is not None:
                    outcome.entity.embedding = vec

        method_counts = {m: 0 for m in ("exact", "fuzzy", "embedding", "new")}
        for outcome in outcomes:
            method_counts[outcome.method] += 1

        span = mlflow.get_current_active_span()
        if span is not None:
            try:
                span.set_inputs(
                    {
                        "entity_counts": {
                            "Startup": len(extraction.startups),
                            "Investor": len(extraction.investors),
                            "Person": len(extraction.people),
                            "Topic": len(extraction.topics),
                            "Company": len(extraction.companies),
                        }
                    }
                )
                span.set_outputs(
                    {
                        "outcomes": method_counts,
                        "outcomes_total": len(outcomes),
                    }
                )
                span.set_attributes(method_counts)
            except Exception:  # pragma: no cover
                pass
        return mapping, outcomes

    async def _resolve_one_traced(
        self,
        resolver: EntityResolver,
        *,
        entity_type: EntityType,
        entity: ExtractedEntity,
        precomputed_embedding: list[float] | None = None,
        blocked_entity_ids: set[str] | None = None,
        blocked_canonical_keys: set[str] | None = None,
    ) -> ResolutionOutcome:
        with mlflow.start_span(name="resolve_entity", span_type=SpanType.CHAIN) as span:
            try:
                span.set_attributes(
                    {
                        "entity_type": entity_type,
                        "candidate_name": entity.name,
                        "candidate_evidence_status": entity.evidence_status,
                        "relationship_blocked_ids": len(blocked_entity_ids or set()),
                        "relationship_blocked_names": len(blocked_canonical_keys or set()),
                    }
                )
                span.set_inputs(
                    {
                        "name": entity.name,
                        "aliases": list(entity.aliases),
                        "description": entity.description,
                    }
                )
            except Exception:  # pragma: no cover
                pass

            evidence = self._resolve_candidate_evidence(entity)
            outcome = await resolver.resolve(
                entity_type,
                entity,
                candidate_evidence=evidence,
                precomputed_embedding=precomputed_embedding,
                blocked_entity_ids=blocked_entity_ids,
                blocked_canonical_keys=blocked_canonical_keys,
            )

            try:
                span.set_outputs(
                    {
                        "method": outcome.method,
                        "canonical": outcome.entity.canonical_name,
                        "entity_id": outcome.entity.id,
                        "similarity_min": outcome.similarity_min,
                    }
                )
                span.set_attribute("resolution_method", outcome.method)
            except Exception:  # pragma: no cover
                pass
            return outcome

    async def _write_traced(
        self,
        *,
        article: ArticleIn,
        extraction: ExtractionResult,
        resolved: dict[tuple[str, str], NormalizedEntity],
        raw_extracted: dict[str, Any],
        job_run_id: str,
    ) -> int:
        # Manual span instead of @mlflow.trace because `resolved` has
        # tuple keys (entity_type, normalized_name) which auto-input
        # capture can't JSON-serialize. We set explicit inputs/outputs.
        with mlflow.start_span(name="write_to_neo4j", span_type=SpanType.TOOL) as span:
            trace_id = _current_trace_id()
            mlflow_trace_url, mlflow_experiment_id = _mlflow_trace_link(self.settings, trace_id)
            try:
                span.set_inputs(
                    {
                        "article_url": article.url,
                        "entity_count": extraction.entity_count(),
                        "resolved_entity_count": len({e.id for e in resolved.values()}),
                    }
                )
                span.set_attributes(
                    {
                        "article_url": article.url,
                        "entity_count": extraction.entity_count(),
                    }
                )
            except Exception:  # pragma: no cover
                pass
            operations = await self.graph.ingest_article_bundle(
                article,
                extraction,
                resolved,
                raw_extracted_entities=raw_extracted,
                trace_id=trace_id,
                mlflow_trace_url=mlflow_trace_url,
                mlflow_experiment_id=mlflow_experiment_id,
                job_run_id=job_run_id,
            )
            try:
                span.set_attribute("graph_operations", operations)
                span.set_outputs({"graph_operations": operations})
            except Exception:  # pragma: no cover
                pass
            return operations

    # ------------------------------------------------------------------
    # Pure helpers (no MLflow / no tracing)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_candidate_evidence(entity: ExtractedEntity) -> str | None:
        return (entity.source.evidence or "").strip()[:_MAX_EVIDENCE_CHARS] or None

    @staticmethod
    def _registry_size(resolver: EntityResolver) -> int:
        for attr in ("registry_size", "_alias_index"):
            value = getattr(resolver, attr, None)
            if isinstance(value, int):
                return value
            if hasattr(value, "__len__"):
                try:
                    return len(value)
                except Exception:
                    pass
        return 0

    @staticmethod
    def _dedupe_topics(extraction: ExtractionResult) -> None:
        seen: set[str] = set()
        topics: list[ExtractedEntity] = []
        for topic in extraction.topics:
            key = NameNormalizer.key(topic.name, "Topic")
            if not key:
                continue
            if key not in seen:
                seen.add(key)
                topics.append(topic)
        extraction.topics = topics

    @staticmethod
    def _ensure_relationship_entities(extraction: ExtractionResult) -> None:
        by_type: dict[EntityType, list[ExtractedEntity]] = {
            "Startup": extraction.startups,
            "Investor": extraction.investors,
            "Person": extraction.people,
            "Topic": extraction.topics,
            "Company": extraction.companies,
        }

        # An admitted direct relationship is sufficient evidence to materialize its
        # endpoints when the model omitted or quarantined a separate entity record.
        rel_status: dict[tuple[EntityType, str], EvidenceStatus] = {}
        for relationship in extraction.relationships:
            if relationship.evidence_status not in ADMITTED_EVIDENCE_STATUSES:
                continue
            for etype, ename in (
                (relationship.source_type, relationship.source_name),
                (relationship.target_type, relationship.target_name),
            ):
                rkey = NameNormalizer.key(ename, etype)
                if rkey:
                    previous = rel_status.get((etype, rkey), "unsure")
                    rel_status[(etype, rkey)] = strongest_evidence_status(
                        previous, relationship.evidence_status
                    )

        def ensure(entity_type: EntityType, name: str) -> None:
            key = NameNormalizer.key(name, entity_type)
            if not key:
                return
            exists = any(
                NameNormalizer.key(entity.name, entity_type) == key
                for entity in by_type[entity_type]
            )
            if not exists:
                inherited = rel_status.get((entity_type, key), "unsure")
                if inherited in ADMITTED_EVIDENCE_STATUSES:
                    by_type[entity_type].append(
                        ExtractedEntity(
                            name=name,
                            evidence_status=inherited,
                        )
                    )

        for relationship in extraction.relationships:
            ensure(relationship.source_type, relationship.source_name)
            ensure(relationship.target_type, relationship.target_name)

    @staticmethod
    def _add_mapping(
        mapping: dict[tuple[str, str], NormalizedEntity],
        entity_type: str,
        entity: ExtractedEntity,
        outcome: ResolutionOutcome,
    ) -> None:
        resolved = outcome.entity
        names = [
            entity.name,
            *entity.aliases,
            resolved.name,
            resolved.canonical_name,
            *resolved.aliases,
        ]
        for name in names:
            key = NameNormalizer.key(name, entity_type)
            if key:
                mapping[(entity_type, key)] = resolved


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _raw_extracted_entities_snapshot(extraction: ExtractionResult) -> dict[str, Any]:
    source = extraction.raw_model_output or extraction.model_dump(mode="json")
    return {
        key: source.get(key, [])
        for key in (
            "startups",
            "investors",
            "people",
            "topics",
            "companies",
            "relationships",
        )
    }


def _evidence_status_counts(extraction: ExtractionResult) -> dict[str, int]:
    counts = {
        "entity_stated": 0,
        "entity_attributed": 0,
        "entity_unsure": 0,
        "entity_status_defaulted": 0,
        "relationship_stated": 0,
        "relationship_attributed": 0,
        "relationship_unsure": 0,
        "relationship_status_defaulted": 0,
    }
    for bucket in (
        extraction.startups,
        extraction.investors,
        extraction.people,
        extraction.topics,
        extraction.companies,
    ):
        for entity in bucket:
            counts[f"entity_{entity.evidence_status}"] += 1
            counts["entity_status_defaulted"] += int(entity.evidence_status_defaulted)
    for relationship in extraction.relationships:
        counts[f"relationship_{relationship.evidence_status}"] += 1
        counts["relationship_status_defaulted"] += int(relationship.evidence_status_defaulted)
    return counts


def _relationship_neighbor_keys(
    extraction: ExtractionResult,
) -> dict[tuple[EntityType, str], set[tuple[EntityType, str]]]:
    neighbors: dict[tuple[EntityType, str], set[tuple[EntityType, str]]] = {}
    for relationship in extraction.relationships:
        if relationship.evidence_status not in ADMITTED_EVIDENCE_STATUSES:
            continue
        source = _entity_key(relationship.source_type, relationship.source_name)
        target = _entity_key(relationship.target_type, relationship.target_name)
        if not source or not target or source == target:
            continue
        neighbors.setdefault(source, set()).add(target)
        neighbors.setdefault(target, set()).add(source)
    return neighbors


def _entity_key(entity_type: EntityType, name: str) -> tuple[EntityType, str] | None:
    key = NameNormalizer.key(name, entity_type)
    return (entity_type, key) if key else None


def _mlflow_trace_link(settings: Settings, trace_id: str | None) -> tuple[str | None, str | None]:
    if not trace_id:
        return None, None
    experiment_id = _mlflow_experiment_id(settings)
    if not experiment_id:
        return None, None
    base_url = (settings.mlflow_public_url or settings.mlflow_tracking_uri).rstrip("/")
    return (
        f"{base_url}/#/experiments/{quote(experiment_id, safe='')}/traces/{quote(trace_id, safe='')}",
        experiment_id,
    )


def _mlflow_experiment_id(settings: Settings) -> str | None:
    try:
        experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    except Exception:  # pragma: no cover - depends on the configured MLflow service
        return None
    return str(experiment.experiment_id) if experiment else None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[k])


__all__ = ["IngestionService"]
