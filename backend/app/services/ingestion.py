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
import json
from time import perf_counter
from typing import Any

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
    SourceAttribution,
    strongest_evidence_status,
)
from app.observability import IngestionRun, RunHandle
from app.observability.context import current_trace_id as _current_trace_id
from app.observability.costs import aggregate_llm_costs_for_job, llm_cost_metrics
from app.observability.artifacts import (
    build_dedup_report,
    build_ingestion_summary_md,
)
from app.observability.scorers import attach_extraction_scores
from app.observability.traces import mlflow_trace_link
from app.prompts.extraction import article_prompt_audit, build_article_prompt_input
from app.graph.article_graph_writer import ArticleGraphWriter
from app.graph.entity_resolution_store import EntityResolutionStore
from app.services.embedding import EmbeddingService
from app.services.entity_resolution import EntityResolver, NameNormalizer, ResolutionOutcome
from app.services.llm import LLMExtractionService, MISSING_OPENAI_API_KEY_MESSAGE
from app.services.progress import article_fields
from app.services.scraper import ArticleScraper
from app.topic_ontology import TopicOntologyReport, get_topic_ontology


class _ArticleResult:
    """Per-article aggregate returned by `_process_article`."""

    __slots__ = (
        "success",
        "article",
        "extract_row",
        "graph_row",
        "outcome_rows",
        "extraction_dump",
        "topic_ontology_report",
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
        self.topic_ontology_report: dict[str, Any] = {}
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
        article_writer: ArticleGraphWriter,
        resolution_store: EntityResolutionStore,
        embedding: EmbeddingService | None = None,
        profile_curation_policy_hash: str | None = None,
    ):
        self.settings = settings
        self.scraper = scraper
        self.llm = llm
        self.article_writer = article_writer
        self.resolution_store = resolution_store
        self.embedding = embedding
        self.profile_curation_policy_hash = profile_curation_policy_hash
        self.topic_ontology = get_topic_ontology()
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

        self.llm.pop_openai_rate_limit_stats()
        started = perf_counter()
        stats = IngestStats(source_name=request.source_name)
        run_id = ingest_run_id or "no-task-id"
        article_extraction_policy = self.llm.article_extraction_policy()
        article_extraction_policy_hash = self.llm.article_extraction_policy_hash()
        stats.article_extraction_policy_hash = article_extraction_policy_hash

        run_params = self._build_run_params(request, article_extraction_policy_hash)
        run_tags = self._build_run_tags(request)
        article_action = (
            "reprocess every article"
            if request.force_rescrape
            else "skip articles processed before"
        )

        async with timed_step(
            self.logger,
            "ingestion",
            workflow_step="ingest",
            task_id=run_id,
            url=str(request.source_url),
            start_summary=(
                f"Starting ingestion for {request.source_name}: collect article links, "
                f"{article_action}, extract entities, and update the graph."
            ),
        ):
            async with IngestionRun(
                job_run_id=run_id,
                source_name=request.source_name,
                params=run_params,
                tags=run_tags,
            ) as ingest_run:
                articles = await self._scrape_stage(
                    request,
                    stats,
                    ingest_run,
                    article_extraction_policy_hash=article_extraction_policy_hash,
                )
                if articles:
                    resolver = await self._registry_stage(ingest_run)
                    results = await self._dispatch_articles(
                        articles=articles,
                        resolver=resolver,
                        job_run_id=run_id,
                        article_extraction_policy_hash=article_extraction_policy_hash,
                        article_extraction_policy=article_extraction_policy,
                    )
                else:
                    results = []
                self._tally(results, stats)
                self._finalize_run(
                    ingest_run=ingest_run,
                    stats=stats,
                    results=results,
                    duration_ms=round((perf_counter() - started) * 1000, 2),
                )

        stats.duration_ms = round((perf_counter() - started) * 1000, 2)
        fetch_failures = self._article_fetch_failure_count(stats)
        total_failures = fetch_failures + stats.articles_failed
        self.logger.info(
            "ingest_completed",
            extra={
                "event": "ingestion",
                "workflow_step": "ingest",
                "task_id": run_id,
                "url": str(request.source_url),
                "duration_ms": stats.duration_ms,
                "summary": (
                    f"Finished {request.source_name}: {stats.articles_scraped} fetched, "
                    f"{stats.articles_skipped} skipped (already processed), "
                    f"{stats.articles_processed} processed, {total_failures} failed; "
                    f"{stats.entities_extracted} entities, "
                    f"{stats.relationships_created} graph operations."
                ),
                "article_links_selected": stats.articles_found,
                "articles_skipped_as_processed": stats.articles_cached,
                "articles_fetched": stats.articles_scraped,
                "articles_failed_or_invalid": fetch_failures,
                "articles_processed": stats.articles_processed,
                "articles_processing_failed": stats.articles_failed,
                "entities_extracted": stats.entities_extracted,
                "graph_operations": stats.relationships_created,
            },
        )
        return stats

    # ------------------------------------------------------------------
    # Run-level helpers
    # ------------------------------------------------------------------

    def _build_run_params(
        self,
        request: IngestRequest,
        article_extraction_policy_hash: str,
    ) -> dict[str, Any]:
        ontology = getattr(self, "topic_ontology", None) or get_topic_ontology()
        return {
            "max_pages": request.max_pages,
            "source_name": request.source_name,
            "source_url": str(request.source_url),
            "include_feed": request.include_feed,
            "force_rescrape": request.force_rescrape,
            "paths_count": len(request.paths),
            "openai_model": self.settings.openai_model,
            "openai_temperature": self.settings.openai_temperature,
            "openai_seed": self.settings.openai_seed,
            "admission_policy": "evidence_status:stated|attributed",
            "embedding_model": self.settings.embedding_model,
            "embedding_similarity_threshold": self.settings.embedding_similarity_threshold,
            "enable_embedding_resolution": self.settings.enable_embedding_resolution,
            "llm_retry_attempts": self.settings.llm_retry_attempts,
            "llm_max_concurrency": self.settings.llm_max_concurrency,
            "llm_gleaning_passes": self.settings.llm_gleaning_passes,
            "enable_entity_description_curation": (
                self.settings.enable_entity_description_curation
            ),
            "entity_curation_max_concurrency": self.settings.entity_curation_max_concurrency,
            "prompt_extraction_uri": self.settings.mlflow_prompt_extraction_uri,
            "prompt_gleaning_uri": self.settings.mlflow_prompt_gleaning_uri,
            "prompt_profile_review_uri": self.settings.mlflow_prompt_profile_review_uri,
            "prompt_profile_curation_uri": (self.settings.mlflow_prompt_profile_curation_uri),
            "use_prompt_registry": self.settings.mlflow_use_prompt_registry,
            "article_extraction_policy_hash": article_extraction_policy_hash,
            "profile_curation_policy_hash": self.profile_curation_policy_hash,
            "topic_ontology_version": ontology.ontology_version,
            "topic_ontology_hash": ontology.content_hash,
            "topic_ontology_mode": "closed_exact_aliases",
            "app_env": self.settings.app_env,
            "git_sha": self.settings.git_sha,
            "app_version": self.settings.app_version,
        }

    def _build_run_tags(self, request: IngestRequest) -> dict[str, str]:
        ontology = getattr(self, "topic_ontology", None) or get_topic_ontology()
        return {
            "env": self.settings.app_env,
            "git_sha": self.settings.git_sha,
            "app_version": self.settings.app_version,
            "source_name": request.source_name,
            "model": self.settings.openai_model,
            "topic_ontology_version": ontology.ontology_version,
        }

    @staticmethod
    def _article_count(count: int) -> str:
        noun = "article" if count == 1 else "articles"
        return f"{count} {noun}"

    @staticmethod
    def _article_link_count(count: int) -> str:
        noun = "article link" if count == 1 else "article links"
        return f"{count} {noun}"

    @staticmethod
    def _article_fetch_failure_count(stats: IngestStats) -> int:
        return max(
            stats.articles_found - stats.articles_skipped - stats.articles_scraped,
            0,
        )

    def _article_collection_summary(self, stats: IngestStats) -> str:
        return (
            f"{stats.articles_found} links: {stats.articles_scraped} fetched, "
            f"{stats.articles_skipped} skipped, "
            f"{self._article_fetch_failure_count(stats)} failed."
        )

    def _article_cache_summary(
        self,
        *,
        found_count: int,
        cached_count: int,
        fetch_count: int,
        force_rescrape: bool,
    ) -> str:
        if found_count == 0:
            return "No links found."
        if force_rescrape:
            return f"{found_count} links: force re-scrape, fetching {fetch_count}."
        return f"{found_count} links: {cached_count} already processed, fetching {fetch_count}."

    def _article_cache_all_cached_message(self, found_count: int) -> str:
        found = self._article_count(found_count)
        return f"All {found} were already processed."

    async def _scrape_stage(
        self,
        request: IngestRequest,
        stats: IngestStats,
        ingest_run: RunHandle,
        *,
        article_extraction_policy_hash: str,
    ) -> list[ArticleIn]:
        stage_started = perf_counter()
        self.logger.info(
            "workflow_stage_started",
            extra={
                "event": "ingestion",
                "workflow_step": "scraping",
                "url": str(request.source_url),
                "summary": f"Collecting article links from {request.source_name}.",
            },
        )
        links = await self.scraper.collect_links(
            source_url=str(request.source_url),
            source_name=request.source_name,
            max_pages=request.max_pages,
            include_feed=request.include_feed,
            paths=request.paths,
        )
        stats.articles_found = len(links)

        cached_links: set[str] = set()
        cache_lookup_failed = False
        if not request.force_rescrape and links:
            cached_links, cache_lookup_failed = await self._cached_article_links(
                links,
                article_extraction_policy_hash,
            )

        if request.force_rescrape:
            links_to_fetch = links
        else:
            links_to_fetch = [link for link in links if link not in cached_links]
        stats.articles_cached = len(cached_links)
        stats.articles_skipped = 0 if request.force_rescrape else len(cached_links)

        if not cache_lookup_failed:
            self.logger.info(
                "article_extraction_cache_checked",
                extra={
                    "event": "scraping",
                    "workflow_step": "article_cache",
                    "summary": self._article_cache_summary(
                        found_count=len(links),
                        cached_count=len(cached_links),
                        fetch_count=len(links_to_fetch),
                        force_rescrape=request.force_rescrape,
                    ),
                    "article_links_selected": len(links),
                    "articles_skipped_as_processed": len(cached_links),
                    "articles_to_fetch": len(links_to_fetch),
                },
            )

        if links and not links_to_fetch:
            stats.cache_message = self._article_cache_all_cached_message(stats.articles_found)
            articles = []
        elif links_to_fetch:
            articles = await self.scraper.fetch_articles(
                source_url=str(request.source_url),
                source_name=request.source_name,
                links=links_to_fetch,
            )
        else:
            articles = []

        stats.articles_scraped = len(articles)
        duration_ms = round((perf_counter() - stage_started) * 1000, 2)
        ingest_run.record_metric("scrape_duration_ms", duration_ms)
        ingest_run.record_metric("articles_found", float(stats.articles_found))
        ingest_run.record_metric("articles_cached", float(stats.articles_cached))
        ingest_run.record_metric("articles_scraped", float(stats.articles_scraped))
        ingest_run.record_metric("articles_skipped", float(stats.articles_skipped))
        self.logger.info(
            "workflow_stage_completed",
            extra={
                "event": "ingestion",
                "workflow_step": "scraping",
                "url": str(request.source_url),
                "duration_ms": duration_ms,
                "summary": self._article_collection_summary(stats),
                "article_links_selected": stats.articles_found,
                "articles_skipped_as_processed": stats.articles_cached,
                "articles_fetched": stats.articles_scraped,
                "articles_failed_or_invalid": self._article_fetch_failure_count(stats),
            },
        )
        return articles

    async def _cached_article_links(
        self,
        links: list[str],
        article_extraction_policy_hash: str,
    ) -> tuple[set[str], bool]:
        try:
            cached_links = await self.article_writer.cached_article_urls_for_extraction_policy(
                links,
                article_extraction_policy_hash,
            )
            return cached_links, False
        except Exception as exc:
            self.logger.warning(
                "article_extraction_cache_lookup_failed",
                extra={
                    "event": "scraping",
                    "workflow_step": "article_cache",
                    "article_links_found": len(links),
                    "error": str(exc),
                    "summary": (
                        "Could not check which articles were already processed; fetching all "
                        f"{self._article_link_count(len(links))}."
                    ),
                },
            )
            return set(), True

    async def _registry_stage(self, ingest_run: RunHandle) -> EntityResolver:
        stage_started = perf_counter()
        self.logger.info(
            "workflow_stage_started",
            extra={
                "event": "ingestion",
                "workflow_step": "resolution_registry",
                "summary": "Loading existing graph entities and aliases for matching.",
            },
        )
        resolver = EntityResolver(self.settings, self.embedding, self.resolution_store)
        await resolver.load_from_graph(self.resolution_store)
        duration_ms = round((perf_counter() - stage_started) * 1000, 2)
        registry_size = self._registry_size(resolver)
        ingest_run.record_metric("registry_load_duration_ms", duration_ms)
        ingest_run.record_metric("registry_size", float(registry_size))
        self.logger.info(
            "workflow_stage_completed",
            extra={
                "event": "ingestion",
                "workflow_step": "resolution_registry",
                "duration_ms": duration_ms,
                "summary": (
                    f"Loaded {registry_size} existing entities and their aliases for matching."
                ),
                "registry_size": registry_size,
            },
        )
        return resolver

    async def _dispatch_articles(
        self,
        *,
        articles: list[ArticleIn],
        resolver: EntityResolver,
        job_run_id: str,
        article_extraction_policy_hash: str,
        article_extraction_policy: dict[str, Any],
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
                    article_extraction_policy_hash=article_extraction_policy_hash,
                    article_extraction_policy=article_extraction_policy,
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
                stats.topic_candidates += int(
                    result.topic_ontology_report.get("topic_candidates", 0) or 0
                )
                stats.topics_kept += int(result.topic_ontology_report.get("topics_kept", 0) or 0)
                stats.topics_rejected += int(
                    result.topic_ontology_report.get("topics_rejected", 0) or 0
                )
                stats.has_topic_relationships_rejected += int(
                    result.topic_ontology_report.get("has_topic_rejected", 0) or 0
                )
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
        topic_rejection_rows: list[dict[str, Any]] = []
        fallback_input_tokens = 0
        fallback_output_tokens = 0

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
                for rejection in result.topic_ontology_report.get("rejections", []):
                    topic_rejection_rows.append(
                        {
                            "article_url": (
                                result.article.url if result.article is not None else None
                            ),
                            "article_title": (
                                result.article.title if result.article is not None else None
                            ),
                            "ontology_version": result.topic_ontology_report.get(
                                "ontology_version"
                            ),
                            "ontology_hash": result.topic_ontology_report.get("ontology_hash"),
                            **rejection,
                        }
                    )
                fallback_input_tokens += result.input_tokens
                fallback_output_tokens += result.output_tokens
            elif result.failure_row:
                failure_rows.append(result.failure_row)

        llm_cost_rows = aggregate_llm_costs_for_job(
            settings=self.settings,
            job_run_id=ingest_run.job_run_id,
        )
        if llm_cost_rows:
            ingest_run.add_token_usage(
                input_tokens=sum(int(row.get("input_tokens") or 0) for row in llm_cost_rows),
                output_tokens=sum(int(row.get("output_tokens") or 0) for row in llm_cost_rows),
                cost_usd=sum(float(row.get("total_cost_usd") or 0.0) for row in llm_cost_rows),
            )
        elif fallback_input_tokens or fallback_output_tokens:
            ingest_run.add_token_usage(
                input_tokens=fallback_input_tokens,
                output_tokens=fallback_output_tokens,
            )

        attempts_list = [
            int(row.get("attempts") or 0) for row in extraction_rows if row.get("attempts")
        ]
        metrics: dict[str, float] = {
            "articles_found": float(stats.articles_found),
            "articles_cached": float(stats.articles_cached),
            "articles_scraped": float(stats.articles_scraped),
            "articles_skipped": float(stats.articles_skipped),
            "articles_processed": float(stats.articles_processed),
            "articles_failed": float(stats.articles_failed),
            "entities_extracted": float(stats.entities_extracted),
            "relationships_created": float(stats.relationships_created),
            "topic_candidates": float(stats.topic_candidates),
            "topics_kept": float(stats.topics_kept),
            "topics_rejected": float(stats.topics_rejected),
            "has_topic_relationships_rejected": float(stats.has_topic_relationships_rejected),
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
        rate_limit_stats = self.llm.pop_openai_rate_limit_stats()
        metrics.update(
            {
                "openai_rate_limit_events": rate_limit_stats["events"],
                "openai_rate_limit_retries": rate_limit_stats["retries"],
                "openai_rate_limit_failures": rate_limit_stats["failures"],
                "openai_rate_limit_wait_seconds": rate_limit_stats["wait_seconds"],
            }
        )
        if llm_cost_rows:
            cost_metrics = llm_cost_metrics(llm_cost_rows)
            metrics.update(cost_metrics)
            if "llm_cost_usd" in cost_metrics:
                metrics["llm_total_cost_usd"] = cost_metrics["llm_cost_usd"]
            if stats.articles_processed:
                metrics["llm_cost_per_article_usd"] = round(
                    cost_metrics.get("llm_cost_usd", 0.0) / stats.articles_processed,
                    6,
                )
            if stats.entities_extracted:
                metrics["llm_cost_per_entity_usd"] = round(
                    cost_metrics.get("llm_cost_usd", 0.0) / stats.entities_extracted,
                    6,
                )
        ingest_run.record_metrics(metrics)
        self._log_rate_limit_summary(rate_limit_stats)

        if extraction_rows:
            ingest_run.add_jsonl_artifact("extraction_summary.jsonl", extraction_rows)
        if graph_rows:
            ingest_run.add_jsonl_artifact("graph_ops.jsonl", graph_rows)
        if extraction_dumps:
            ingest_run.add_jsonl_artifact("extraction_dump.jsonl", extraction_dumps)
        if topic_rejection_rows:
            ingest_run.add_jsonl_artifact(
                "topic_ontology_rejections.jsonl",
                topic_rejection_rows,
            )
        if failure_rows:
            ingest_run.add_jsonl_artifact("failed_articles.jsonl", failure_rows)
        if outcome_rows:
            ingest_run.add_json_artifact(
                "dedup_report.json",
                build_dedup_report(outcome_rows=outcome_rows),
            )
        if llm_cost_rows:
            ingest_run.add_jsonl_artifact("llm_costs.jsonl", llm_cost_rows)

        ingest_run.add_artifact(
            "ingestion_summary.md",
            build_ingestion_summary_md(
                job_run_id=ingest_run.job_run_id,
                source_name=ingest_run.metrics.get("source_name", "") or self.settings.app_env,
                metrics=metrics,
                failures_sample=failure_rows,
            ),
        )

    def _log_rate_limit_summary(self, stats: dict[str, float]) -> None:
        detail = (
            f"events={int(stats['events'])}, retries={int(stats['retries'])}, "
            f"failures={int(stats['failures'])}, wait_seconds={stats['wait_seconds']}"
        )
        self.logger.info(
            "openai_rate_limit_summary",
            extra={
                "event": "ingestion",
                "workflow_step": "ingest",
                "count": int(stats["events"]),
                "failed_count": int(stats["failures"]),
                "retry_delay_seconds": stats["wait_seconds"],
                "detail": detail,
            },
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
        article_extraction_policy_hash: str,
        article_extraction_policy: dict[str, Any],
        article_index: int,
        article_total: int,
    ) -> _ArticleResult:
        result = _ArticleResult()
        result.article = article
        stage_started = perf_counter()
        cleaning_payload = article.cleaning.model_dump(mode="json") if article.cleaning else None
        prompt_input = build_article_prompt_input(article)
        prompt_audit = article_prompt_audit(article)

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
                        "primary_type": article.primary_type,
                        "prompt_input": prompt_input,
                        **prompt_audit,
                        "cleaning": cleaning_payload,
                    }
                )
                attributes: dict[str, Any] = {
                    "article_url": article.url,
                    "source_name": article.source_name,
                    "text_chars": len(article.text),
                    "tag_count": len(article.tags),
                    "primary_type": article.primary_type or "unknown",
                    **prompt_audit,
                }
                if article.cleaning:
                    attributes.update(
                        {
                            "content_container": article.cleaning.selected_container,
                            "content_blocks_before": article.cleaning.blocks_before,
                            "content_blocks_after": article.cleaning.blocks_after,
                            "content_chars_before": article.cleaning.text_chars_before,
                            "content_chars_after": article.cleaning.text_chars_after,
                            "content_removed_blocks": len(article.cleaning.removed_blocks),
                            "content_remaining_promotion_markers": len(
                                article.cleaning.remaining_promotion_markers
                            ),
                        }
                    )
                span.set_attributes(attributes)
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
        topic_report = self._enforce_topic_ontology_traced(article, cleaned)
        topic_report_payload = topic_report.to_dict()
        result.topic_ontology_report = topic_report_payload

        if span is not None:
            try:
                span.set_attribute("entity_count_pre_filter", pre_filter_count)
                span.set_attribute("entity_count_post_filter", cleaned.entity_count())
                span.set_attributes(
                    {
                        key: value
                        for key, value in topic_report_payload.items()
                        if key != "rejections"
                    }
                )
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
                    article_extraction_policy_hash=article_extraction_policy_hash,
                    article_extraction_policy=article_extraction_policy,
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
            "primary_type": article.primary_type,
            "selected_container": (
                article.cleaning.selected_container if article.cleaning else None
            ),
            "text_chars_before_cleaning": (
                article.cleaning.text_chars_before if article.cleaning else len(article.text)
            ),
            "text_chars_after_cleaning": (
                article.cleaning.text_chars_after if article.cleaning else len(article.text)
            ),
            "removed_block_count": (
                len(article.cleaning.removed_blocks) if article.cleaning else 0
            ),
            "remaining_promotion_marker_count": (
                len(article.cleaning.remaining_promotion_markers) if article.cleaning else 0
            ),
            **prompt_audit,
            "status": "succeeded",
            "entity_count": cleaned.entity_count(),
            "relationship_count": len(cleaned.relationships),
            **raw_status_counts,
            **_topic_report_counts(topic_report_payload),
            "attempts": extraction_metadata.get("attempts"),
            "latency_ms": extraction_metadata.get("latency_ms"),
            "trace_id": extraction_metadata.get("trace_id") or _current_trace_id(),
            "prompt_name": extraction_metadata.get("prompt_name"),
            "prompt_version": extraction_metadata.get("prompt_version"),
            "prompt_uri": extraction_metadata.get("prompt_uri"),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        }
        result.graph_row = {
            "article_url": article.url,
            "title": (article.title or "")[:200],
            "source_name": article.source_name,
            "status": "succeeded",
            "entity_count": cleaned.entity_count(),
            "relationship_count": len(cleaned.relationships),
            **raw_status_counts,
            **_topic_report_counts(topic_report_payload),
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
            "primary_type": article.primary_type,
            "cleaning": cleaning_payload,
            "prompt_input": extraction_metadata.get("prompt_input", prompt_input),
            "user_prompt": extraction_metadata.get("user_prompt"),
            **prompt_audit,
            "trace_id": _current_trace_id(),
            "extraction": cleaned.model_dump(mode="json"),
            "raw_extraction": raw_snapshot,
            "evidence_status_counts": raw_status_counts,
            "topic_ontology": topic_report_payload,
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
                        "topic_ontology": topic_report_payload,
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

    def _enforce_topic_ontology_traced(
        self,
        article: ArticleIn,
        extraction: ExtractionResult,
    ) -> TopicOntologyReport:
        ontology = getattr(self, "topic_ontology", None) or get_topic_ontology()
        with mlflow.start_span(name="topic_ontology_gate", span_type=SpanType.CHAIN) as span:
            report = ontology.enforce(extraction)
            payload = report.to_dict()
            if report.rejections:
                self.logger.warning(
                    "topic_ontology_rejections",
                    extra={
                        "event": "extraction",
                        "workflow_step": "topic_ontology",
                        "url": article.url,
                        "count": len(report.rejections),
                        "skipped_count": (report.topics_rejected + report.has_topic_rejected),
                        "detail": json.dumps(
                            report.rejections,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                )
            try:
                span.set_inputs(
                    {
                        "article_url": article.url,
                        "topic_candidates": report.topic_candidates,
                        "has_topic_candidates": report.has_topic_candidates,
                    }
                )
                span.set_outputs(payload)
                span.set_attributes(
                    {key: value for key, value in payload.items() if key != "rejections"}
                )
            except Exception:  # pragma: no cover
                pass
            return report

    async def _resolve_traced(
        self,
        resolver: EntityResolver,
        article: ArticleIn,
        extraction: ExtractionResult,
    ) -> tuple[dict[tuple[str, str], NormalizedEntity], list[ResolutionOutcome]]:
        # Manual span — the return value contains a dict with tuple keys
        # which the @mlflow.trace auto-capture cannot JSON-serialize.
        with mlflow.start_span(name="resolve_entities", span_type=SpanType.CHAIN) as span:
            return await self._do_resolve_entities(span, resolver, article, extraction)

    async def _do_resolve_entities(
        self,
        span,
        resolver: EntityResolver,
        article: ArticleIn,
        extraction: ExtractionResult,
    ) -> tuple[dict[tuple[str, str], NormalizedEntity], list[ResolutionOutcome]]:
        self._ensure_relationship_entities(extraction, article)
        relationship_neighbors = _relationship_neighbor_keys(extraction)
        ontology = getattr(self, "topic_ontology", None) or get_topic_ontology()

        _entity_groups: list[tuple[str, list[ExtractedEntity]]] = [
            ("Startup", extraction.startups),
            ("Investor", extraction.investors),
            ("Person", extraction.people),
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
        for topic in extraction.topics:
            concept = ontology.lookup(topic.name)
            if concept is None:
                raise ValueError(
                    f"Topic reached resolution outside the closed ontology: {topic.name!r}"
                )
            resolved_topic = ontology.normalized_entity(concept, topic)
            self._add_mapping(mapping, "Topic", topic, resolved_topic)

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
                self._add_mapping(mapping, entity_type, entity, outcome.entity)

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

            outcome = await resolver.resolve(
                entity_type,
                entity,
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
                        "candidate_type": entity_type,
                        "resolved_type": outcome.entity.label,
                        "type_promoted": outcome.type_promoted,
                        "similarity_min": outcome.similarity_min,
                    }
                )
                span.set_attribute("resolution_method", outcome.method)
                span.set_attribute("resolved_entity_type", outcome.entity.label)
                span.set_attribute("type_promoted", outcome.type_promoted)
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
        article_extraction_policy_hash: str,
        article_extraction_policy: dict[str, Any],
    ) -> int:
        # Manual span instead of @mlflow.trace because `resolved` has
        # tuple keys (entity_type, normalized_name) which auto-input
        # capture can't JSON-serialize. We set explicit inputs/outputs.
        with mlflow.start_span(name="write_to_neo4j", span_type=SpanType.TOOL) as span:
            trace_id = _current_trace_id()
            mlflow_trace_url, mlflow_experiment_id = mlflow_trace_link(self.settings, trace_id)
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
            operations = await self.article_writer.ingest_article_bundle(
                article,
                extraction,
                resolved,
                raw_extracted_entities=raw_extracted,
                trace_id=trace_id,
                mlflow_trace_url=mlflow_trace_url,
                mlflow_experiment_id=mlflow_experiment_id,
                job_run_id=job_run_id,
                article_extraction_policy_hash=article_extraction_policy_hash,
                article_extraction_policy=article_extraction_policy,
                profile_curation_policy_hash=self.profile_curation_policy_hash,
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
    def _ensure_relationship_entities(
        extraction: ExtractionResult,
        article: ArticleIn,
    ) -> None:
        by_type: dict[EntityType, list[ExtractedEntity]] = {
            "Startup": extraction.startups,
            "Investor": extraction.investors,
            "Person": extraction.people,
            "Topic": extraction.topics,
            "Company": extraction.companies,
        }

        # An admitted direct relationship is sufficient evidence to materialize its
        # endpoints when the model omitted or quarantined a separate entity record.
        rel_support: dict[tuple[EntityType, str], tuple[EvidenceStatus, str]] = {}
        for relationship in extraction.relationships:
            if relationship.evidence_status not in ADMITTED_EVIDENCE_STATUSES:
                continue
            evidence = (relationship.evidence or "").strip()
            if not evidence:
                continue
            for etype, ename in (
                (relationship.source_type, relationship.source_name),
                (relationship.target_type, relationship.target_name),
            ):
                rkey = NameNormalizer.key(ename, etype)
                if rkey:
                    key = (etype, rkey)
                    previous = rel_support.get(key)
                    if (
                        previous is None
                        or strongest_evidence_status(
                            previous[0],
                            relationship.evidence_status,
                        )
                        != previous[0]
                    ):
                        rel_support[key] = (
                            relationship.evidence_status,
                            evidence,
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
                support = rel_support.get((entity_type, key))
                if support is not None:
                    evidence_status, evidence = support
                    by_type[entity_type].append(
                        ExtractedEntity(
                            name=name,
                            type_basis="contextual",
                            evidence_status=evidence_status,
                            description=evidence,
                            source=SourceAttribution(
                                article_url=article.url,
                                article_title=article.title,
                                evidence=evidence,
                            ),
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
        resolved: NormalizedEntity,
    ) -> None:
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


def _topic_report_counts(report: dict[str, Any]) -> dict[str, int]:
    return {
        "topic_candidates": int(report.get("topic_candidates", 0) or 0),
        "topics_kept": int(report.get("topics_kept", 0) or 0),
        "topics_rejected": int(report.get("topics_rejected", 0) or 0),
        "has_topic_candidates": int(report.get("has_topic_candidates", 0) or 0),
        "has_topic_kept": int(report.get("has_topic_kept", 0) or 0),
        "has_topic_relationships_rejected": int(report.get("has_topic_rejected", 0) or 0),
        "topic_names_canonicalized": int(report.get("topic_names_canonicalized", 0) or 0),
        "orphan_topics_removed": int(report.get("orphan_topics_removed", 0) or 0),
        "duplicate_topics_merged": int(report.get("duplicate_topics_merged", 0) or 0),
        "duplicate_has_topic_merged": int(report.get("duplicate_has_topic_merged", 0) or 0),
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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[k])


__all__ = ["IngestionService"]
