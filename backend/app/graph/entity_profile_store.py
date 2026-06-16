import json
from datetime import UTC, datetime
from typing import Any, Iterable

from app.db.neo4j import Neo4jClient
from app.graph.common import (
    PROFILE_ENTITY_LABELS,
    PROFILE_EVIDENCE_MAX_CHARS,
    jsonable,
    profile_evidence_id,
)
from app.models.extraction import ArticleIn, EvidenceStatus


def profile_article_policy_key(
    article_id: Any,
    profile_curation_policy_hash: str,
) -> str | None:
    article_id_text = str(article_id or "").strip()
    policy_hash = str(profile_curation_policy_hash or "").strip()
    if not article_id_text or not policy_hash:
        return None
    return f"{article_id_text}::{policy_hash}"


def profile_article_policy_keys(
    article_ids: Iterable[Any],
    profile_curation_policy_hash: str,
) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for article_id in article_ids:
        key = profile_article_policy_key(article_id, profile_curation_policy_hash)
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


class EntityProfileStore:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

    async def profile_review_inputs(
        self,
        *,
        profile_curation_policy_hash: str,
        entity_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates for entity profile curation.

        When ``entity_ids`` is provided, candidates are limited to those resolved
        entities. The evidence selection remains the same as the graph-wide
        curation path: only profile evidence from articles that have not already
        contributed to the current profile under the current curation policy is
        surfaced as new evidence.
        """
        if entity_ids is not None and not entity_ids:
            return []
        query = """
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN $entity_labels)
          AND ($entity_ids IS NULL OR n.id IN $entity_ids)
        WITH n, coalesce(n.description_considered_article_policy_keys, []) AS considered_article_policy_keys
        OPTIONAL MATCH (n)-[:HAS_PROFILE_EVIDENCE]->(evidence:ProfileEvidence)
        WITH n, considered_article_policy_keys,
             collect(DISTINCT evidence) AS profile_evidence_nodes
        WITH n,
             [
               evidence IN profile_evidence_nodes
               WHERE evidence IS NOT NULL
                 AND evidence.kind = "entity_description"
                 AND coalesce(evidence.text, "") <> ""
                 AND coalesce(evidence.article_id, "") <> ""
                 AND coalesce(evidence.article_id, "") + "::" + $profile_curation_policy_hash
                     IN considered_article_policy_keys
               | {
                 id: evidence.id,
                 article_id: evidence.article_id,
                 text: evidence.text
               }
             ] AS considered_evidence,
             [
               evidence IN profile_evidence_nodes
               WHERE coalesce(evidence.profile_candidate, false) = true
                 AND evidence.kind = "entity_description"
                 AND coalesce(evidence.article_id, "") <> ""
                 AND NOT (
                   coalesce(evidence.article_id, "") + "::" + $profile_curation_policy_hash
                       IN considered_article_policy_keys
                 )
             ] AS new_evidence
        WHERE size(new_evidence) > 0
           OR coalesce(n.description, "") = ""
           OR n.embedding IS NULL
        RETURN n.id AS id,
               head([label IN labels(n) WHERE label IN $entity_labels]) AS label,
               n.name AS name,
               n.canonical_name AS canonical_name,
               n.aliases AS aliases,
               n.description AS current_description,
               n.description_source AS description_source,
               coalesce(n.description_profile_revision, 0) AS profile_revision,
               n.description_curation_status AS description_curation_status,
               n.description_confidence AS description_confidence,
               n.description_review_decision AS description_review_decision,
               n.description_trace_id AS description_trace_id,
               n.description_trace_url AS description_trace_url,
               n.description_mlflow_experiment_id AS description_mlflow_experiment_id,
               n.description_traced_at AS description_traced_at,
               n.description_curation_history AS description_curation_history,
               n.embedding IS NOT NULL AS has_embedding,
               n.embedding_input_hash AS embedding_input_hash,
               [evidence IN new_evidence | {
                 id: evidence.id,
                 article_id: evidence.article_id,
                 article_url: evidence.article_url,
                 text: evidence.text,
                 published_at: evidence.published_at,
                 created_at: evidence.created_at,
                 last_seen_at: evidence.last_seen_at
               }] AS new_evidence,
               considered_evidence AS considered_evidence
        """
        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                entity_labels=PROFILE_ENTITY_LABELS,
                entity_ids=entity_ids,
                profile_curation_policy_hash=profile_curation_policy_hash,
            )
            return [jsonable(dict(record)) async for record in result]

    async def save_profile_review_keep(
        self,
        *,
        entity_id: str,
        evidence_ids: list[str],
        article_ids: list[str],
        profile_curation_policy_hash: str,
        review: dict[str, Any],
        model: str,
        embedding: list[float] | None = None,
        embedding_input_hash: str | None = None,
        trace_id: str | None = None,
        mlflow_trace_url: str | None = None,
        mlflow_experiment_id: str | None = None,
    ) -> None:
        async with self.neo4j.session() as session:
            await session.run(
                """
                MATCH (n {id: $entity_id})
                WITH n, coalesce(n.description_profile_revision, 0) AS revision
                SET n.updated_at = datetime(),
                    n.description_source = CASE
                      WHEN coalesce(n.description, "") = "" THEN n.description_source
                      WHEN $decision = "exact_duplicate_evidence" THEN "reviewed_exact_duplicate"
                      ELSE "reviewed_llm"
                    END,
                    n.description_curation_status = "reviewed_keep",
                    n.description_reviewed_at = datetime(),
                    n.description_model = $model,
                    n.description_review_decision = $decision,
                    n.description_review_reason = $reason,
                    n.description_confidence = $confidence,
                    n.description_curation_policy_hash = $profile_curation_policy_hash,
                    n.description_trace_id = CASE
                      WHEN $trace_id IS NULL OR $trace_id = "" THEN n.description_trace_id
                      ELSE $trace_id
                    END,
                    n.description_trace_url = CASE
                      WHEN $mlflow_trace_url IS NULL OR $mlflow_trace_url = "" THEN n.description_trace_url
                      ELSE $mlflow_trace_url
                    END,
                    n.description_mlflow_experiment_id = CASE
                      WHEN $mlflow_experiment_id IS NULL OR $mlflow_experiment_id = ""
                      THEN n.description_mlflow_experiment_id
                      ELSE $mlflow_experiment_id
                    END,
                    n.description_traced_at = CASE
                      WHEN $trace_id IS NULL OR $trace_id = "" THEN n.description_traced_at
                      ELSE datetime()
                    END,
                    n.description_considered_evidence_ids =
                      reduce(acc = coalesce(n.description_considered_evidence_ids, []),
                             evidence_id IN $evidence_ids |
                             CASE WHEN evidence_id IN acc THEN acc ELSE acc + [evidence_id] END),
                    n.description_considered_article_ids =
                      reduce(acc = coalesce(n.description_considered_article_ids, []),
                             article_id IN $article_ids |
                             CASE WHEN article_id IN acc THEN acc ELSE acc + [article_id] END),
                    n.description_considered_article_policy_keys =
                      reduce(acc = coalesce(n.description_considered_article_policy_keys, []),
                             article_policy_key IN $article_policy_keys |
                             CASE
                               WHEN article_policy_key IN acc THEN acc
                               ELSE acc + [article_policy_key]
                             END),
                    n.embedding = CASE
                      WHEN $embedding IS NULL THEN n.embedding ELSE $embedding
                    END,
                    n.embedding_input_hash = CASE
                      WHEN $embedding_input_hash IS NULL THEN n.embedding_input_hash
                      ELSE $embedding_input_hash
                    END,
                    n.embedding_profile_revision = CASE
                      WHEN $embedding IS NULL THEN n.embedding_profile_revision ELSE revision
                    END,
	                    n.embedding_model = CASE
	                      WHEN $embedding IS NULL THEN n.embedding_model ELSE $model
	                    END,
	                    n.embedding_updated_at = CASE
	                      WHEN $embedding IS NULL THEN n.embedding_updated_at ELSE datetime()
	                    END,
	                    n.description_curation_history =
	                      coalesce(n.description_curation_history, []) + [$history_event]
                """,
                entity_id=entity_id,
                evidence_ids=evidence_ids,
                article_ids=article_ids,
                article_policy_keys=profile_article_policy_keys(
                    article_ids,
                    profile_curation_policy_hash,
                ),
                profile_curation_policy_hash=profile_curation_policy_hash,
                model=model,
                decision=review.get("decision"),
                reason=review.get("reason"),
                confidence=review.get("confidence"),
                embedding=embedding,
                embedding_input_hash=embedding_input_hash,
                trace_id=trace_id,
                mlflow_trace_url=mlflow_trace_url,
                mlflow_experiment_id=mlflow_experiment_id,
                history_event=_profile_history_event(
                    status=(
                        "skipped_exact_duplicate_evidence"
                        if review.get("decision") == "exact_duplicate_evidence"
                        else "kept"
                    ),
                    review=review,
                    model=model,
                    trace_id=trace_id,
                    mlflow_trace_url=mlflow_trace_url,
                    mlflow_experiment_id=mlflow_experiment_id,
                ),
            )
            await _mark_profile_evidence_considered(
                session,
                entity_id=entity_id,
                evidence_ids=evidence_ids,
                revision=None,
                decision=str(review.get("decision") or "keep_profile"),
                review=review,
                profile_curation_policy_hash=profile_curation_policy_hash,
            )

    async def save_profile_revision(
        self,
        *,
        entity_id: str,
        description: str,
        evidence_ids_considered: list[str],
        used_evidence_ids: list[str],
        article_ids_considered: list[str],
        profile_curation_policy_hash: str,
        review: dict[str, Any],
        curation: dict[str, Any],
        model: str,
        embedding: list[float] | None = None,
        embedding_input_hash: str | None = None,
        trace_id: str | None = None,
        mlflow_trace_url: str | None = None,
        mlflow_experiment_id: str | None = None,
    ) -> None:
        async with self.neo4j.session() as session:
            result = await session.run(
                """
                MATCH (n {id: $entity_id})
                WITH n, coalesce(n.description_profile_revision, 0) + 1 AS revision
                SET n.updated_at = datetime(),
                    n.description = $description,
                    n.description_source = "curated_llm",
                    n.description_profile_revision = revision,
                    n.description_curation_status = "curated",
                    n.description_curated_at = datetime(),
                    n.description_reviewed_at = datetime(),
                    n.description_model = $model,
                    n.description_review_decision = $review_decision,
                    n.description_review_reason = $review_reason,
                    n.description_confidence = $confidence,
                    n.description_curation_policy_hash = $profile_curation_policy_hash,
                    n.description_trace_id = CASE
                      WHEN $trace_id IS NULL OR $trace_id = "" THEN n.description_trace_id
                      ELSE $trace_id
                    END,
                    n.description_trace_url = CASE
                      WHEN $mlflow_trace_url IS NULL OR $mlflow_trace_url = "" THEN n.description_trace_url
                      ELSE $mlflow_trace_url
                    END,
                    n.description_mlflow_experiment_id = CASE
                      WHEN $mlflow_experiment_id IS NULL OR $mlflow_experiment_id = ""
                      THEN n.description_mlflow_experiment_id
                      ELSE $mlflow_experiment_id
                    END,
                    n.description_traced_at = CASE
                      WHEN $trace_id IS NULL OR $trace_id = "" THEN n.description_traced_at
                      ELSE datetime()
                    END,
                    n.description_considered_evidence_ids =
                      reduce(acc = coalesce(n.description_considered_evidence_ids, []),
                             evidence_id IN $evidence_ids_considered |
                             CASE WHEN evidence_id IN acc THEN acc ELSE acc + [evidence_id] END),
                    n.description_used_evidence_ids =
                      reduce(acc = coalesce(n.description_used_evidence_ids, []),
                             evidence_id IN $used_evidence_ids |
                             CASE WHEN evidence_id IN acc THEN acc ELSE acc + [evidence_id] END),
                    n.description_considered_article_ids =
                      reduce(acc = coalesce(n.description_considered_article_ids, []),
                             article_id IN $article_ids_considered |
                             CASE WHEN article_id IN acc THEN acc ELSE acc + [article_id] END),
                    n.description_considered_article_policy_keys =
                      reduce(acc = coalesce(n.description_considered_article_policy_keys, []),
                             article_policy_key IN $article_policy_keys |
                             CASE
                               WHEN article_policy_key IN acc THEN acc
                               ELSE acc + [article_policy_key]
                             END),
                    n.embedding = CASE
                      WHEN $embedding IS NULL THEN n.embedding ELSE $embedding
                    END,
                    n.embedding_input_hash = CASE
                      WHEN $embedding_input_hash IS NULL THEN n.embedding_input_hash
                      ELSE $embedding_input_hash
                    END,
                    n.embedding_profile_revision = CASE
                      WHEN $embedding IS NULL THEN n.embedding_profile_revision ELSE revision
                    END,
	                    n.embedding_model = CASE
	                      WHEN $embedding IS NULL THEN n.embedding_model ELSE $model
	                    END,
	                    n.embedding_updated_at = CASE
	                      WHEN $embedding IS NULL THEN n.embedding_updated_at ELSE datetime()
	                    END,
	                    n.description_curation_history =
	                      coalesce(n.description_curation_history, []) + [$history_event]
                RETURN revision AS revision
                """,
                entity_id=entity_id,
                description=description,
                evidence_ids_considered=evidence_ids_considered,
                used_evidence_ids=used_evidence_ids,
                article_ids_considered=article_ids_considered,
                article_policy_keys=profile_article_policy_keys(
                    article_ids_considered,
                    profile_curation_policy_hash,
                ),
                profile_curation_policy_hash=profile_curation_policy_hash,
                model=model,
                review_decision=review.get("decision"),
                review_reason=review.get("reason"),
                confidence=curation.get("confidence") or review.get("confidence"),
                embedding=embedding,
                embedding_input_hash=embedding_input_hash,
                trace_id=trace_id,
                mlflow_trace_url=mlflow_trace_url,
                mlflow_experiment_id=mlflow_experiment_id,
                history_event=_profile_history_event(
                    status="updated",
                    review=review,
                    model=model,
                    trace_id=trace_id,
                    mlflow_trace_url=mlflow_trace_url,
                    mlflow_experiment_id=mlflow_experiment_id,
                    confidence=curation.get("confidence") or review.get("confidence"),
                ),
            )
            record = await result.single()
            revision = int(record["revision"]) if record else 1
            await _mark_profile_evidence_considered(
                session,
                entity_id=entity_id,
                evidence_ids=evidence_ids_considered,
                revision=revision,
                decision=str(review.get("decision") or "update_profile"),
                review=review,
                profile_curation_policy_hash=profile_curation_policy_hash,
            )
            await _mark_profile_evidence_used(
                session,
                entity_id=entity_id,
                evidence_ids=used_evidence_ids,
                revision=revision,
                curation=curation,
            )

    async def flag_profile_for_review(
        self,
        *,
        entity_id: str,
        evidence_ids: list[str],
        article_ids: list[str],
        profile_curation_policy_hash: str,
        review: dict[str, Any],
        model: str,
        trace_id: str | None = None,
        mlflow_trace_url: str | None = None,
        mlflow_experiment_id: str | None = None,
    ) -> None:
        async with self.neo4j.session() as session:
            await session.run(
                """
                MATCH (n {id: $entity_id})
                SET n.updated_at = datetime(),
                    n.description_curation_status = "needs_human_review",
                    n.description_reviewed_at = datetime(),
                    n.description_model = $model,
                    n.description_review_decision = $decision,
                    n.description_review_reason = $reason,
                    n.description_confidence = $confidence,
                    n.description_curation_policy_hash = $profile_curation_policy_hash,
                    n.description_trace_id = CASE
                      WHEN $trace_id IS NULL OR $trace_id = "" THEN n.description_trace_id
                      ELSE $trace_id
                    END,
                    n.description_trace_url = CASE
                      WHEN $mlflow_trace_url IS NULL OR $mlflow_trace_url = "" THEN n.description_trace_url
                      ELSE $mlflow_trace_url
                    END,
                    n.description_mlflow_experiment_id = CASE
                      WHEN $mlflow_experiment_id IS NULL OR $mlflow_experiment_id = ""
                      THEN n.description_mlflow_experiment_id
                      ELSE $mlflow_experiment_id
                    END,
                    n.description_traced_at = CASE
                      WHEN $trace_id IS NULL OR $trace_id = "" THEN n.description_traced_at
                      ELSE datetime()
                    END,
                    n.description_considered_evidence_ids =
                      reduce(acc = coalesce(n.description_considered_evidence_ids, []),
                             evidence_id IN $evidence_ids |
                             CASE WHEN evidence_id IN acc THEN acc ELSE acc + [evidence_id] END),
	                    n.description_considered_article_ids =
	                      reduce(acc = coalesce(n.description_considered_article_ids, []),
	                             article_id IN $article_ids |
	                             CASE WHEN article_id IN acc THEN acc ELSE acc + [article_id] END),
	                    n.description_considered_article_policy_keys =
	                      reduce(acc = coalesce(n.description_considered_article_policy_keys, []),
	                             article_policy_key IN $article_policy_keys |
	                             CASE
	                               WHEN article_policy_key IN acc THEN acc
	                               ELSE acc + [article_policy_key]
	                             END),
	                    n.description_curation_history =
	                      coalesce(n.description_curation_history, []) + [$history_event]
                """,
                entity_id=entity_id,
                evidence_ids=evidence_ids,
                article_ids=article_ids,
                article_policy_keys=profile_article_policy_keys(
                    article_ids,
                    profile_curation_policy_hash,
                ),
                profile_curation_policy_hash=profile_curation_policy_hash,
                model=model,
                decision=review.get("decision"),
                reason=review.get("reason"),
                confidence=review.get("confidence"),
                trace_id=trace_id,
                mlflow_trace_url=mlflow_trace_url,
                mlflow_experiment_id=mlflow_experiment_id,
                history_event=_profile_history_event(
                    status="needs_human_review",
                    review=review,
                    model=model,
                    trace_id=trace_id,
                    mlflow_trace_url=mlflow_trace_url,
                    mlflow_experiment_id=mlflow_experiment_id,
                ),
            )
            await _mark_profile_evidence_considered(
                session,
                entity_id=entity_id,
                evidence_ids=evidence_ids,
                revision=None,
                decision=str(review.get("decision") or "needs_human_review"),
                review=review,
                profile_curation_policy_hash=profile_curation_policy_hash,
            )

    async def save_profile_embedding(
        self,
        *,
        entity_id: str,
        embedding: list[float],
        embedding_input_hash: str,
        model: str,
    ) -> None:
        query = """
        MATCH (n {id: $entity_id})
        SET n.embedding = $embedding,
            n.embedding_input_hash = $embedding_input_hash,
            n.embedding_profile_revision = coalesce(n.description_profile_revision, 0),
            n.embedding_model = $model,
            n.embedding_updated_at = datetime()
        """
        async with self.neo4j.session() as session:
            await session.run(
                query,
                entity_id=entity_id,
                embedding=embedding,
                embedding_input_hash=embedding_input_hash,
                model=model,
            )


async def upsert_profile_evidence_tx(
    tx,
    *,
    entity_id: str,
    kind: str,
    text: str | None,
    article: ArticleIn,
    article_id: str,
    job_run_id: str | None,
    processed_at: str,
    evidence_status: EvidenceStatus,
    relationship_type: str | None = None,
    other_entity_id: str | None = None,
    other_entity_name: str | None = None,
    other_entity_type: str | None = None,
) -> None:
    evidence_text = (text or "").strip()
    if not evidence_text:
        return
    evidence_text = evidence_text[:PROFILE_EVIDENCE_MAX_CHARS]
    evidence_id = profile_evidence_id(
        entity_id,
        kind,
        relationship_type or "",
        other_entity_id or "",
        evidence_text,
    )
    query = """
    MATCH (n {id: $entity_id})
    MERGE (e:ProfileEvidence {id: $evidence_id})
    ON CREATE SET e.created_at = datetime(),
                  e.entity_id = $entity_id,
                  e.kind = $kind,
                  e.text = $text,
                  e.relationship_type = $relationship_type,
                  e.other_entity_id = $other_entity_id,
                  e.other_entity_name = $other_entity_name,
                  e.other_entity_type = $other_entity_type,
                  e.article_id = $article_id,
                  e.article_url = $article_url,
                  e.article_title = $article_title,
                  e.source_name = $source_name,
                  e.published_at = $published_at,
                  e.job_run_id = $job_run_id,
                  e.processed_at = $processed_at,
                  e.evidence_status = $evidence_status,
                  e.profile_candidate = true,
                  e.profile_candidate_reason = $kind
    SET e.last_seen_at = datetime(),
        e.last_job_run_id = $job_run_id,
        e.last_article_id = $article_id,
        e.article_urls = CASE
          WHEN $article_url IS NULL OR $article_url = "" THEN coalesce(e.article_urls, [])
          WHEN $article_url IN coalesce(e.article_urls, []) THEN e.article_urls
          ELSE coalesce(e.article_urls, []) + [$article_url]
        END,
        e.article_titles = CASE
          WHEN $article_title IS NULL OR $article_title = "" THEN coalesce(e.article_titles, [])
          WHEN $article_title IN coalesce(e.article_titles, []) THEN e.article_titles
          ELSE coalesce(e.article_titles, []) + [$article_title]
        END,
        e.job_run_ids = CASE
          WHEN $job_run_id IS NULL OR $job_run_id = "" THEN coalesce(e.job_run_ids, [])
          WHEN $job_run_id IN coalesce(e.job_run_ids, []) THEN e.job_run_ids
          ELSE coalesce(e.job_run_ids, []) + [$job_run_id]
        END
    MERGE (n)-[:HAS_PROFILE_EVIDENCE]->(e)
    """
    await tx.run(
        query,
        entity_id=entity_id,
        evidence_id=evidence_id,
        kind=kind,
        text=evidence_text,
        relationship_type=relationship_type,
        other_entity_id=other_entity_id,
        other_entity_name=other_entity_name,
        other_entity_type=other_entity_type,
        article_id=article_id,
        article_url=article.url,
        article_title=article.title,
        source_name=article.source_name,
        published_at=article.published_at,
        job_run_id=job_run_id,
        processed_at=processed_at,
        evidence_status=evidence_status,
    )


async def _mark_profile_evidence_considered(
    runner,
    *,
    entity_id: str,
    evidence_ids: list[str],
    revision: int | None,
    decision: str,
    review: dict[str, Any],
    profile_curation_policy_hash: str,
) -> None:
    if not evidence_ids:
        return
    await runner.run(
        """
        MATCH (n {id: $entity_id})-[:HAS_PROFILE_EVIDENCE]->(e:ProfileEvidence)
        WHERE e.id IN $evidence_ids
        WITH n, e, coalesce($revision, coalesce(n.description_profile_revision, 0)) AS revision
        SET e.considered_profile_revision = revision,
            e.considered_profile_policy_hash = $profile_curation_policy_hash,
            e.considered_at = datetime(),
            e.review_decision = $decision,
            e.review_json = $review_json
        """,
        entity_id=entity_id,
        evidence_ids=evidence_ids,
        revision=revision,
        decision=decision,
        review_json=json.dumps(review, ensure_ascii=False, separators=(",", ":"), default=str),
        profile_curation_policy_hash=profile_curation_policy_hash,
    )


def _profile_history_event(
    *,
    status: str,
    review: dict[str, Any],
    model: str,
    trace_id: str | None,
    mlflow_trace_url: str | None,
    mlflow_experiment_id: str | None,
    confidence: Any | None = None,
) -> str:
    return json.dumps(
        {
            "status": status,
            "decision": review.get("decision"),
            "confidence": confidence or review.get("confidence"),
            "model": model,
            "trace_id": trace_id,
            "mlflow_trace_url": mlflow_trace_url,
            "mlflow_experiment_id": mlflow_experiment_id,
            "checked_at": datetime.now(UTC).isoformat(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


async def _mark_profile_evidence_used(
    runner,
    *,
    entity_id: str,
    evidence_ids: list[str],
    revision: int,
    curation: dict[str, Any],
) -> None:
    if not evidence_ids:
        return
    await runner.run(
        """
        MATCH (n {id: $entity_id})-[:HAS_PROFILE_EVIDENCE]->(e:ProfileEvidence)
        WHERE e.id IN $evidence_ids
        SET e.used_profile_revision = $revision,
            e.used_at = datetime(),
            e.curation_json = $curation_json
        """,
        entity_id=entity_id,
        evidence_ids=evidence_ids,
        revision=revision,
        curation_json=json.dumps(curation, ensure_ascii=False, separators=(",", ":"), default=str),
    )
