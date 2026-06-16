import json
from datetime import UTC, datetime
from typing import Any

from app.db.neo4j import Neo4jClient
from app.graph.claim_store import (
    claim_endpoint_ids,
    has_valid_claim_direction,
    mark_claim_conflicts_tx,
)
from app.graph.common import (
    DIRECTIONAL_CONFLICT_RELATIONSHIPS,
    DOMAIN_RELATIONSHIPS,
    EVIDENCE_MAX_CHARS,
    RELATIONSHIPS,
    safe_label,
    sha1,
)
from app.graph.entity_profile_store import upsert_profile_evidence_tx
from app.models.extraction import (
    ADMITTED_EVIDENCE_STATUSES,
    ArticleIn,
    EntityType,
    EvidenceStatus,
    ExtractedEntity,
    ExtractionResult,
    NormalizedEntity,
)
from app.services.entity_resolution import NameNormalizer


class ArticleGraphWriter:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

    async def ingest_article_bundle(
        self,
        article: ArticleIn,
        extraction: ExtractionResult,
        resolved_entities: dict[tuple[str, str], NormalizedEntity],
        *,
        raw_extracted_entities: dict[str, Any] | None = None,
        trace_id: str | None = None,
        mlflow_trace_url: str | None = None,
        mlflow_experiment_id: str | None = None,
        job_run_id: str | None = None,
        processed_at: str | None = None,
    ) -> int:
        async with self.neo4j.session() as session:
            return await session.execute_write(
                ingest_article_tx,
                article,
                extraction,
                resolved_entities,
                raw_extracted_entities,
                trace_id,
                mlflow_trace_url,
                mlflow_experiment_id,
                job_run_id,
                processed_at,
            )


async def ingest_article_tx(
    tx,
    article: ArticleIn,
    extraction: ExtractionResult,
    resolved_entities: dict[tuple[str, str], NormalizedEntity],
    raw_extracted_entities: dict[str, Any] | None,
    trace_id: str | None = None,
    mlflow_trace_url: str | None = None,
    mlflow_experiment_id: str | None = None,
    job_run_id: str | None = None,
    processed_at: str | None = None,
) -> int:
    op_count = 0
    source_id = f"source:{NameNormalizer.slug(article.source_name)}"
    article_id = f"article:{sha1(article.url)}"
    processed_at = processed_at or datetime.now(UTC).isoformat()
    article_provenance = provenance_json(
        article=article,
        trace_id=trace_id,
        mlflow_trace_url=mlflow_trace_url,
        mlflow_experiment_id=mlflow_experiment_id,
        article_id=article_id,
        job_run_id=job_run_id,
        processed_at=processed_at,
        event="processed",
    )

    await tx.run(
        """
        MERGE (s:Source {id: $id})
        ON CREATE SET s.created_at = datetime()
        SET s.updated_at = datetime(),
            s.name = $name,
            s.url = $url
        """,
        id=source_id,
        name=article.source_name,
        url=article.source_url or article.url,
    )
    await tx.run(
        """
        MERGE (a:Article {id: $id})
        ON CREATE SET a.created_at = datetime()
        SET a.updated_at = datetime(),
            a.url = $url,
            a.title = $title,
            a.summary = $summary,
            a.text = $text,
            a.author = $author,
            a.source_name = $source_name,
            a.source_url = $source_url,
            a.published_at = $published_at,
            a.tags = $tags,
            a.trace_id = CASE
              WHEN $trace_id IS NULL OR $trace_id = "" THEN a.trace_id ELSE $trace_id
            END,
            a.mlflow_trace_url = CASE
              WHEN $mlflow_trace_url IS NULL OR $mlflow_trace_url = "" THEN a.mlflow_trace_url ELSE $mlflow_trace_url
            END,
            a.mlflow_experiment_id = CASE
              WHEN $mlflow_experiment_id IS NULL OR $mlflow_experiment_id = "" THEN a.mlflow_experiment_id ELSE $mlflow_experiment_id
            END,
            a.trace_provenance = CASE
              WHEN $provenance IN coalesce(a.trace_provenance, []) THEN a.trace_provenance
              ELSE coalesce(a.trace_provenance, []) + [$provenance]
            END,
            a.raw_extracted_entities = $raw_extracted_entities
        """,
        id=article_id,
        url=article.url,
        title=article.title,
        summary=article.summary,
        text=article.text[:20000],
        author=article.author,
        source_name=article.source_name,
        source_url=article.source_url,
        published_at=article.published_at,
        tags=article.tags,
        trace_id=trace_id,
        mlflow_trace_url=mlflow_trace_url,
        mlflow_experiment_id=mlflow_experiment_id,
        provenance=article_provenance,
        raw_extracted_entities=raw_extracted_entities_json(raw_extracted_entities, extraction),
    )
    await relate_tx(
        tx,
        article_id,
        source_id,
        "FROM_SOURCE",
        evidence=None,
        evidence_status="stated",
        provenance=article_provenance,
    )
    op_count += 3

    unique_entities = {entity.id: entity for entity in resolved_entities.values()}
    for entity in unique_entities.values():
        if entity.evidence_status not in ADMITTED_EVIDENCE_STATUSES:
            continue
        await upsert_entity_tx(
            tx,
            entity,
            descriptions=entity.descriptions or None,
            embedding=entity.embedding,
        )
        op_count += 1

    for entity_type, extracted_entities in extracted_entity_groups(extraction):
        for extracted_entity in extracted_entities:
            if extracted_entity.evidence_status not in ADMITTED_EVIDENCE_STATUSES:
                continue
            resolved_entity = lookup(resolved_entities, entity_type, extracted_entity.name)
            if resolved_entity is None:
                continue
            entity_evidence = entity_article_evidence(extracted_entity)
            rel_type = "HAS_TOPIC" if entity_type == "Topic" else "MENTIONS"
            await relate_tx(
                tx,
                article_id,
                resolved_entity.id,
                rel_type,
                evidence=entity_evidence,
                evidence_status=extracted_entity.evidence_status,
                article_url=article.url,
                article_title=article.title,
                provenance=provenance_json(
                    article=article,
                    trace_id=trace_id,
                    mlflow_trace_url=mlflow_trace_url,
                    mlflow_experiment_id=mlflow_experiment_id,
                    article_id=article_id,
                    job_run_id=job_run_id,
                    processed_at=processed_at,
                    event="asserted",
                    evidence_status=extracted_entity.evidence_status,
                    evidence=entity_evidence,
                ),
            )
            await upsert_profile_evidence_tx(
                tx,
                entity_id=resolved_entity.id,
                kind="entity_description",
                text=entity_evidence,
                article=article,
                article_id=article_id,
                job_run_id=job_run_id,
                processed_at=processed_at,
                evidence_status=extracted_entity.evidence_status,
            )
            op_count += 1

    emitted_claims: list[dict[str, str]] = []
    touched_entity_ids: set[str] = set()
    for rel in extraction.relationships:
        if (
            rel.type not in RELATIONSHIPS
            or rel.evidence_status not in ADMITTED_EVIDENCE_STATUSES
            or not has_valid_claim_direction(rel)
        ):
            continue
        source = lookup(resolved_entities, rel.source_type, rel.source_name)
        target = lookup(resolved_entities, rel.target_type, rel.target_name)
        if source and target:
            source_id, target_id = claim_endpoint_ids(rel.type, source.id, target.id)
            if source_id == target_id:
                continue
            await relate_tx(
                tx,
                source_id,
                target_id,
                rel.type,
                rel.evidence,
                evidence_status=rel.evidence_status,
                keywords=rel.keywords,
                article_url=article.url,
                article_title=article.title,
                tracks_support=True,
                provenance=provenance_json(
                    article=article,
                    trace_id=trace_id,
                    mlflow_trace_url=mlflow_trace_url,
                    mlflow_experiment_id=mlflow_experiment_id,
                    article_id=article_id,
                    job_run_id=job_run_id,
                    processed_at=processed_at,
                    event="asserted",
                    evidence_status=rel.evidence_status,
                    evidence=rel.evidence,
                ),
            )
            emitted_claims.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relationship": rel.type,
                }
            )
            touched_entity_ids.update((source_id, target_id))
            op_count += 1

    await mark_not_reproduced_claims_tx(
        tx,
        article=article,
        article_id=article_id,
        emitted_claims=emitted_claims,
        trace_id=trace_id,
        mlflow_trace_url=mlflow_trace_url,
        mlflow_experiment_id=mlflow_experiment_id,
        job_run_id=job_run_id,
        processed_at=processed_at,
    )
    if touched_entity_ids:
        await mark_claim_conflicts_tx(tx)

    return op_count


def provenance_json(
    *,
    article: ArticleIn,
    trace_id: str | None,
    mlflow_trace_url: str | None,
    mlflow_experiment_id: str | None,
    article_id: str | None = None,
    job_run_id: str | None = None,
    processed_at: str | None = None,
    event: str | None = None,
    evidence_status: EvidenceStatus | None = None,
    evidence: str | None = None,
) -> str:
    provenance: dict[str, Any] = {
        "event": event,
        "article_id": article_id,
        "article_url": article.url,
        "article_title": article.title,
        "source_name": article.source_name,
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "job_run_id": job_run_id,
        "processed_at": processed_at,
        "trace_id": trace_id,
        "mlflow_trace_url": mlflow_trace_url,
        "mlflow_experiment_id": mlflow_experiment_id,
    }
    if evidence_status is not None:
        provenance["evidence_status"] = evidence_status
    if evidence:
        provenance["evidence"] = evidence
    return json.dumps(provenance, ensure_ascii=False, separators=(",", ":"), default=str)


async def upsert_entity_tx(
    tx,
    entity: NormalizedEntity,
    descriptions: list[str] | None = None,
    embedding: list[float] | None = None,
) -> None:
    label = safe_label(entity.label)
    query = f"""
    MERGE (n:{label} {{id: $id}})
    ON CREATE SET n.created_at = datetime()
    SET n.updated_at = datetime(),
        n.name = $name,
        n.canonical_name = $canonical_name,
        n.aliases = CASE
          WHEN n.aliases IS NULL THEN $aliases
          ELSE reduce(acc = n.aliases, alias IN $aliases |
            CASE WHEN alias IN acc THEN acc ELSE acc + [alias] END)
        END,
        n.evidence_status = CASE
          WHEN n.evidence_status = "stated" OR $evidence_status = "unsure" THEN n.evidence_status
          WHEN $evidence_status = "stated" OR n.evidence_status IS NULL THEN $evidence_status
          ELSE n.evidence_status
        END,
        n.description = CASE
          WHEN n.description_source IN ["curated_llm", "reviewed_llm"] THEN n.description
          WHEN $description IS NULL OR $description = "" THEN n.description
          WHEN n.description IS NULL OR n.description = "" OR n.evidence_status IS NULL THEN $description
          WHEN $evidence_status = "stated" OR $evidence_status = n.evidence_status THEN $description
          ELSE n.description
        END,
        n.description_source = CASE
          WHEN n.description_source IN ["curated_llm", "reviewed_llm"] THEN n.description_source
          WHEN $description IS NULL OR $description = "" THEN n.description_source
          ELSE coalesce(n.description_source, "extraction")
        END,
        n.descriptions = CASE
          WHEN $descriptions IS NULL THEN n.descriptions ELSE $descriptions
        END,
        n.embedding = CASE
          WHEN $embedding IS NULL THEN n.embedding ELSE $embedding
        END
    """
    await tx.run(
        query,
        id=entity.id,
        name=entity.name,
        canonical_name=entity.canonical_name,
        aliases=entity.aliases,
        evidence_status=entity.evidence_status,
        description=entity.description,
        descriptions=descriptions,
        embedding=embedding,
    )


async def relate_tx(
    tx,
    source_id: str,
    target_id: str,
    rel_type: str,
    evidence: str | None,
    evidence_status: EvidenceStatus = "stated",
    keywords: str | None = None,
    article_url: str | None = None,
    article_title: str | None = None,
    provenance: str | None = None,
    tracks_support: bool = False,
) -> None:
    if rel_type not in RELATIONSHIPS:
        raise ValueError(f"Unsupported relationship type: {rel_type}")
    query = f"""
    MATCH (source {{id: $source_id}})
    MATCH (target {{id: $target_id}})
    MERGE (source)-[r:{rel_type}]->(target)
    ON CREATE SET r.created_at = datetime()
    SET r.updated_at = datetime(),
        r.evidence_status = CASE
          WHEN r.evidence_status = "stated" OR $evidence_status = "unsure" THEN r.evidence_status
          WHEN $evidence_status = "stated" OR r.evidence_status IS NULL THEN $evidence_status
          ELSE r.evidence_status
        END,
        r.evidence = CASE
          WHEN $evidence IS NULL OR $evidence = "" THEN r.evidence
          WHEN r.evidence IS NULL OR r.evidence = "" OR r.evidence_status IS NULL THEN $evidence
          WHEN $evidence_status = "stated" OR $evidence_status = r.evidence_status THEN $evidence
          ELSE r.evidence
        END,
        r.keywords = CASE
          WHEN $keywords IS NULL OR $keywords = "" THEN r.keywords ELSE $keywords
        END,
        r.article_urls = CASE
          WHEN $article_url IS NULL OR $article_url = "" THEN coalesce(r.article_urls, [])
          WHEN r.article_urls IS NULL THEN [$article_url]
          WHEN $article_url IN r.article_urls THEN r.article_urls
          ELSE r.article_urls + [$article_url]
        END,
        r.article_titles = CASE
          WHEN $article_title IS NULL OR $article_title = "" THEN coalesce(r.article_titles, [])
          WHEN r.article_titles IS NULL THEN [$article_title]
          WHEN $article_title IN r.article_titles THEN r.article_titles
          ELSE r.article_titles + [$article_title]
        END,
        r.active_article_urls = CASE
          WHEN NOT $tracks_support THEN r.active_article_urls
          WHEN $article_url IS NULL OR $article_url = "" THEN coalesce(r.active_article_urls, [])
          WHEN $article_url IN coalesce(r.active_article_urls, r.article_urls, [])
            THEN coalesce(r.active_article_urls, r.article_urls, [])
          ELSE coalesce(r.active_article_urls, r.article_urls, []) + [$article_url]
        END,
        r.lifecycle_status = CASE
          WHEN $tracks_support THEN "supported" ELSE r.lifecycle_status
        END,
        r.review_status = CASE
          WHEN NOT $tracks_support THEN r.review_status
          WHEN r.review_status IN ["accepted", "rejected"] THEN r.review_status
          WHEN size([
            reason IN coalesce(r.review_reasons, [])
            WHERE reason <> "not_reproduced_same_article"
              AND reason <> "direction_changed_same_article"
          ]) = 0 THEN "unreviewed"
          ELSE coalesce(r.review_status, "needs_review")
        END,
        r.review_reasons = CASE
          WHEN $tracks_support THEN [
            reason IN coalesce(r.review_reasons, [])
            WHERE reason <> "not_reproduced_same_article"
              AND reason <> "direction_changed_same_article"
          ]
          ELSE r.review_reasons
        END,
        r.support_changed = CASE
          WHEN $tracks_support THEN coalesce(r.support_changed, false) ELSE r.support_changed
        END,
        r.provenance = CASE
          WHEN $provenance IS NULL OR $provenance = "" THEN coalesce(r.provenance, [])
          WHEN $provenance IN coalesce(r.provenance, []) THEN r.provenance
          ELSE coalesce(r.provenance, []) + [$provenance]
        END
    """
    await tx.run(
        query,
        source_id=source_id,
        target_id=target_id,
        evidence_status=evidence_status,
        evidence=evidence,
        keywords=keywords,
        article_url=article_url,
        article_title=article_title,
        provenance=provenance,
        tracks_support=tracks_support,
    )


async def mark_not_reproduced_claims_tx(
    tx,
    *,
    article: ArticleIn,
    article_id: str,
    emitted_claims: list[dict[str, str]],
    trace_id: str | None,
    mlflow_trace_url: str | None,
    mlflow_experiment_id: str | None,
    job_run_id: str | None,
    processed_at: str,
) -> None:
    direction_changed_provenance = provenance_json(
        article=article,
        article_id=article_id,
        job_run_id=job_run_id,
        processed_at=processed_at,
        event="direction_changed",
        trace_id=trace_id,
        mlflow_trace_url=mlflow_trace_url,
        mlflow_experiment_id=mlflow_experiment_id,
    )
    not_reproduced_provenance = provenance_json(
        article=article,
        article_id=article_id,
        job_run_id=job_run_id,
        processed_at=processed_at,
        event="not_reproduced",
        trace_id=trace_id,
        mlflow_trace_url=mlflow_trace_url,
        mlflow_experiment_id=mlflow_experiment_id,
    )
    direction_changed_query = """
    MATCH (source)-[r]->(target)
    WHERE type(r) IN $directional_relationships
      AND NOT "Article" IN labels(source)
      AND NOT "Article" IN labels(target)
      AND $article_url IN coalesce(r.active_article_urls, r.article_urls, [])
      AND none(claim IN $emitted_claims WHERE
        claim.source_id = source.id
        AND claim.target_id = target.id
        AND claim.relationship = type(r)
      )
      AND any(claim IN $emitted_claims WHERE
        claim.source_id = target.id
        AND claim.target_id = source.id
        AND claim.relationship = type(r)
      )
    WITH r, [url IN coalesce(r.active_article_urls, r.article_urls, [])
             WHERE url <> $article_url] AS remaining_support
    SET r.active_article_urls = remaining_support,
        r.lifecycle_status = CASE
          WHEN size(remaining_support) = 0
            THEN "unsupported_by_latest_source_processing"
          ELSE "supported"
        END,
        r.support_changed = true,
        r.review_status = CASE
          WHEN size(remaining_support) = 0
               AND coalesce(r.review_status, "unreviewed") <> "rejected"
            THEN "needs_review"
          ELSE coalesce(r.review_status, "unreviewed")
        END,
        r.review_reasons = CASE
          WHEN size(remaining_support) = 0
               AND NOT "direction_changed_same_article" IN coalesce(r.review_reasons, [])
            THEN coalesce(r.review_reasons, []) + ["direction_changed_same_article"]
          ELSE coalesce(r.review_reasons, [])
        END,
        r.provenance = CASE
          WHEN $provenance IN coalesce(r.provenance, []) THEN r.provenance
          ELSE coalesce(r.provenance, []) + [$provenance]
        END
    """
    await tx.run(
        direction_changed_query,
        directional_relationships=DIRECTIONAL_CONFLICT_RELATIONSHIPS,
        article_url=article.url,
        emitted_claims=emitted_claims,
        provenance=direction_changed_provenance,
    )

    not_reproduced_query = """
    MATCH (source)-[r]->(target)
    WHERE type(r) IN $claim_relationships
      AND NOT "Article" IN labels(source)
      AND NOT "Article" IN labels(target)
      AND $article_url IN coalesce(r.active_article_urls, r.article_urls, [])
      AND none(claim IN $emitted_claims WHERE
        claim.source_id = source.id
        AND claim.target_id = target.id
        AND claim.relationship = type(r)
      )
      AND NOT (
        type(r) IN $directional_relationships
        AND any(claim IN $emitted_claims WHERE
          claim.source_id = target.id
          AND claim.target_id = source.id
          AND claim.relationship = type(r)
        )
      )
    WITH r, [url IN coalesce(r.active_article_urls, r.article_urls, [])
             WHERE url <> $article_url] AS remaining_support
    SET r.active_article_urls = remaining_support,
        r.lifecycle_status = CASE
          WHEN size(remaining_support) = 0
            THEN "unsupported_by_latest_source_processing"
          ELSE "supported"
        END,
        r.support_changed = true,
        r.review_status = CASE
          WHEN size(remaining_support) = 0
               AND coalesce(r.review_status, "unreviewed") <> "rejected"
            THEN "needs_review"
          ELSE coalesce(r.review_status, "unreviewed")
        END,
        r.review_reasons = CASE
          WHEN size(remaining_support) = 0
               AND NOT "not_reproduced_same_article" IN coalesce(r.review_reasons, [])
            THEN coalesce(r.review_reasons, []) + ["not_reproduced_same_article"]
          ELSE coalesce(r.review_reasons, [])
        END,
        r.provenance = CASE
          WHEN $provenance IN coalesce(r.provenance, []) THEN r.provenance
          ELSE coalesce(r.provenance, []) + [$provenance]
        END
    """
    await tx.run(
        not_reproduced_query,
        claim_relationships=DOMAIN_RELATIONSHIPS,
        directional_relationships=DIRECTIONAL_CONFLICT_RELATIONSHIPS,
        article_url=article.url,
        emitted_claims=emitted_claims,
        provenance=not_reproduced_provenance,
    )


def extracted_entity_groups(
    extraction: ExtractionResult,
) -> tuple[tuple[EntityType, list[ExtractedEntity]], ...]:
    return (
        ("Startup", extraction.startups),
        ("Investor", extraction.investors),
        ("Person", extraction.people),
        ("Topic", extraction.topics),
        ("Company", extraction.companies),
    )


def entity_article_evidence(entity: ExtractedEntity) -> str | None:
    evidence = (entity.source.evidence or entity.description or "").strip()
    if not evidence:
        return None
    return evidence[:EVIDENCE_MAX_CHARS]


def lookup(
    resolved_entities: dict[tuple[str, str], NormalizedEntity],
    label: str,
    raw_name: str,
) -> NormalizedEntity | None:
    return resolved_entities.get((label, NameNormalizer.key(raw_name, label)))


def raw_extracted_entities_json(
    raw_extracted_entities: dict[str, Any] | None,
    extraction: ExtractionResult | None = None,
) -> str:
    payload = (
        raw_extracted_entities
        or (extraction.raw_model_output if extraction else None)
        or (extraction.model_dump(mode="json") if extraction else {})
    )

    entity_payload = {
        key: payload.get(key, [])
        for key in (
            "startups",
            "investors",
            "people",
            "topics",
            "companies",
            "relationships",
        )
    }
    return json.dumps(entity_payload, ensure_ascii=False, separators=(",", ":"), default=str)
