import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from neo4j.exceptions import Neo4jError
from neo4j.graph import Node, Relationship

from app.core.logging import get_logger
from app.db.neo4j import Neo4jClient
from app.db.schema import build_schema_statements
from app.models.extraction import (
    ADMITTED_EVIDENCE_STATUSES,
    ArticleIn,
    EntityType,
    EvidenceStatus,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    GraphEdge,
    GraphNode,
    GraphResponse,
    NormalizedEntity,
    SearchResult,
)
from app.services.entity_resolution import NameNormalizer


LABELS = {
    "Startup": "Startup",
    "Investor": "Investor",
    "Person": "Person",
    "Topic": "Topic",
    "Company": "Company",
    "Article": "Article",
    "Source": "Source",
}

RELATIONSHIPS = {
    "INVESTED_IN",
    "FOUNDED_BY",
    "EMPLOYED_BY",
    "PARTNERED_WITH",
    "MERGED_WITH",
    "HAS_TOPIC",
    "FROM_SOURCE",
    "ACQUIRED",
    "MENTIONS",
}

_LANDSCAPE_RELATIONSHIPS = [
    "INVESTED_IN",
    "FOUNDED_BY",
    "EMPLOYED_BY",
    "PARTNERED_WITH",
    "MERGED_WITH",
    "ACQUIRED",
    "HAS_TOPIC",
]
_ARTICLE_FEED_RELATIONSHIPS = ["HAS_TOPIC", "FROM_SOURCE", "MENTIONS"]
_DOMAIN_RELATIONSHIPS = [
    "INVESTED_IN",
    "FOUNDED_BY",
    "EMPLOYED_BY",
    "PARTNERED_WITH",
    "MERGED_WITH",
    "ACQUIRED",
    "HAS_TOPIC",
]
_DIRECTIONAL_CONFLICT_RELATIONSHIPS = [
    "INVESTED_IN",
    "FOUNDED_BY",
    "EMPLOYED_BY",
    "ACQUIRED",
]
_EVIDENCE_MAX_CHARS = 1200

_INVERSE_CONFLICT_QUERY = """
MATCH (left_source)-[left]->(left_target)
MATCH (left_target)-[right]->(left_source)
WHERE type(left) IN $directional_relationships
  AND type(right) = type(left)
  AND elementId(left) < elementId(right)
  AND coalesce(left.lifecycle_status, "supported") = "supported"
  AND coalesce(right.lifecycle_status, "supported") = "supported"
  AND coalesce(left.review_status, "unreviewed") <> "rejected"
  AND coalesce(right.review_status, "unreviewed") <> "rejected"
SET left.review_status = "needs_review",
    right.review_status = "needs_review",
    left.review_reasons = CASE
      WHEN "inverse_direction" IN coalesce(left.review_reasons, []) THEN left.review_reasons
      ELSE coalesce(left.review_reasons, []) + ["inverse_direction"]
    END,
    right.review_reasons = CASE
      WHEN "inverse_direction" IN coalesce(right.review_reasons, []) THEN right.review_reasons
      ELSE coalesce(right.review_reasons, []) + ["inverse_direction"]
    END
RETURN count(left) AS conflicts
"""

_TRANSACTION_CONFLICT_QUERY = """
MATCH (first)-[acquired:ACQUIRED]->(second)
MATCH (merge_source)-[merged:MERGED_WITH]->(merge_target)
WHERE (
    (first = merge_source AND second = merge_target)
    OR (first = merge_target AND second = merge_source)
  )
  AND coalesce(acquired.lifecycle_status, "supported") = "supported"
  AND coalesce(merged.lifecycle_status, "supported") = "supported"
  AND coalesce(acquired.review_status, "unreviewed") <> "rejected"
  AND coalesce(merged.review_status, "unreviewed") <> "rejected"
SET acquired.review_status = "needs_review",
    merged.review_status = "needs_review",
    acquired.review_reasons = CASE
      WHEN "competing_transaction_type" IN coalesce(acquired.review_reasons, [])
      THEN acquired.review_reasons
      ELSE coalesce(acquired.review_reasons, []) + ["competing_transaction_type"]
    END,
    merged.review_reasons = CASE
      WHEN "competing_transaction_type" IN coalesce(merged.review_reasons, [])
      THEN merged.review_reasons
      ELSE coalesce(merged.review_reasons, []) + ["competing_transaction_type"]
    END
RETURN count(acquired) AS conflicts
"""


class GraphStore:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j
        self.logger = get_logger("graph_repository")

    async def apply_schema(self, embedding_provider: str = "openai") -> None:
        async with self.neo4j.session() as session:
            for statement in build_schema_statements(embedding_provider):
                await session.run(statement)
        deleted = await self.delete_geography_topics()
        if deleted:
            self.logger.info(
                "geography_topics_deleted",
                extra={"event": "schema", "workflow_step": "apply", "count": deleted},
            )
        cleared = await self.clear_topic_categories()
        if cleared:
            self.logger.info(
                "topic_categories_cleared",
                extra={"event": "schema", "workflow_step": "apply", "count": cleared},
            )
        initialized = await self.initialize_claim_state()
        conflicts = await self.refresh_claim_conflicts()
        self.logger.info("schema_applied", extra={"event": "schema", "workflow_step": "apply"})
        if initialized or conflicts:
            self.logger.info(
                "claim_state_initialized",
                extra={
                    "event": "schema",
                    "workflow_step": "claim_state",
                    "count": initialized,
                    "conflicts": conflicts,
                },
            )

    async def initialize_claim_state(self) -> int:
        query = """
        MATCH (source)-[r]->(target)
        WHERE type(r) IN $claim_relationships
          AND NOT "Article" IN labels(source)
          AND NOT "Article" IN labels(target)
        SET r.active_article_urls = coalesce(r.active_article_urls, r.article_urls, []),
            r.lifecycle_status = coalesce(r.lifecycle_status, "supported"),
            r.review_status = coalesce(r.review_status, "unreviewed"),
            r.review_reasons = coalesce(r.review_reasons, []),
            r.support_changed = coalesce(r.support_changed, false)
        RETURN count(r) AS initialized
        """
        async with self.neo4j.session() as session:
            result = await session.run(query, claim_relationships=_DOMAIN_RELATIONSHIPS)
            record = await result.single()
            return int(record["initialized"]) if record else 0

    async def refresh_claim_conflicts(self) -> int:
        async with self.neo4j.session() as session:
            inverse = await session.run(
                _INVERSE_CONFLICT_QUERY,
                directional_relationships=_DIRECTIONAL_CONFLICT_RELATIONSHIPS,
            )
            inverse_record = await inverse.single()
            transactions = await session.run(_TRANSACTION_CONFLICT_QUERY)
            transaction_record = await transactions.single()
        return int(inverse_record["conflicts"] if inverse_record else 0) + int(
            transaction_record["conflicts"] if transaction_record else 0
        )

    async def delete_geography_topics(self) -> int:
        query = """
        MATCH (topic:Topic)
        WHERE toLower(coalesce(topic.category, "")) = "geography"
        WITH collect(topic) AS topics
        WITH topics, size(topics) AS deleted
        UNWIND topics AS topic
        DETACH DELETE topic
        RETURN deleted
        """
        async with self.neo4j.session() as session:
            result = await session.run(query)
            record = await result.single()
            return int(record["deleted"]) if record else 0

    async def clear_topic_categories(self) -> int:
        query = """
        MATCH (topic:Topic)
        WHERE topic.category IS NOT NULL
        WITH topic
        REMOVE topic.category
        RETURN count(topic) AS cleared
        """
        async with self.neo4j.session() as session:
            result = await session.run(query)
            record = await result.single()
            return int(record["cleared"]) if record else 0

    async def clear_all(self) -> int:
        """Delete every node and relationship in the database.

        Returns the number of nodes deleted. Use before a clean re-ingest run.
        """
        async with self.neo4j.session() as session:
            count_result = await session.run("MATCH (n) RETURN count(n) AS total")
            record = await count_result.single()
            deleted = int(record["total"]) if record else 0
            await session.run("MATCH (n) DETACH DELETE n")
        self.logger.info(
            "graph_cleared",
            extra={"event": "admin", "workflow_step": "clear_all", "deleted_nodes": deleted},
        )
        return deleted

    async def list_entities(self, label: str) -> list[dict[str, Any]]:
        label = _safe_label(label)
        query = f"""
        MATCH (n:{label})
        RETURN n.id AS id,
               n.name AS name,
               n.canonical_name AS canonical_name,
               n.aliases AS aliases,
               n.evidence_status AS evidence_status,
               n.description AS description,
               n.descriptions AS descriptions,
               n.embedding AS embedding
        LIMIT 50000
        """
        async with self.neo4j.session() as session:
            result = await session.run(query)
            return [_jsonable(dict(record)) async for record in result]

    async def has_relationship_to_named_entity(
        self,
        target_id: str,
        candidate_label: str,
        candidate_names: list[str],
    ) -> bool:
        label = _safe_label(candidate_label)
        name_variants = {
            variant for name in candidate_names for variant in _name_match_variants(name) if variant
        }
        key_variants = {key for name in candidate_names if (key := NameNormalizer.key(name, label))}
        candidate_ids = {
            f"{label.lower()}:{NameNormalizer.slug(name)}"
            for name in candidate_names
            if NameNormalizer.display(name)
        }
        if not name_variants and not key_variants and not candidate_ids:
            return False

        query = f"""
        MATCH (target {{id: $target_id}})
        MATCH (candidate:{label})
        WHERE coalesce(candidate.id, "") <> $target_id
        WITH target, candidate,
             [raw IN [coalesce(candidate.name, ""), coalesce(candidate.canonical_name, "")]
               + coalesce(candidate.aliases, []) | replace(toLower(raw), "-", " ")] AS names
        WHERE coalesce(candidate.id, "") IN $candidate_ids
           OR any(name IN names WHERE name IN $name_variants OR name IN $key_variants)
           OR any(name IN names WHERE any(key IN $key_variants
                WHERE size(key) >= 4 AND name STARTS WITH key + " "))
        MATCH (target)-[rel]-(candidate)
        WHERE type(rel) IN $relationships
          AND coalesce(rel.lifecycle_status, "supported") = "supported"
          AND coalesce(rel.review_status, "unreviewed") <> "rejected"
        RETURN count(rel) > 0 AS blocked
        LIMIT 1
        """
        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                target_id=target_id,
                candidate_ids=list(candidate_ids),
                name_variants=list(name_variants),
                key_variants=list(key_variants),
                relationships=list(_DOMAIN_RELATIONSHIPS),
            )
            record = await result.single()
            return bool(record and record["blocked"])

    async def save_entity_embedding(self, entity_id: str, embedding: list[float]) -> None:
        query = "MATCH (n {id: $id}) SET n.embedding = $embedding"
        async with self.neo4j.session() as session:
            await session.run(query, id=entity_id, embedding=embedding)

    async def find_nearest_entity(
        self,
        label: str,
        embedding: list[float],
        min_score: float,
        k: int = 5,
        exclude_ids: set[str] | None = None,
    ) -> tuple[str | None, float]:
        """Return (entity_id, score) of the closest entity via Neo4j vector index, or (None, 0)."""
        index_name = f"{label.lower()}_embedding"
        query = """
        CALL db.index.vector.queryNodes($index_name, $k, $embedding)
        YIELD node, score
        WHERE score >= $min_score
          AND NOT coalesce(node.id, "") IN $exclude_ids
        RETURN node.id AS id, score
        ORDER BY score DESC
        LIMIT 1
        """
        async with self.neo4j.session() as session:
            try:
                result = await session.run(
                    query,
                    index_name=index_name,
                    k=k,
                    embedding=embedding,
                    min_score=min_score,
                    exclude_ids=list(exclude_ids or set()),
                )
                record = await result.single()
                if record:
                    return record["id"], float(record["score"])
            except Exception:
                pass
        return None, 0.0

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
                self._ingest_article_tx,
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

    async def search(self, query: str, limit: int = 15) -> list[SearchResult]:
        if not query.strip():
            return []
        lucene = _lucene_query(query)
        fulltext_statement = """
        CALL db.index.fulltext.queryNodes("entitySearch", $search_query) YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        try:
            async with self.neo4j.session() as session:
                result = await session.run(
                    fulltext_statement,
                    search_query=lucene,
                    limit=limit,
                )
                rows = [record async for record in result]
                if rows:
                    results = [
                        _search_result(record["node"], record["score"])
                        for record in rows
                        if not _is_geography_topic_node(record["node"])
                    ]
                    if results:
                        return results
        except Neo4jError as exc:
            self.logger.warning(
                "fulltext_search_failed",
                extra={"event": "search", "workflow_step": "fulltext", "error": str(exc)},
            )

        partial_query = """
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN ["Startup", "Investor", "Company", "Person", "Topic"])
          AND (
            toLower(coalesce(n.name, "")) CONTAINS $needle OR
            toLower(coalesce(n.canonical_name, "")) CONTAINS $needle OR
            any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) CONTAINS $needle)
          )
        RETURN n AS node, 1.0 AS score
        LIMIT $limit
        """
        async with self.neo4j.session() as session:
            result = await session.run(partial_query, needle=query.casefold(), limit=limit)
            return [
                _search_result(record["node"], record["score"])
                async for record in result
                if not _is_geography_topic_node(record["node"])
            ]

    async def entity_profile(self, label: str, name: str) -> dict[str, Any] | None:
        label = _safe_label(label)
        needle = NameNormalizer.key(name, label)
        query = f"""
        MATCH (n:{label})
        WHERE (
            toLower(coalesce(n.name, "")) CONTAINS $needle
            OR toLower(coalesce(n.canonical_name, "")) CONTAINS $needle
            OR any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) CONTAINS $needle)
        )
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n AS node,
               collect(DISTINCT {{
                 id: m.id,
                 name: coalesce(m.name, m.title),
                 type: head(labels(m)),
                 relationship: type(r)
               }})[0..50] AS related
        LIMIT 1
        """
        async with self.neo4j.session() as session:
            result = await session.run(query, needle=needle)
            record = await result.single()
            if not record:
                return None
            node = record["node"]
            if _is_geography_topic_node(node):
                return None
            return {
                "id": node.get("id"),
                "name": node.get("name"),
                "type": list(node.labels)[0],
                "properties": _public_node_properties(dict(node)),
                "related": [item for item in record["related"] if item.get("id")],
            }

    async def node_claims(self, node_id: str) -> dict[str, Any] | None:
        node_query = "MATCH (node {id: $node_id}) RETURN node LIMIT 1"
        claim_query = """
        MATCH (source)-[r]->(target)
        WHERE (source.id = $node_id OR target.id = $node_id)
          AND type(r) IN $relationships
        RETURN source, r, target
        """
        async with self.neo4j.session() as session:
            node_result = await session.run(node_query, node_id=node_id)
            node_record = await node_result.single()
            if not node_record:
                return None
            result = await session.run(
                claim_query,
                node_id=node_id,
                relationships=list(RELATIONSHIPS),
            )
            rows = [record async for record in result]

        claims: list[dict[str, Any]] = []
        mentions: list[dict[str, Any]] = []
        for record in rows:
            source = record["source"]
            relationship = record["r"]
            target = record["target"]
            rel_type = relationship.type
            source_id = source.get("id") or source.element_id
            target_id = target.get("id") or target.element_id
            properties = _public_relationship_properties(dict(relationship))
            payload = {
                "edge_id": relationship.element_id,
                "relationship": rel_type,
                "direction": (
                    "undirected"
                    if rel_type == "MERGED_WITH"
                    else "outgoing"
                    if source_id == node_id
                    else "incoming"
                ),
                "counterparty": _node_reference(target if source_id == node_id else source),
                "lifecycle_status": properties.get("lifecycle_status", "supported"),
                "review_status": properties.get("review_status", "unreviewed"),
                "review_reasons": properties.get("review_reasons", []),
                "review_comment": properties.get("review_comment"),
                "reviewed_by": properties.get("reviewed_by"),
                "reviewed_at": properties.get("reviewed_at"),
                "review_history": _review_history(properties.get("review_history")),
                "support_changed": properties.get("support_changed", False),
                "active_support_count": len(
                    properties.get("active_article_urls") or properties.get("article_urls") or []
                ),
                "assertions": _claim_assertions(properties.get("provenance")),
                "source_id": source_id,
                "target_id": target_id,
            }
            source_is_article = "Article" in source.labels
            target_is_article = "Article" in target.labels
            if (
                rel_type in _DOMAIN_RELATIONSHIPS
                and not source_is_article
                and not target_is_article
            ):
                claims.append(payload)
            elif rel_type in {"MENTIONS", "HAS_TOPIC"}:
                mentions.append(payload)

        claims.sort(key=_claim_sort_key)
        mentions.sort(key=_claim_sort_key)
        return {
            "node_id": node_id,
            "claims": claims,
            "mentions": mentions,
        }

    async def review_claim(
        self,
        *,
        source_id: str,
        relationship: str,
        target_id: str,
        decision: str,
        comment: str | None,
        reviewer: str | None,
    ) -> bool:
        review_event = json.dumps(
            {
                "decision": decision,
                "comment": comment,
                "reviewer": reviewer,
                "reviewed_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        query = """
        MATCH (source {id: $source_id})-[r]->(target {id: $target_id})
        WHERE type(r) = $relationship
        SET r.review_status = $decision,
            r.review_reasons = CASE
              WHEN $decision IN ["accepted", "rejected"] THEN []
              ELSE coalesce(r.review_reasons, [])
            END,
            r.review_comment = $comment,
            r.reviewed_by = $reviewer,
            r.reviewed_at = datetime(),
            r.review_history = coalesce(r.review_history, []) + [$review_event]
        RETURN count(r) AS reviewed
        """
        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                source_id=source_id,
                relationship=relationship,
                target_id=target_id,
                decision=decision,
                comment=comment,
                reviewer=reviewer,
                review_event=review_event,
            )
            record = await result.single()
            return bool(record and record["reviewed"])

    async def graph(
        self, entity: str | None = None, limit: int = 120, view: str = "landscape"
    ) -> GraphResponse:
        if entity:
            return await self._entity_graph(entity, limit)
        if view == "feed":
            return await self._article_feed_graph(limit)
        return await self._entity_landscape(limit)

    async def trending_startups(self, days: int = 30, limit: int = 10) -> list[dict[str, Any]]:
        query = """
        MATCH (s:Startup)<-[r:MENTIONS]-(a:Article)
        WHERE coalesce(r.evidence_status, "stated") IN ["stated", "attributed"]
        WITH s, a, coalesce(a.published_at, a.updated_at, s.updated_at) AS seen_at
        WHERE seen_at IS NOT NULL
          AND seen_at >= datetime() - duration({days: $days})
        RETURN s.id AS id,
               s.name AS name,
               count(DISTINCT a) AS mentions,
               collect(DISTINCT coalesce(a.title, a.url))[0..5] AS articles,
               max(seen_at) AS last_seen
        ORDER BY mentions DESC, last_seen DESC, name ASC
        LIMIT $limit
        """
        async with self.neo4j.session() as session:
            result = await session.run(query, days=days, limit=limit)
            return [_jsonable(dict(record)) async for record in result]

    async def top_investors(self, limit: int = 10) -> list[dict[str, Any]]:
        query = """
        MATCH (i)-[r:INVESTED_IN]->(target)
        WHERE (i:Investor OR i:Company OR i:Person)
          AND target:Startup
          AND coalesce(r.lifecycle_status, "supported") = "supported"
          AND coalesce(r.review_status, "unreviewed") IN ["unreviewed", "accepted"]
        RETURN i.id AS id,
               i.name AS name,
               CASE
                 WHEN i:Investor THEN "Investor"
                 WHEN i:Company THEN "Company"
                 WHEN i:Person THEN "Person"
                 ELSE head(labels(i))
               END AS type,
               count(DISTINCT target) AS investments,
               collect(DISTINCT coalesce(target.name, target.stage))[0..5] AS examples
        ORDER BY investments DESC, name ASC
        LIMIT $limit
        """
        async with self.neo4j.session() as session:
            result = await session.run(query, limit=limit)
            return [_jsonable(dict(record)) async for record in result]

    async def co_investments(self, limit: int = 20) -> list[dict[str, Any]]:
        query = """
        MATCH (i1)-[r1:INVESTED_IN]->(s:Startup)<-[r2:INVESTED_IN]-(i2)
        WHERE (i1:Investor OR i1:Company OR i1:Person)
          AND (i2:Investor OR i2:Company OR i2:Person)
          AND i1.id < i2.id
          AND coalesce(r1.lifecycle_status, "supported") = "supported"
          AND coalesce(r2.lifecycle_status, "supported") = "supported"
          AND coalesce(r1.review_status, "unreviewed") IN ["unreviewed", "accepted"]
          AND coalesce(r2.review_status, "unreviewed") IN ["unreviewed", "accepted"]
        RETURN i1.id AS source_id,
               i1.name AS source,
               CASE
                 WHEN i1:Investor THEN "Investor"
                 WHEN i1:Company THEN "Company"
                 WHEN i1:Person THEN "Person"
                 ELSE head(labels(i1))
               END AS source_type,
               i2.id AS target_id,
               i2.name AS target,
               CASE
                 WHEN i2:Investor THEN "Investor"
                 WHEN i2:Company THEN "Company"
                 WHEN i2:Person THEN "Person"
                 ELSE head(labels(i2))
               END AS target_type,
               count(DISTINCT s) AS shared_startups,
               count(DISTINCT s) AS rounds,
               collect(DISTINCT s.name)[0..5] AS examples
        ORDER BY shared_startups DESC, source ASC, target ASC
        LIMIT $limit
        """
        async with self.neo4j.session() as session:
            result = await session.run(query, limit=limit)
            return [_jsonable(dict(record)) async for record in result]

    async def topic_clusters(self, limit: int = 20) -> list[dict[str, Any]]:
        query = """
        MATCH (topic:Topic)<-[r:HAS_TOPIC]-(entity)
        WHERE any(label IN labels(entity) WHERE label IN ["Startup", "Investor", "Company", "Person", "Article"])
          AND coalesce(r.lifecycle_status, "supported") = "supported"
          AND coalesce(r.review_status, "unreviewed") IN ["unreviewed", "accepted"]
        RETURN topic.id AS id,
               topic.name AS name,
               count(DISTINCT entity) AS entity_count,
               collect(DISTINCT coalesce(entity.name, entity.title))[0..6] AS examples
        ORDER BY entity_count DESC, name ASC
        LIMIT $limit
        """
        async with self.neo4j.session() as session:
            result = await session.run(query, limit=limit)
            return [_jsonable(dict(record)) async for record in result]

    async def _entity_graph(self, entity: str, limit: int) -> GraphResponse:
        needle = NameNormalizer.key(entity, "Startup")
        query = """
        MATCH (center)
        WHERE any(label IN labels(center) WHERE label IN ["Startup", "Investor", "Company", "Person", "Topic"])
          AND (
            toLower(coalesce(center.name, "")) CONTAINS $needle OR
            toLower(coalesce(center.canonical_name, "")) CONTAINS $needle OR
            any(alias IN coalesce(center.aliases, []) WHERE toLower(alias) CONTAINS $needle)
          )
                WITH center,
                         CASE
                             WHEN toLower(coalesce(center.name, "")) = $needle THEN 4
                             WHEN toLower(coalesce(center.canonical_name, "")) = $needle THEN 3
                             WHEN any(alias IN coalesce(center.aliases, []) WHERE toLower(alias) = $needle) THEN 2
                             ELSE 1
                         END AS relevance,
                         COUNT { (center)--() } AS degree
                ORDER BY relevance DESC,
                                 degree DESC,
                                 toLower(coalesce(center.name, center.canonical_name, center.id, "")) ASC
                LIMIT 1
                OPTIONAL MATCH p=(center)-[*1..2]-(neighbor)
                WITH center, neighbor, min(length(p)) AS hops
                ORDER BY hops ASC,
                                 toLower(coalesce(neighbor.name, neighbor.title, neighbor.id, "")) ASC
                WITH center, collect(DISTINCT neighbor)[0..$limit] AS neighbors
                WITH [center] + [node IN neighbors WHERE node IS NOT NULL] AS nodes
                OPTIONAL MATCH (a)-[rel]-(b)
                WHERE a IN nodes AND b IN nodes
                RETURN nodes, collect(DISTINCT rel) AS rels
        """
        async with self.neo4j.session() as session:
            result = await session.run(query, needle=needle, limit=limit)
            record = await result.single()
            if not record:
                return GraphResponse(nodes=[], edges=[])
            return _graph_response(record["nodes"], record["rels"])

    async def _entity_landscape(self, limit: int) -> GraphResponse:
        """Return the most-connected startups with article and topic provenance."""
        query = """
        MATCH (s:Startup)
        OPTIONAL MATCH (s)-[deg_r]-()
        WHERE type(deg_r) IN $landscape_relationships
        OPTIONAL MATCH (article_degree:Article)-[mention_degree]->(s)
        WHERE type(mention_degree) = $mention_relationship
        WITH s, count(DISTINCT deg_r) + count(DISTINCT mention_degree) AS degree
        ORDER BY degree DESC, s.name ASC
        LIMIT $limit
        OPTIONAL MATCH (s)-[core_rel]-(partner)
        WHERE type(core_rel) IN $landscape_relationships
        OPTIONAL MATCH (article:Article)-[mention_rel]->(s)
        WHERE type(mention_rel) = $mention_relationship
        OPTIONAL MATCH (article)-[source_rel]->(source:Source)
        WHERE type(source_rel) = $source_relationship
        OPTIONAL MATCH (article)-[article_topic_rel]->(article_topic:Topic)
        WHERE type(article_topic_rel) = $topic_relationship
        WITH collect(DISTINCT s) AS startups,
             collect(DISTINCT partner) AS partners,
             collect(DISTINCT article) AS articles,
             collect(DISTINCT source) AS sources,
             collect(DISTINCT article_topic) AS article_topics,
             collect(DISTINCT core_rel) AS core_rels,
             collect(DISTINCT mention_rel) AS mention_rels,
             collect(DISTINCT source_rel) AS source_rels,
             collect(DISTINCT article_topic_rel) AS article_topic_rels
        RETURN startups
               + [x IN partners WHERE x IS NOT NULL]
               + [x IN articles WHERE x IS NOT NULL]
               + [x IN sources WHERE x IS NOT NULL]
               + [x IN article_topics WHERE x IS NOT NULL] AS nodes,
               [x IN core_rels WHERE x IS NOT NULL]
               + [x IN mention_rels WHERE x IS NOT NULL]
               + [x IN source_rels WHERE x IS NOT NULL]
               + [x IN article_topic_rels WHERE x IS NOT NULL] AS rels
        """
        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                limit=limit,
                landscape_relationships=_LANDSCAPE_RELATIONSHIPS,
                mention_relationship="MENTIONS",
                source_relationship="FROM_SOURCE",
                topic_relationship="HAS_TOPIC",
            )
            record = await result.single()
            if not record:
                return GraphResponse(nodes=[], edges=[])
            return _graph_response(record["nodes"], record["rels"])

    async def _article_feed_graph(self, limit: int) -> GraphResponse:
        query = """
        MATCH (a:Article)-[r]->(n)
        WHERE type(r) IN $article_feed_relationships
        WITH a, r, n
        ORDER BY a.published_at DESC
        LIMIT $limit
        RETURN collect(DISTINCT a) + collect(DISTINCT n) AS nodes,
               collect(DISTINCT r) AS rels
        """
        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                limit=limit,
                article_feed_relationships=_ARTICLE_FEED_RELATIONSHIPS,
            )
            record = await result.single()
            if not record:
                return GraphResponse(nodes=[], edges=[])
            return _graph_response(record["nodes"], record["rels"])

    @staticmethod
    async def _ingest_article_tx(
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
        article_id = f"article:{_sha1(article.url)}"
        processed_at = processed_at or datetime.now(UTC).isoformat()
        article_provenance = _provenance_json(
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
            raw_extracted_entities=_raw_extracted_entities_json(raw_extracted_entities, extraction),
        )
        await _relate_tx(
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
            await _upsert_entity_tx(
                tx,
                entity,
                descriptions=entity.descriptions or None,
                embedding=entity.embedding,
            )
            op_count += 1

        for entity_type, extracted_entities in _extracted_entity_groups(extraction):
            for extracted_entity in extracted_entities:
                if extracted_entity.evidence_status not in ADMITTED_EVIDENCE_STATUSES:
                    continue
                resolved_entity = _lookup(resolved_entities, entity_type, extracted_entity.name)
                if resolved_entity is None:
                    continue
                if entity_type == "Topic":
                    rel_type = "HAS_TOPIC"
                else:
                    rel_type = "MENTIONS"
                await _relate_tx(
                    tx,
                    article_id,
                    resolved_entity.id,
                    rel_type,
                    evidence=_entity_article_evidence(extracted_entity),
                    evidence_status=extracted_entity.evidence_status,
                    article_url=article.url,
                    article_title=article.title,
                    provenance=_provenance_json(
                        article=article,
                        trace_id=trace_id,
                        mlflow_trace_url=mlflow_trace_url,
                        mlflow_experiment_id=mlflow_experiment_id,
                        article_id=article_id,
                        job_run_id=job_run_id,
                        processed_at=processed_at,
                        event="asserted",
                        evidence_status=extracted_entity.evidence_status,
                        evidence=_entity_article_evidence(extracted_entity),
                    ),
                )
                op_count += 1

        emitted_claims: list[dict[str, str]] = []
        touched_entity_ids: set[str] = set()
        for rel in extraction.relationships:
            if (
                rel.type not in RELATIONSHIPS
                or rel.evidence_status not in ADMITTED_EVIDENCE_STATUSES
                or not _has_valid_claim_direction(rel)
            ):
                continue
            source = _lookup(resolved_entities, rel.source_type, rel.source_name)
            target = _lookup(resolved_entities, rel.target_type, rel.target_name)
            if source and target:
                source_id, target_id = _claim_endpoint_ids(rel.type, source.id, target.id)
                if source_id == target_id:
                    continue
                await _relate_tx(
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
                    provenance=_provenance_json(
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

        await _mark_not_reproduced_claims_tx(
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
            await _mark_claim_conflicts_tx(tx)

        return op_count


def _provenance_json(
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


async def _upsert_entity_tx(
    tx,
    entity: NormalizedEntity,
    descriptions: list[str] | None = None,
    embedding: list[float] | None = None,
) -> None:
    label = _safe_label(entity.label)
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
          WHEN $description IS NULL OR $description = "" THEN n.description
          WHEN n.description IS NULL OR n.description = "" OR n.evidence_status IS NULL THEN $description
          WHEN $evidence_status = "stated" OR $evidence_status = n.evidence_status THEN $description
          ELSE n.description
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


async def _relate_tx(
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


async def _mark_not_reproduced_claims_tx(
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
    direction_changed_provenance = _provenance_json(
        article=article,
        article_id=article_id,
        job_run_id=job_run_id,
        processed_at=processed_at,
        event="direction_changed",
        trace_id=trace_id,
        mlflow_trace_url=mlflow_trace_url,
        mlflow_experiment_id=mlflow_experiment_id,
    )
    not_reproduced_provenance = _provenance_json(
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
        directional_relationships=_DIRECTIONAL_CONFLICT_RELATIONSHIPS,
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
        claim_relationships=_DOMAIN_RELATIONSHIPS,
        directional_relationships=_DIRECTIONAL_CONFLICT_RELATIONSHIPS,
        article_url=article.url,
        emitted_claims=emitted_claims,
        provenance=not_reproduced_provenance,
    )


async def _mark_claim_conflicts_tx(tx) -> None:
    await tx.run(
        _INVERSE_CONFLICT_QUERY,
        directional_relationships=_DIRECTIONAL_CONFLICT_RELATIONSHIPS,
    )
    await tx.run(_TRANSACTION_CONFLICT_QUERY)


def _has_valid_claim_direction(relationship: ExtractedRelationship) -> bool:
    source_type = relationship.source_type
    target_type = relationship.target_type
    if relationship.type == "INVESTED_IN":
        return source_type in {"Investor", "Company", "Person"} and target_type == "Startup"
    if relationship.type == "FOUNDED_BY":
        return source_type == "Startup" and target_type == "Person"
    if relationship.type == "EMPLOYED_BY":
        return source_type == "Person" and target_type in {"Startup", "Company"}
    if relationship.type == "ACQUIRED":
        return source_type in {"Startup", "Company"} and target_type in {"Startup", "Company"}
    if relationship.type == "HAS_TOPIC":
        return source_type != "Topic" and target_type == "Topic"
    return True


def _claim_endpoint_ids(relationship_type: str, source_id: str, target_id: str) -> tuple[str, str]:
    if relationship_type == "MERGED_WITH" and target_id < source_id:
        return target_id, source_id
    return source_id, target_id


def _extracted_entity_groups(
    extraction: ExtractionResult,
) -> tuple[tuple[EntityType, list[ExtractedEntity]], ...]:
    return (
        ("Startup", extraction.startups),
        ("Investor", extraction.investors),
        ("Person", extraction.people),
        ("Topic", extraction.topics),
        ("Company", extraction.companies),
    )


def _entity_article_evidence(entity: ExtractedEntity) -> str | None:
    evidence = (entity.source.evidence or entity.description or "").strip()
    if not evidence:
        return None
    return evidence[:_EVIDENCE_MAX_CHARS]


def _lookup(
    resolved_entities: dict[tuple[str, str], NormalizedEntity],
    label: str,
    raw_name: str,
) -> NormalizedEntity | None:
    return resolved_entities.get((label, NameNormalizer.key(raw_name, label)))


def _safe_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Unsupported label: {label}")
    return LABELS[label]


def _name_match_variants(name: str) -> set[str]:
    display = NameNormalizer.display(name)
    lowered = display.casefold()
    hyphen_normalized = lowered.replace("-", " ")
    return {
        lowered,
        hyphen_normalized,
        " ".join(hyphen_normalized.split()),
    }


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _lucene_query(query: str) -> str:
    terms = [term for term in NameNormalizer.key(query, "Startup").split() if term]
    if not terms:
        return query
    return " AND ".join(f"{term}*" for term in terms)


def _raw_extracted_entities_json(
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


def _search_result(node: Node, score: float) -> SearchResult:
    labels = [
        label
        for label in node.labels
        if label in {"Startup", "Investor", "Company", "Person", "Topic"}
    ]
    return SearchResult(
        id=node.get("id"),
        name=node.get("name") or node.get("canonical_name"),
        type=labels[0] if labels else "Entity",
        score=float(score),
        aliases=node.get("aliases") or [],
        description=node.get("description"),
    )


def _node_reference(node: Node) -> dict[str, Any]:
    labels = list(node.labels)
    return {
        "id": node.get("id") or node.element_id,
        "label": node.get("name") or node.get("title") or node.get("id"),
        "type": labels[0] if labels else "Node",
    }


def _claim_assertions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    assertions: list[dict[str, Any]] = []
    for serialized in value:
        if not isinstance(serialized, str):
            continue
        try:
            record = json.loads(serialized)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        record["event"] = record.get("event") or "asserted"
        assertions.append(record)
    assertions.sort(
        key=lambda record: str(record.get("processed_at") or record.get("published_at") or ""),
        reverse=True,
    )
    return assertions


def _review_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    events: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            try:
                record = json.loads(item)
            except json.JSONDecodeError:
                continue
        elif isinstance(item, dict):
            record = item
        else:
            continue
        if isinstance(record, dict):
            events.append(_jsonable(record))
    return events


def _claim_sort_key(claim: dict[str, Any]) -> tuple[int, str, str]:
    review_rank = 0 if claim["review_status"] == "needs_review" else 1
    counterpart = claim["counterparty"]
    return review_rank, str(claim["relationship"]), str(counterpart["label"])


def _is_geography_topic_node(node: Node) -> bool:
    return "Topic" in node.labels and _is_geography_category(node.get("category"))


def _is_geography_category(value: Any) -> bool:
    return str(value or "").casefold() == "geography"


def _graph_response(nodes: list[Node], rels: list[Relationship]) -> GraphResponse:
    node_models: dict[str, GraphNode] = {}
    for node in nodes:
        if not node:
            continue
        if _is_geography_topic_node(node):
            continue
        node_id = node.get("id") or node.element_id
        labels = list(node.labels)
        node_type = labels[0] if labels else "Node"
        title = node.get("name") or node.get("title") or node_id
        properties = _public_node_properties(dict(node))
        if node_type == "Topic":
            properties.pop("category", None)
        node_models[node_id] = GraphNode(
            id=node_id,
            label=title,
            type=node_type,
            properties=properties,
        )

    edge_models: dict[str, GraphEdge] = {}
    for rel in rels:
        if not rel:
            continue
        if rel.get("review_status") == "rejected":
            continue
        source_id = rel.start_node.get("id") or rel.start_node.element_id
        target_id = rel.end_node.get("id") or rel.end_node.element_id
        if source_id not in node_models or target_id not in node_models:
            continue
        edge_id = rel.element_id
        edge_models[edge_id] = GraphEdge(
            id=edge_id,
            source=source_id,
            target=target_id,
            label=rel.type,
            properties=_public_relationship_properties(dict(rel)),
        )
    return GraphResponse(nodes=list(node_models.values()), edges=list(edge_models.values()))


def _public_node_properties(properties: dict[str, Any]) -> dict[str, Any]:
    result = _jsonable(properties)
    result.pop("embedding", None)
    result.pop("confidence", None)
    return result


def _public_relationship_properties(properties: dict[str, Any]) -> dict[str, Any]:
    result = _jsonable(properties)
    result.pop("confidence", None)
    return result


def _jsonable(properties: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in properties.items():
        result[key] = _jsonable_value(value)
    return result


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if hasattr(value, "to_native"):
        native = value.to_native()
        if isinstance(native, datetime):
            return native.astimezone(UTC).isoformat()
        return native.isoformat() if hasattr(native, "isoformat") else native
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    return value
