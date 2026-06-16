from typing import Any

from neo4j.exceptions import Neo4jError
from neo4j.graph import Node

from app.core.logging import get_logger
from app.db.neo4j import Neo4jClient
from app.graph.common import (
    ARTICLE_FEED_RELATIONSHIPS,
    LANDSCAPE_RELATIONSHIPS,
    PROFILE_ENTITY_LABELS,
    graph_response,
    is_geography_topic_node,
    public_node_properties,
)
from app.models.extraction import GraphResponse, SearchResult
from app.services.entity_resolution import NameNormalizer


class GraphReadStore:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j
        self.logger = get_logger("graph_read")

    async def entity_counts(self) -> dict[str, int]:
        query = """
        MATCH (n)
        WITH labels(n) AS node_labels
        UNWIND $entity_labels AS label
        WITH label, node_labels
        WHERE label IN node_labels
        RETURN label, count(*) AS count
        """
        counts = {label: 0 for label in PROFILE_ENTITY_LABELS}
        async with self.neo4j.session() as session:
            result = await session.run(query, entity_labels=PROFILE_ENTITY_LABELS)
            async for record in result:
                label = str(record["label"])
                if label in counts:
                    counts[label] = int(record["count"])
        return counts

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
                        if not is_geography_topic_node(record["node"])
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
                if not is_geography_topic_node(record["node"])
            ]

    async def entity_profile(self, label: str, name: str) -> dict[str, Any] | None:
        label = _safe_read_label(label)
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
            if is_geography_topic_node(node):
                return None
            return {
                "id": node.get("id"),
                "name": node.get("name"),
                "type": list(node.labels)[0],
                "properties": public_node_properties(dict(node)),
                "related": [item for item in record["related"] if item.get("id")],
            }

    async def graph(
        self, entity: str | None = None, limit: int = 120, view: str = "landscape"
    ) -> GraphResponse:
        if entity:
            return await self._entity_graph(entity, limit)
        if view == "feed":
            return await self._article_feed_graph(limit)
        return await self._entity_landscape(limit)

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
            return graph_response(record["nodes"], record["rels"])

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
                landscape_relationships=LANDSCAPE_RELATIONSHIPS,
                mention_relationship="MENTIONS",
                source_relationship="FROM_SOURCE",
                topic_relationship="HAS_TOPIC",
            )
            record = await result.single()
            if not record:
                return GraphResponse(nodes=[], edges=[])
            return graph_response(record["nodes"], record["rels"])

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
                article_feed_relationships=ARTICLE_FEED_RELATIONSHIPS,
            )
            record = await result.single()
            if not record:
                return GraphResponse(nodes=[], edges=[])
            return graph_response(record["nodes"], record["rels"])


def _lucene_query(query: str) -> str:
    terms = [term for term in NameNormalizer.key(query, "Startup").split() if term]
    if not terms:
        return query
    return " AND ".join(f"{term}*" for term in terms)


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


def _safe_read_label(label: str) -> str:
    if label not in {"Startup", "Investor", "Company", "Person", "Topic"}:
        raise ValueError(f"Unsupported label: {label}")
    return label
