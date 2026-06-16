from typing import Any

from app.db.neo4j import Neo4jClient
from app.graph.common import jsonable


class InsightStore:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

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
            return [jsonable(dict(record)) async for record in result]

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
            return [jsonable(dict(record)) async for record in result]

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
            return [jsonable(dict(record)) async for record in result]

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
            return [jsonable(dict(record)) async for record in result]
