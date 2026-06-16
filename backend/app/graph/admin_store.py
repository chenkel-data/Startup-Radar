from app.core.logging import get_logger
from app.db.neo4j import Neo4jClient
from app.db.schema import build_schema_statements
from app.graph.claim_store import ClaimStore


class AdminStore:
    def __init__(self, neo4j: Neo4jClient, claim_store: ClaimStore):
        self.neo4j = neo4j
        self.claim_store = claim_store
        self.logger = get_logger("graph_admin")

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
        initialized = await self.claim_store.initialize_claim_state()
        conflicts = await self.claim_store.refresh_claim_conflicts()
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
        """Delete every node and relationship in the database for a clean re-ingest run."""
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
