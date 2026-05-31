from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.core.config import Settings


class Neo4jClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def verify(self) -> None:
        await self.driver.verify_connectivity()

    @asynccontextmanager
    async def session(self) -> AsyncIterator:
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            yield session

    async def close(self) -> None:
        await self.driver.close()
