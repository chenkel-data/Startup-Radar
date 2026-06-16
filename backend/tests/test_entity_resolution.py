import pytest

from app.core.config import Settings
from app.models.extraction import ExtractedEntity
from app.services.entity_resolution import EntityResolver, NameNormalizer


class FakeRelationshipGraph:
    def __init__(self, *, blocks_merge: bool) -> None:
        self.blocks_merge = blocks_merge
        self.calls: list[dict[str, object]] = []

    async def has_relationship_to_named_entity(
        self,
        *,
        target_id: str,
        candidate_label: str,
        candidate_names: list[str],
    ) -> bool:
        self.calls.append(
            {
                "target_id": target_id,
                "candidate_label": candidate_label,
                "candidate_names": candidate_names,
            }
        )
        return self.blocks_merge


def resolver_settings() -> Settings:
    return Settings(enable_embedding_resolution=False)


@pytest.mark.asyncio
async def test_current_article_relationship_endpoints_are_not_merged_by_name() -> None:
    graph = FakeRelationshipGraph(blocks_merge=False)
    resolver = EntityResolver(resolver_settings(), graph=graph)
    await resolver.resolve("Startup", ExtractedEntity(name="Acme", evidence_status="stated"))

    outcome = await resolver.resolve(
        "Startup",
        ExtractedEntity(name="Acme GmbH", evidence_status="stated"),
        blocked_canonical_keys={NameNormalizer.key("Acme", "Startup")},
    )

    assert outcome.method == "new"
    assert outcome.entity.canonical_name == "Acme GmbH"
    assert graph.calls == []


@pytest.mark.asyncio
async def test_existing_graph_relationship_neighbor_prevents_name_merge() -> None:
    graph = FakeRelationshipGraph(blocks_merge=True)
    resolver = EntityResolver(resolver_settings(), graph=graph)
    await resolver.resolve("Startup", ExtractedEntity(name="Acme", evidence_status="stated"))

    outcome = await resolver.resolve(
        "Startup",
        ExtractedEntity(name="Acme GmbH", evidence_status="stated"),
    )

    assert outcome.method == "new"
    assert outcome.entity.canonical_name == "Acme GmbH"
    assert graph.calls[0] == {
        "target_id": "startup:acme",
        "candidate_label": "Startup",
        "candidate_names": ["Acme GmbH"],
    }
