import pytest

from app.core.config import Settings
from app.models.extraction import ExtractedEntity
from app.services.entity_resolution import EntityResolver


class FakeRelationshipGraph:
    def __init__(self) -> None:
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
        return True


def resolver_settings() -> Settings:
    return Settings(enable_embedding_resolution=False)


@pytest.mark.asyncio
async def test_topics_cannot_enter_general_fuzzy_or_embedding_resolution() -> None:
    resolver = EntityResolver(resolver_settings())

    with pytest.raises(ValueError, match="closed topic ontology"):
        await resolver.resolve(
            "Topic",
            ExtractedEntity(name="AI Automation Platform", evidence_status="stated"),
        )


@pytest.mark.asyncio
async def test_organization_identity_is_stable_across_roles_suffixes_and_order() -> None:
    company_first = EntityResolver(resolver_settings())
    company = await company_first.resolve(
        "Company", ExtractedEntity(name="SensusQ GmbH", evidence_status="stated")
    )
    promoted = await company_first.resolve(
        "Startup", ExtractedEntity(name="SensusQ", evidence_status="stated")
    )
    investor = await company_first.resolve(
        "Investor", ExtractedEntity(name="SensusQ", evidence_status="stated")
    )

    assert promoted.method == "exact"
    assert promoted.type_promoted is True
    assert promoted.entity.id == company.entity.id == investor.entity.id
    assert investor.entity.label == "Startup"
    assert set(investor.entity.observed_types) == {"Startup", "Investor", "Company"}

    startup_first = EntityResolver(resolver_settings())
    startup = await startup_first.resolve(
        "Startup", ExtractedEntity(name="SensusQ", evidence_status="stated")
    )
    reused = await startup_first.resolve(
        "Company", ExtractedEntity(name="SensusQ", evidence_status="stated")
    )

    assert reused.method == "exact"
    assert reused.type_promoted is False
    assert reused.entity.id == startup.entity.id
    assert reused.entity.label == "Startup"


@pytest.mark.asyncio
async def test_explicit_entity_types_override_contextual_and_fallback_roles() -> None:
    contextual_resolver = EntityResolver(resolver_settings())
    company = await contextual_resolver.resolve(
        "Company",
        ExtractedEntity(
            name="Soravia",
            type_basis="explicit",
            evidence_status="stated",
        ),
    )
    investor_role = await contextual_resolver.resolve(
        "Investor",
        ExtractedEntity(
            name="Soravia",
            type_basis="contextual",
            evidence_status="stated",
        ),
    )

    assert investor_role.method == "exact"
    assert investor_role.entity.id == company.entity.id
    assert investor_role.entity.label == "Company"
    assert investor_role.entity.primary_type_basis == "explicit"
    assert set(investor_role.entity.observed_types) == {"Investor", "Company"}
    assert investor_role.entity.type_conflict is False

    fallback_resolver = EntityResolver(resolver_settings())
    fallback = await fallback_resolver.resolve(
        "Company",
        ExtractedEntity(
            name="Nordkap",
            type_basis="fallback",
            evidence_status="stated",
        ),
    )
    explicit = await fallback_resolver.resolve(
        "Investor",
        ExtractedEntity(
            name="Nordkap",
            type_basis="explicit",
            evidence_status="stated",
        ),
    )

    assert explicit.entity.id == fallback.entity.id
    assert explicit.entity.label == "Investor"
    assert explicit.entity.primary_type_basis == "explicit"
    assert explicit.type_promoted is True

    conflict_resolver = EntityResolver(resolver_settings())
    investor = await conflict_resolver.resolve(
        "Investor",
        ExtractedEntity(
            name="Dual Identity",
            type_basis="explicit",
            evidence_status="stated",
        ),
    )
    startup = await conflict_resolver.resolve(
        "Startup",
        ExtractedEntity(
            name="Dual Identity",
            type_basis="explicit",
            evidence_status="stated",
        ),
    )

    assert startup.entity.id == investor.entity.id
    assert startup.entity.label == "Startup"
    assert startup.entity.type_conflict is True


@pytest.mark.asyncio
async def test_existing_graph_relationship_neighbor_prevents_name_merge() -> None:
    graph = FakeRelationshipGraph()
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
