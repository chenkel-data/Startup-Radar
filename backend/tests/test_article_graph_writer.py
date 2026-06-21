import json

import pytest

from app.graph.article_graph_writer import (
    ingest_article_tx,
)
from app.models.extraction import (
    ArticleIn,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    NormalizedEntity,
)
from app.services.entity_resolution import NameNormalizer


class FakeTx:
    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, object]]] = []

    async def run(self, query: str, **params: object) -> None:
        self.runs.append((query, params))


def article() -> ArticleIn:
    return ArticleIn(
        url="https://example.test/articles/sap-prior-labs",
        title="SAP kauft Prior Labs",
        source_name="deutsche-startups.de",
        text="SAP kauft das junge KI-Startup Prior Labs und baut seine KI-Aktivitaeten aus.",
    )


def resolved(entity_type: str, name: str) -> NormalizedEntity:
    key = NameNormalizer.key(name, entity_type)
    return NormalizedEntity(
        id=f"{entity_type.lower()}:{key.replace(' ', '-')}",
        label=entity_type,  # type: ignore[arg-type]
        canonical_name=name,
        name=name,
        aliases=[],
        evidence_status="stated",
    )


def resolved_mapping(*entities: NormalizedEntity) -> dict[tuple[str, str], NormalizedEntity]:
    return {
        (entity.label, NameNormalizer.key(entity.name, entity.label)): entity for entity in entities
    }


@pytest.mark.asyncio
async def test_ingest_article_tx_writes_acquisition_with_extracted_direction() -> None:
    tx = FakeTx()
    sap = resolved("Company", "SAP")
    prior_labs = resolved("Startup", "Prior Labs")
    extraction = ExtractionResult(
        companies=[ExtractedEntity(name="SAP", evidence_status="stated")],
        startups=[ExtractedEntity(name="Prior Labs", evidence_status="stated")],
        relationships=[
            ExtractedRelationship(
                type="ACQUIRED",
                source_name="SAP",
                source_type="Company",
                target_name="Prior Labs",
                target_type="Startup",
                evidence_status="stated",
                keywords="Uebernahme",
                evidence="SAP kauft das junge KI-Startup Prior Labs.",
            )
        ],
    )

    await ingest_article_tx(
        tx,
        article(),
        extraction,
        resolved_mapping(sap, prior_labs),
        raw_extracted_entities=None,
        trace_id="trace-123",
        mlflow_trace_url="http://mlflow/traces/trace-123",
        mlflow_experiment_id="7",
        job_run_id="job-123",
        processed_at="2026-05-31T16:35:22+02:00",
    )

    acquisition_writes = [
        (query, params)
        for query, params in tx.runs
        if "MERGE (source)-[r:ACQUIRED]->(target)" in query
    ]
    assert len(acquisition_writes) == 1
    _query, write = acquisition_writes[0]
    assert write["source_id"] == sap.id
    assert write["target_id"] == prior_labs.id
    assert write["evidence_status"] == "stated"
    assert write["evidence"] == "SAP kauft das junge KI-Startup Prior Labs."
    assert write["keywords"] == "Uebernahme"
    assert write["article_url"] == "https://example.test/articles/sap-prior-labs"
    assert write["tracks_support"] is True

    provenance = json.loads(str(write["provenance"]))
    assert provenance["event"] == "asserted"
    assert provenance["trace_id"] == "trace-123"
    assert provenance["evidence"] == "SAP kauft das junge KI-Startup Prior Labs."


@pytest.mark.asyncio
async def test_ingest_article_tx_writes_profile_evidence() -> None:
    tx = FakeTx()
    sap = resolved("Company", "SAP")
    prior_labs = resolved("Startup", "Prior Labs")
    extraction = ExtractionResult(
        companies=[
            ExtractedEntity(
                name="SAP",
                evidence_status="stated",
                description="SAP ist ein Softwareunternehmen.",
            )
        ],
        startups=[
            ExtractedEntity(
                name="Prior Labs",
                evidence_status="stated",
                description="Prior Labs ist ein KI-Startup.",
            )
        ],
        relationships=[
            ExtractedRelationship(
                type="ACQUIRED",
                source_name="SAP",
                source_type="Company",
                target_name="Prior Labs",
                target_type="Startup",
                evidence_status="stated",
                keywords="Uebernahme",
                evidence="SAP kauft das junge KI-Startup Prior Labs.",
            )
        ],
    )

    await ingest_article_tx(
        tx,
        article(),
        extraction,
        resolved_mapping(sap, prior_labs),
        raw_extracted_entities=None,
        job_run_id="job-123",
        processed_at="2026-05-31T16:35:22+02:00",
    )

    profile_writes = [params for query, params in tx.runs if "MERGE (e:ProfileEvidence" in query]
    assert [params["kind"] for params in profile_writes].count("entity_description") == 2
    assert [params["kind"] for params in profile_writes].count("relationship_fact") == 0
    assert all(params["job_run_id"] == "job-123" for params in profile_writes)


@pytest.mark.asyncio
async def test_ingest_article_tx_skips_invalid_founder_direction() -> None:
    tx = FakeTx()
    startup = resolved("Startup", "ViViRA")
    person = resolved("Person", "Philip Heimann")
    extraction = ExtractionResult(
        startups=[ExtractedEntity(name="ViViRA", evidence_status="stated")],
        people=[ExtractedEntity(name="Philip Heimann", evidence_status="stated")],
        relationships=[
            ExtractedRelationship(
                type="FOUNDED_BY",
                source_name="Philip Heimann",
                source_type="Person",
                target_name="ViViRA",
                target_type="Startup",
                evidence_status="stated",
                keywords="Gruendung",
                evidence="Philip Heimann ist Mitgruender von ViViRA.",
            )
        ],
    )

    await ingest_article_tx(
        tx,
        article(),
        extraction,
        resolved_mapping(startup, person),
        raw_extracted_entities=None,
    )

    assert not any("MERGE (source)-[r:FOUNDED_BY]->(target)" in query for query, _ in tx.runs)
