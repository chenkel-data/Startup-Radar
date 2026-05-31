import json

import pytest

from app.graph.graph_store import (
    GraphStore,
    _claim_endpoint_ids,
    _has_valid_claim_direction,
    _provenance_json,
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


def claim(
    relationship_type: str,
    source_name: str,
    source_type: str,
    target_name: str,
    target_type: str,
) -> ExtractedRelationship:
    return ExtractedRelationship(
        type=relationship_type,  # type: ignore[arg-type]
        source_name=source_name,
        source_type=source_type,  # type: ignore[arg-type]
        target_name=target_name,
        target_type=target_type,  # type: ignore[arg-type]
        evidence_status="stated",
        keywords="Test",
        evidence=f"{source_name} -> {target_name}",
    )


def test_claim_direction_validation_enforces_typed_relationship_endpoints() -> None:
    assert _has_valid_claim_direction(
        claim("INVESTED_IN", "UVC Partners", "Investor", "Aleph Alpha", "Startup")
    )
    assert not _has_valid_claim_direction(
        claim("INVESTED_IN", "Aleph Alpha", "Startup", "UVC Partners", "Investor")
    )
    assert _has_valid_claim_direction(
        claim("FOUNDED_BY", "ViViRA", "Startup", "Philip Heimann", "Person")
    )
    assert not _has_valid_claim_direction(
        claim("FOUNDED_BY", "Philip Heimann", "Person", "ViViRA", "Startup")
    )
    assert _has_valid_claim_direction(claim("ACQUIRED", "SAP", "Company", "Prior Labs", "Startup"))
    assert not _has_valid_claim_direction(
        claim("ACQUIRED", "SAP", "Company", "UVC Partners", "Investor")
    )


def test_claim_endpoint_ids_canonicalize_only_undirected_mergers() -> None:
    assert _claim_endpoint_ids("MERGED_WITH", "startup:zeta", "startup:alpha") == (
        "startup:alpha",
        "startup:zeta",
    )
    assert _claim_endpoint_ids("ACQUIRED", "company:sap", "startup:prior-labs") == (
        "company:sap",
        "startup:prior-labs",
    )


def test_provenance_json_keeps_trace_article_and_evidence_context() -> None:
    provenance = json.loads(
        _provenance_json(
            article=article(),
            article_id="article:123",
            trace_id="trace-123",
            mlflow_trace_url="http://mlflow/traces/trace-123",
            mlflow_experiment_id="7",
            job_run_id="job-123",
            processed_at="2026-05-31T16:35:22+02:00",
            event="asserted",
            evidence_status="stated",
            evidence="SAP kauft das junge KI-Startup Prior Labs.",
        )
    )

    assert provenance["event"] == "asserted"
    assert provenance["article_id"] == "article:123"
    assert provenance["article_title"] == "SAP kauft Prior Labs"
    assert provenance["source_name"] == "deutsche-startups.de"
    assert provenance["trace_id"] == "trace-123"
    assert provenance["mlflow_trace_url"] == "http://mlflow/traces/trace-123"
    assert provenance["evidence_status"] == "stated"
    assert provenance["evidence"] == "SAP kauft das junge KI-Startup Prior Labs."


@pytest.mark.asyncio
async def test_ingest_article_bundle_writes_acquisition_with_extracted_direction() -> None:
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

    await GraphStore._ingest_article_tx(
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
    query, write = acquisition_writes[0]
    assert write["source_id"] == sap.id
    assert write["target_id"] == prior_labs.id
    assert write["evidence_status"] == "stated"
    assert write["evidence"] == "SAP kauft das junge KI-Startup Prior Labs."
    assert write["keywords"] == "Uebernahme"
    assert write["article_url"] == "https://example.test/articles/sap-prior-labs"
    assert write["tracks_support"] is True
    assert "r.review_status" in query
    assert 'r.review_status IN ["accepted", "rejected"]' in query
    assert "r.active_article_urls" in query
    assert "r.lifecycle_status" in query

    provenance = json.loads(str(write["provenance"]))
    assert provenance["event"] == "asserted"
    assert provenance["trace_id"] == "trace-123"
    assert provenance["evidence"] == "SAP kauft das junge KI-Startup Prior Labs."


@pytest.mark.asyncio
async def test_ingest_article_bundle_skips_invalid_founder_direction() -> None:
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

    await GraphStore._ingest_article_tx(
        tx,
        article(),
        extraction,
        resolved_mapping(startup, person),
        raw_extracted_entities=None,
    )

    assert not any("MERGE (source)-[r:FOUNDED_BY]->(target)" in query for query, _ in tx.runs)
