import json

import pytest

from app.graph.article_graph_writer import (
    cached_article_urls_for_extraction_policy_tx,
    ingest_article_tx,
)
from app.models.extraction import (
    ArticleCleaningReport,
    ArticleContentRemoval,
    ArticleIn,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    NormalizedEntity,
    SourceAttribution,
)
from app.services.entity_resolution import NameNormalizer


class FakeTx:
    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, object]]] = []

    async def run(self, query: str, **params: object) -> None:
        self.runs.append((query, params))


class FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    async def data(self) -> list[dict[str, object]]:
        return self.rows


class CacheTx:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.runs: list[tuple[str, dict[str, object]]] = []

    async def run(self, query: str, **params: object) -> FakeResult:
        self.runs.append((query, params))
        return FakeResult(self.rows)


def article() -> ArticleIn:
    return ArticleIn(
        url="https://example.test/articles/sap-prior-labs",
        title="SAP kauft Prior Labs",
        source_name="deutsche-startups.de",
        primary_type="#Deals",
        text="SAP kauft das junge KI-Startup Prior Labs und baut seine KI-Aktivitaeten aus.",
        cleaning=ArticleCleaningReport(
            selected_container="#post .wysiwyg",
            blocks_before=3,
            blocks_after=2,
            text_chars_before=120,
            text_chars_after=79,
            removed_blocks=[
                ArticleContentRemoval(
                    reason="image_credit",
                    text_chars=25,
                    text_preview="Foto (oben): Redaktion",
                )
            ],
        ),
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
async def test_redirected_article_is_written_and_cached_by_its_discovered_url() -> None:
    tx = FakeTx()
    canonical_url = "https://example.test/?p=448049"
    listing_url = "https://example.test/2026/07/29/five-new-startups/"
    aliased_article = article().model_copy(
        update={"url": canonical_url, "discovered_url": listing_url}
    )

    await ingest_article_tx(
        tx,
        aliased_article,
        ExtractionResult(),
        {},
        raw_extracted_entities=None,
    )

    article_params = next(params for query, params in tx.runs if "MERGE (a:Article" in query)
    assert article_params["url_aliases"] == [canonical_url, listing_url]

    cache_tx = CacheTx([{"url": listing_url}])

    cached = await cached_article_urls_for_extraction_policy_tx(
        cache_tx,
        [listing_url],
        "policy-1",
    )

    _query, params = cache_tx.runs[0]
    assert params == {
        "urls": [listing_url],
        "article_extraction_policy_hash": "policy-1",
    }
    assert cached == {listing_url}


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
        article_extraction_policy_hash="policy-hash-1",
        article_extraction_policy={
            "cache_version": "article_extraction_cache_v1",
            "model": "gpt-test",
        },
    )

    article_writes = [params for query, params in tx.runs if "MERGE (a:Article" in query]
    assert len(article_writes) == 1
    assert article_writes[0]["text"] == article().text
    assert article_writes[0]["primary_type"] == "#Deals"
    cleaning_report = json.loads(str(article_writes[0]["cleaning_report"]))
    assert cleaning_report["selected_container"] == "#post .wysiwyg"
    assert cleaning_report["removed_blocks"][0]["reason"] == "image_credit"
    assert article_writes[0]["article_extraction_policy_hash"] == "policy-hash-1"
    assert json.loads(str(article_writes[0]["article_extraction_policy"])) == {
        "cache_version": "article_extraction_cache_v1",
        "model": "gpt-test",
    }

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
                source=SourceAttribution(evidence=article().text),
            )
        ],
        startups=[
            ExtractedEntity(
                name="Prior Labs",
                evidence_status="stated",
                description="Prior Labs ist ein KI-Startup.",
                source=SourceAttribution(evidence=article().text),
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
        profile_curation_policy_hash="curation-policy-1",
    )

    profile_writes = [
        (query, params) for query, params in tx.runs if "MERGE (e:ProfileEvidence" in query
    ]
    assert [params["kind"] for _query, params in profile_writes].count("entity_description") == 2
    assert [params["kind"] for _query, params in profile_writes].count("relationship_fact") == 0
    assert all(params["text"] == article().text for _query, params in profile_writes)
    assert all(params["job_run_id"] == "job-123" for _query, params in profile_writes)
    assert all(
        params["profile_curation_policy_hash"] == "curation-policy-1"
        for _query, params in profile_writes
    )
    assert all(
        "e.ingested_profile_curation_policy_hashes" in query for query, _params in profile_writes
    )


@pytest.mark.asyncio
async def test_ingest_article_tx_does_not_use_description_as_profile_evidence() -> None:
    tx = FakeTx()
    prior_labs = resolved("Startup", "Prior Labs")
    extraction = ExtractionResult(
        startups=[
            ExtractedEntity(
                name="Prior Labs",
                evidence_status="stated",
                description="Prior Labs ist angeblich in Berlin ansässig.",
            )
        ],
    )

    await ingest_article_tx(
        tx,
        article(),
        extraction,
        resolved_mapping(prior_labs),
        raw_extracted_entities=None,
    )

    profile_writes = [params for query, params in tx.runs if "MERGE (e:ProfileEvidence" in query]
    assert profile_writes == []


@pytest.mark.asyncio
async def test_ingest_article_tx_skips_invalid_founder_direction() -> None:
    tx = FakeTx()
    startup = resolved("Startup", "ViViRA")
    person = resolved("Person", "Philip Heimann")
    extraction = ExtractionResult(
        startups=[ExtractedEntity(name="ViViRA", evidence_status="stated")],
        people=[ExtractedEntity(name="Philip Heimann", evidence_status="stated")],
    )
    extraction.relationships.append(
        ExtractedRelationship.model_construct(
            type="FOUNDED_BY",
            source_name="Philip Heimann",
            source_type="Person",
            target_name="ViViRA",
            target_type="Startup",
            evidence_status="stated",
            keywords="Gruendung",
            evidence="Philip Heimann ist Mitgruender von ViViRA.",
        )
    )

    await ingest_article_tx(
        tx,
        article(),
        extraction,
        resolved_mapping(startup, person),
        raw_extracted_entities=None,
    )

    assert not any("MERGE (source)-[r:FOUNDED_BY]->(target)" in query for query, _ in tx.runs)
