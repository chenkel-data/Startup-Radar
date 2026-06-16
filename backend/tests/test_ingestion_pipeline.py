from types import SimpleNamespace

import pytest

from app.models.extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    NormalizedEntity,
)
from app.services.entity_resolution import ResolutionOutcome
from app.services.ingestion import IngestionService


def test_evidence_gate_keeps_only_admitted_entities_and_relationships() -> None:
    extraction = ExtractionResult(
        startups=[
            ExtractedEntity(name="Leegle", evidence_status="stated"),
            ExtractedEntity(name="Rumor Startup", evidence_status="unsure"),
        ],
        investors=[ExtractedEntity(name="Christian Lindner", evidence_status="attributed")],
        relationships=[
            ExtractedRelationship(
                type="INVESTED_IN",
                source_name="Christian Lindner",
                source_type="Person",
                target_name="Leegle",
                target_type="Startup",
                evidence_status="stated",
            ),
            ExtractedRelationship(
                type="MERGED_WITH",
                source_name="Aleph Alpha",
                source_type="Startup",
                target_name="Cohere",
                target_type="Startup",
                evidence_status="unsure",
            ),
        ],
    )
    method = getattr(
        IngestionService._filter_by_evidence_traced,
        "__wrapped__",
        IngestionService._filter_by_evidence_traced,
    )

    filtered = method(object.__new__(IngestionService), extraction)

    assert [startup.name for startup in filtered.startups] == ["Leegle"]
    assert [investor.name for investor in filtered.investors] == ["Christian Lindner"]
    assert [(rel.type, rel.source_name, rel.target_name) for rel in filtered.relationships] == [
        ("INVESTED_IN", "Christian Lindner", "Leegle")
    ]


def test_ensure_relationship_entities_materializes_admitted_missing_endpoints() -> None:
    extraction = ExtractionResult(
        relationships=[
            ExtractedRelationship(
                type="FOUNDED_BY",
                source_name="Avelios Medical",
                source_type="Startup",
                target_name="Christopher Muhr",
                target_type="Person",
                evidence_status="stated",
            ),
            ExtractedRelationship(
                type="MERGED_WITH",
                source_name="Speculative One",
                source_type="Startup",
                target_name="Speculative Two",
                target_type="Startup",
                evidence_status="unsure",
            ),
        ]
    )

    IngestionService._ensure_relationship_entities(extraction)

    assert [startup.name for startup in extraction.startups] == ["Avelios Medical"]
    assert [person.name for person in extraction.people] == ["Christopher Muhr"]
    assert all("Speculative" not in startup.name for startup in extraction.startups)


class FakeProfileCuration:
    def __init__(self) -> None:
        self.entity_ids: list[list[str]] = []

    async def curate_entity_ids(self, entity_ids, *, job_run_id=None):
        self.entity_ids.append(list(entity_ids))
        return [
            {
                "entity_id": entity_ids[0],
                "status": "kept",
                "job_run_id": job_run_id,
            }
        ]

    async def curate_profiles(self, *_args, **_kwargs):
        raise AssertionError("graph-wide curation should not be used during ingestion")


class FakeLogger:
    def info(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None


@pytest.mark.asyncio
async def test_curates_only_resolved_entity_ids_after_article_write() -> None:
    service = object.__new__(IngestionService)
    profile_curation = FakeProfileCuration()
    service.settings = SimpleNamespace(enable_entity_description_curation=True)
    service.profile_curation = profile_curation
    service.logger = FakeLogger()
    sap = NormalizedEntity(
        id="company:sap",
        label="Company",
        canonical_name="SAP",
        name="SAP",
    )

    rows = await service._curate_resolved_entities_traced(
        [
            ResolutionOutcome(entity=sap, method="exact", candidate_name="SAP"),
            ResolutionOutcome(entity=sap, method="fuzzy", candidate_name="SAP SE"),
        ],
        job_run_id="job-1",
    )

    assert profile_curation.entity_ids == [["company:sap"]]
    assert rows == [
        {
            "entity_id": "company:sap",
            "status": "kept",
            "job_run_id": "job-1",
        }
    ]
