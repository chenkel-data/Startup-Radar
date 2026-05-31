import pytest

from app.models.extraction import (
    ArticleIn,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    IngestStats,
)
from app.services.ingestion import IngestionService, _ArticleResult, _evidence_status_counts


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


def test_evidence_status_counts_include_defaulted_records() -> None:
    extraction = ExtractionResult(
        startups=[
            ExtractedEntity(
                name="Defaulted Startup",
                evidence_status="unsure",
                evidence_status_defaulted=True,
            )
        ],
        relationships=[
            ExtractedRelationship(
                type="INVESTED_IN",
                source_name="Investor",
                source_type="Investor",
                target_name="Defaulted Startup",
                target_type="Startup",
                evidence_status="unsure",
                evidence_status_defaulted=True,
            )
        ],
    )

    counts = _evidence_status_counts(extraction)

    assert counts["entity_unsure"] == 1
    assert counts["entity_status_defaulted"] == 1
    assert counts["relationship_unsure"] == 1
    assert counts["relationship_status_defaulted"] == 1


def test_tally_accumulates_processed_and_failed_article_results() -> None:
    success = _ArticleResult()
    success.success = True
    success.graph_row = {"entity_count": 4, "graph_operations": 9}
    failure = _ArticleResult()
    failure.success = False
    stats = IngestionService._tally

    ingest_stats = IngestStats(source_name="deutsche-startups.de")
    stats(object.__new__(IngestionService), [success, failure], ingest_stats)

    assert ingest_stats.articles_processed == 1
    assert ingest_stats.articles_failed == 1
    assert ingest_stats.entities_extracted == 4
    assert ingest_stats.relationships_created == 9


@pytest.mark.asyncio
async def test_dispatch_articles_processes_each_article_with_batch_context() -> None:
    class FakeIngestionService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int, int]] = []

        async def _process_article(
            self,
            *,
            article: ArticleIn,
            resolver: object,
            job_run_id: str,
            article_index: int,
            article_total: int,
        ) -> _ArticleResult:
            assert resolver is fake_resolver
            self.calls.append((article.url, job_run_id, article_index, article_total))
            result = _ArticleResult()
            result.success = True
            return result

    articles = [
        ArticleIn(
            url="https://example.test/one",
            title="Article one",
            source_name="deutsche-startups.de",
            text="Article one has enough text for the ingestion model validator.",
        ),
        ArticleIn(
            url="https://example.test/two",
            title="Article two",
            source_name="deutsche-startups.de",
            text="Article two has enough text for the ingestion model validator.",
        ),
    ]
    fake_resolver = object()
    service = FakeIngestionService()

    results = await IngestionService._dispatch_articles(
        service,  # type: ignore[arg-type]
        articles=articles,
        resolver=fake_resolver,  # type: ignore[arg-type]
        job_run_id="job-123",
    )

    assert [result.success for result in results] == [True, True]
    assert service.calls == [
        ("https://example.test/one", "job-123", 1, 2),
        ("https://example.test/two", "job-123", 2, 2),
    ]
