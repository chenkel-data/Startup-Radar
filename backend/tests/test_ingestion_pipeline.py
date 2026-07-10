import pytest

from app.models.extraction import (
    ArticleIn,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    IngestRequest,
    IngestStats,
)
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
    article = ArticleIn(
        url="https://example.test/avelios",
        title="Avelios Medical wurde gegründet",
        source_name="example.test",
        text="Avelios Medical wurde von Christopher Muhr gegründet.",
    )
    extraction = ExtractionResult(
        relationships=[
            ExtractedRelationship(
                type="FOUNDED_BY",
                source_name="Avelios Medical",
                source_type="Startup",
                target_name="Christopher Muhr",
                target_type="Person",
                evidence_status="stated",
                evidence=article.text,
            ),
            ExtractedRelationship(
                type="MERGED_WITH",
                source_name="Speculative One",
                source_type="Startup",
                target_name="Speculative Two",
                target_type="Startup",
                evidence_status="unsure",
                evidence="Speculative One könnte mit Speculative Two fusionieren.",
            ),
        ]
    )

    IngestionService._ensure_relationship_entities(extraction, article)

    assert [startup.name for startup in extraction.startups] == ["Avelios Medical"]
    assert [person.name for person in extraction.people] == ["Christopher Muhr"]
    assert extraction.people[0].type_basis == "contextual"
    assert extraction.people[0].description == article.text
    assert extraction.people[0].source.model_dump() == {
        "article_url": article.url,
        "article_title": article.title,
        "evidence": article.text,
    }
    assert all("Speculative" not in startup.name for startup in extraction.startups)


class FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def info(self, message, *, extra=None) -> None:
        self.records.append(("info", message, extra or {}))

    def warning(self, message, *, extra=None) -> None:
        self.records.append(("warning", message, extra or {}))


class FakeRun:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def record_metric(self, key: str, value: float) -> None:
        self.metrics[key] = value


class FakeScraper:
    def __init__(self, links: list[str]) -> None:
        self.links = links
        self.fetched_links: list[str] = []

    async def collect_links(self, **_kwargs) -> list[str]:
        return self.links

    async def fetch_articles(self, *, links: list[str], **_kwargs) -> list[ArticleIn]:
        self.fetched_links = list(links)
        return [
            ArticleIn(
                url=link,
                title=f"Article {index}",
                source_name="example.test",
                source_url="https://example.test",
                text="This article has enough text to pass validation.",
            )
            for index, link in enumerate(links, start=1)
        ]


class FakeArticleWriter:
    def __init__(self, cached_urls: set[str], *, lookup_fails: bool = False) -> None:
        self.cached_urls = cached_urls
        self.lookup_fails = lookup_fails
        self.lookup_calls: list[tuple[list[str], str]] = []

    async def cached_article_urls_for_extraction_policy(
        self,
        urls: list[str],
        article_extraction_policy_hash: str,
    ) -> set[str]:
        self.lookup_calls.append((list(urls), article_extraction_policy_hash))
        if self.lookup_fails:
            raise RuntimeError("database unavailable")
        return self.cached_urls.intersection(urls)


def scrape_service(
    links: list[str],
    cached_urls: set[str],
    *,
    cache_lookup_fails: bool = False,
) -> IngestionService:
    service = object.__new__(IngestionService)
    service.scraper = FakeScraper(links)
    service.article_writer = FakeArticleWriter(
        cached_urls,
        lookup_fails=cache_lookup_fails,
    )
    service.logger = FakeLogger()
    return service


def ingest_request(*, force_rescrape: bool = False) -> IngestRequest:
    return IngestRequest(
        source_url="https://example.test",
        source_name="example.test",
        max_pages=1,
        include_feed=False,
        force_rescrape=force_rescrape,
        paths=["/news/"],
    )


@pytest.mark.asyncio
async def test_scrape_stage_fetches_only_uncached_articles() -> None:
    links = ["https://example.test/a/1", "https://example.test/a/2"]
    service = scrape_service(links, {"https://example.test/a/1"})
    stats = IngestStats(source_name="example.test")

    articles = await service._scrape_stage(
        ingest_request(),
        stats,
        FakeRun(),
        article_extraction_policy_hash="policy-1",
    )

    assert [article.url for article in articles] == ["https://example.test/a/2"]
    assert service.scraper.fetched_links == ["https://example.test/a/2"]
    assert stats.articles_found == 2
    assert stats.articles_cached == 1
    assert stats.articles_skipped == 1
    assert stats.articles_scraped == 1
    assert stats.cache_message is None


@pytest.mark.asyncio
async def test_scrape_stage_force_rescrape_fetches_cached_articles() -> None:
    links = ["https://example.test/a/1", "https://example.test/a/2"]
    service = scrape_service(links, set(links))
    stats = IngestStats(source_name="example.test")

    articles = await service._scrape_stage(
        ingest_request(force_rescrape=True),
        stats,
        FakeRun(),
        article_extraction_policy_hash="policy-1",
    )

    assert [article.url for article in articles] == links
    assert service.scraper.fetched_links == links
    assert service.article_writer.lookup_calls == []
    assert stats.articles_found == 2
    assert stats.articles_cached == 0
    assert stats.articles_skipped == 0
    assert stats.articles_scraped == 2
    assert stats.cache_message is None


@pytest.mark.asyncio
async def test_scrape_stage_explains_cache_lookup_failure_before_fetching_all() -> None:
    links = ["https://example.test/a/1", "https://example.test/a/2"]
    service = scrape_service(links, set(), cache_lookup_fails=True)
    stats = IngestStats(source_name="example.test")

    articles = await service._scrape_stage(
        ingest_request(),
        stats,
        FakeRun(),
        article_extraction_policy_hash="policy-1",
    )

    assert [article.url for article in articles] == links
    assert service.scraper.fetched_links == links
