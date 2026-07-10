from types import SimpleNamespace

import httpx
import pytest

from app.evidence import build_evidence_catalog
from app.models.extraction import ArticleIn
from app.prompts.extraction import build_extraction_user_prompt
from app.services.scraper import ArticleScraper


SOURCE_URL = "https://www.deutsche-startups.de"
ARTICLE_URL = f"{SOURCE_URL}/2026/07/10/startup-news/"


def article_html(
    *,
    title: str,
    primary_type: str,
    content: str,
    tags: tuple[str, ...] = (),
    canonical_url: str = ARTICLE_URL,
) -> str:
    tag_links = "".join(f'<a rel="tag">#{tag}</a>' for tag in tags)
    return f"""
    <html>
      <head>
        <link rel="canonical" href="{canonical_url}">
      </head>
      <body>
        <div id="post">
          <div class="postHead">
            <div class="topHeadline">{primary_type}</div>
            <h1>{title}</h1>
          </div>
          <div class="wysiwyg">{content}</div>
          <div id="tagOverview">{tag_links}</div>
        </div>
      </body>
    </html>
    """


def mock_client(html: str) -> httpx.AsyncClient:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(respond))


def scraper() -> ArticleScraper:
    settings = SimpleNamespace(scrape_timeout_seconds=5)
    return ArticleScraper(settings)


async def fetch_article(html: str) -> ArticleIn | None:
    async with mock_client(html) as client:
        return await scraper()._fetch_article(
            client,
            ARTICLE_URL,
            "deutsche-startups.de",
            SOURCE_URL,
        )


@pytest.mark.asyncio
async def test_discovery_keeps_all_main_archive_articles_and_excludes_sidebar_links(
    monkeypatch,
) -> None:
    discovered = [
        (
            f"{SOURCE_URL}/2026/{((index - 1) // 28) + 1:02d}/"
            f"{((index - 1) % 28) + 1:02d}/story-{index}/"
        )
        for index in range(1, 220)
    ]
    posts = "".join(
        f'<div class="post"><a href="{url}"><h2>Story {index}</h2></a></div>'
        for index, url in enumerate(discovered, start=1)
    )
    html = f"""
    <div id="content" class="hasSidebar">
      <div id="contentMain">
        <div id="archiveOverview">
          <div class="part">
            <div class="tagHeader only"><h1>Ressort: Startups</h1></div>
            <div class="part">{posts}</div>
          </div>
        </div>
        <div id="archiveOverview" class="categoryPart">
          <div class="part"><div class="post">
            <a href="/2026/06/18/thematic-event-promotion/">Thematic promotion</a>
          </div></div>
        </div>
      </div>
      <div id="contentSidebar"><div class="postlist"><div class="post">
        <a href="/2026/06/19/sidebar-event-promotion/">Sidebar promotion</a>
      </div></div></div>
    </div>
    """

    subject = scraper()
    async with mock_client(html) as client:
        links, exhausted = await subject._links_from_listing_page(
            client,
            SOURCE_URL,
            "/ressort/startups/",
            page=1,
            max_pages=1,
        )

    assert exhausted is False
    assert links == discovered

    async def listings(*_args, **_kwargs) -> list[str]:
        return links

    async def feed(*_args, **_kwargs) -> list[str]:
        raise AssertionError("Deutsche Startups discovery must use its main archives")

    monkeypatch.setattr(subject, "_links_from_listings", listings)
    monkeypatch.setattr(subject, "_links_from_feed", feed)

    collected = await subject.collect_links(
        source_url=SOURCE_URL,
        source_name="deutsche-startups.de",
        max_pages=50,
        include_feed=True,
        paths=["/ressort/startups/"],
    )

    assert collected == discovered


@pytest.mark.asyncio
async def test_fetch_article_preserves_listing_url_when_canonical_differs() -> None:
    canonical_url = f"{SOURCE_URL}/?p=448049"
    html = article_html(
        title="5 neue Startups",
        primary_type="#Brandneu",
        canonical_url=canonical_url,
        content=(
            "<p>Dieser ausführliche Artikel stellt mehrere neue Startups und ihre "
            "Geschäftsmodelle mit genügend Inhalt für die Verarbeitung vor.</p>"
        ),
    )

    article = await fetch_article(html)

    assert article is not None
    assert article.url == canonical_url
    assert article.discovered_url == ARTICLE_URL


@pytest.mark.asyncio
async def test_non_article_pages_are_not_sent_to_extraction() -> None:
    event_html = article_html(
        title="Bis zu 330.000 Euro für neue Ideen",
        primary_type="#Eventtipp",
        tags=("Eventtipp", "Köln"),
        content="""
          <p>Das Innovationsprogramm fördert gemeinsame Projekte von Unternehmen mit
          Zuschüssen von bis zu 330.000 Euro für neue Produkte und Geschäftsmodelle.</p>
          <p>Ein Webcast erklärt interessierten Startups die Voraussetzungen und Fristen.</p>
        """,
    )
    missing_container_html = """
    <html><body><article><div class="entry-content">
      <h1>Body fallback must not be used</h1>
      <p>This article-shaped fallback contains substantially more than one hundred and
      twenty characters, but it is outside the required source container and therefore
      must never be sent to the language model for extraction.</p>
    </div></article></body></html>
    """

    assert await fetch_article(event_html) is None
    assert await fetch_article(missing_container_html) is None


@pytest.mark.asyncio
async def test_mixed_ticker_keeps_business_facts_and_removes_pure_promotions() -> None:
    html = article_html(
        title="Union Square Ventures und CareerPilot im StartupTicker",
        primary_type="#StartupTicker",
        tags=("StartupTicker", "Eventtipp", "HRTech"),
        content="""
          <h2>#STARTUPTICKER</h2>
          <h3>Union Square Ventures</h3>
          <p>Union Square Ventures investiert in alqem. Das DeepTech entwickelt eine
          KI-Plattform für neue Materialien und beschäftigt bereits 45 Mitarbeitende.</p>
          <h3>CareerPilot</h3>
          <p>Das HRTech CareerPilot entwickelt eine Jobplattform für Berufseinsteiger
          und sammelte für den Produktausbau fünf Millionen Euro ein. Das Unternehmen
          beschäftigt inzwischen mehr als 40 Mitarbeitende.</p>
          <h3>Demo Conference</h3>
          <p>Die Konferenz bietet Speaker, Panels und Networking.
          <a href="https://demo.example/tickets">Tickets und Anmeldung</a>
          sind ab sofort verfügbar.</p>
          <h2>#JOBS</h2>
          <h3>BRABUS</h3>
          <p>Unser Job des Tages: BRABUS sucht einen Mitarbeiter Online Marketing.
          Die vollständige Stellenanzeige steht in unserer Jobbörse.</p>
        """,
    )

    article = await fetch_article(html)

    assert article is not None
    for fact in (
        "Union Square Ventures investiert in alqem",
        "CareerPilot entwickelt eine Jobplattform",
        "sammelte für den Produktausbau fünf Millionen Euro ein",
    ):
        assert fact in article.text
    for promotion in ("Demo Conference", "Tickets und Anmeldung", "BRABUS"):
        assert promotion not in article.text

    assert article.cleaning is not None
    assert article.cleaning.blocks_before > article.cleaning.blocks_after
    assert article.cleaning.remaining_promotion_markers == []
    assert article.cleaning.removed_blocks

    prompt = build_extraction_user_prompt(article)
    assert "Union Square Ventures investiert in alqem" in prompt
    assert "CareerPilot entwickelt eine Jobplattform" in prompt
    assert "Demo Conference" not in prompt
    assert "BRABUS" not in prompt


@pytest.mark.asyncio
async def test_mixed_blocks_remain_verbatim_while_pure_promotions_are_removed() -> None:
    civi_reach = (
        "CiviReach aus Garching wurde von Michael Weindl, Tobias Mesmer und Michael "
        "Firlus gegründet und entwickelt eine KI-Plattform für politische Kommunikation. "
        "Tipp: In unserem Newsletter Startup-Radar stellen wir jede Woche neue Startups vor."
    )
    check24_job = (
        "CHECK24 +++ Unser Job des Tages! CHECK24 sucht einen Product Manager. "
        "CHECK24 beschreibt sich anschließend als Vergleichsportal mit agilen Teams."
    )
    founder_event = (
        "Eventtipp: Beim Growth Festival spricht Lea Sommer, Gründerin von Nova, über "
        "den Aufbau des Unternehmens. Tickets und Anmeldung gibt es über die "
        "Veranstaltungsseite."
    )
    html = article_html(
        title="CiviReach und Nova im StartupTicker",
        primary_type="#StartupTicker",
        tags=("StartupTicker", "Eventtipp"),
        content=f"""
          <h2>#STARTUPTICKER</h2>
          <p>{civi_reach}
          <a href="https://startupradar.substack.com/">Newsletter</a></p>
          <h2>#JOBS</h2>
          <p>{check24_job}</p>
          <h3>Growth Festival</h3>
          <p>{founder_event}
          <a href="https://festival.example/tickets">Tickets</a></p>
          <p>Eventtipp: Die Demo Conference bietet Panels, Networking und Tickets.
          <a href="https://demo.example/tickets">Jetzt anmelden</a>.</p>
        """,
    )

    article = await fetch_article(html)

    assert article is not None
    for complete_mixed_block in (civi_reach, check24_job, founder_event):
        assert complete_mixed_block in article.text
    assert "Demo Conference bietet Panels" not in article.text

    prompt = build_extraction_user_prompt(article)
    catalog = build_evidence_catalog(article.text)
    for sentence in catalog.sentences.values():
        assert article.text[sentence.start : sentence.end] in prompt

    assert article.cleaning is not None
    assert "job_of_the_day_advertisement" in article.cleaning.remaining_promotion_markers
