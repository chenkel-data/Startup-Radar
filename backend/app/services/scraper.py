import asyncio
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import urljoin, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.extraction import ArticleIn
from app.services.progress import article_fields


class ArticleScraper:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("scraper")
        self._headers = {
            "User-Agent": "startup-radar/0.1 (+https://localhost; research ingestion)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    async def collect(
        self,
        *,
        source_url: str,
        source_name: str,
        max_pages: int,
        include_feed: bool,
        paths: list[str],
    ) -> list[ArticleIn]:
        timeout = httpx.Timeout(self.settings.scrape_timeout_seconds)
        async with httpx.AsyncClient(
            headers=self._headers, timeout=timeout, follow_redirects=True
        ) as client:
            self.logger.info(
                "scrape_collection_started",
                extra={
                    "event": "scraping",
                    "workflow_step": "collect",
                    "url": source_url,
                    "detail": (
                        f"source={source_name}; include_feed={include_feed}; "
                        f"listing_paths={len(paths)}; max_pages={max_pages}"
                    ),
                },
            )
            links: list[str] = []
            if include_feed:
                links.extend(await self._links_from_feed(client, source_url))

            links.extend(await self._links_from_listings(client, source_url, paths, max_pages))

            links = _dedupe([link for link in links if _same_host(source_url, link)])
            links = links[: self.settings.max_articles_per_ingest]
            self.logger.info(
                "article_links_collected",
                extra={
                    "event": "scraping",
                    "workflow_step": "collect",
                    "count": len(links),
                    "url": source_url,
                    "detail": f"deduped_and_capped={len(links)}",
                },
            )

            semaphore = asyncio.Semaphore(6)
            completed = 0
            progress_lock = asyncio.Lock()

            async def mark_completed() -> tuple[int, int]:
                nonlocal completed
                async with progress_lock:
                    completed += 1
                    return completed, len(links) - completed

            async def fetch_link(index: int, link: str) -> ArticleIn | None:
                async with semaphore:
                    self.logger.info(
                        "article_fetch_started",
                        extra={
                            "event": "scraping",
                            "workflow_step": "article_fetch",
                            "url": link,
                            "article_index": index,
                            "article_total": len(links),
                        },
                    )
                    try:
                        article = await self._fetch_article(client, link, source_name, source_url)
                        done, remaining = await mark_completed()
                        if article:
                            self.logger.info(
                                "article_fetch_completed",
                                extra={
                                    "event": "scraping",
                                    "workflow_step": "article_fetch",
                                    "completed_count": done,
                                    "article_total": len(links),
                                    "remaining": remaining,
                                    "detail": f"text_chars={len(article.text)}, tags={len(article.tags)}",
                                    **article_fields(
                                        article, article_index=index, article_total=len(links)
                                    ),
                                },
                            )
                        else:
                            self.logger.info(
                                "article_fetch_skipped",
                                extra={
                                    "event": "scraping",
                                    "workflow_step": "article_fetch",
                                    "url": link,
                                    "article_index": index,
                                    "article_total": len(links),
                                    "completed_count": done,
                                    "remaining": remaining,
                                },
                            )
                        return article
                    except Exception as exc:
                        done, remaining = await mark_completed()
                        self.logger.warning(
                            "article_fetch_failed",
                            extra={
                                "event": "scraping",
                                "workflow_step": "article_fetch",
                                "url": link,
                                "error": str(exc),
                                "article_index": index,
                                "article_total": len(links),
                                "completed_count": done,
                                "remaining": remaining,
                            },
                        )
                        return None

            articles = await asyncio.gather(
                *(fetch_link(index, link) for index, link in enumerate(links, start=1))
            )
            valid_articles = [article for article in articles if article]
            self.logger.info(
                "articles_scraped",
                extra={
                    "event": "scraping",
                    "workflow_step": "collect",
                    "count": len(valid_articles),
                    "url": source_url,
                    "detail": f"valid={len(valid_articles)}, skipped_or_failed={len(links) - len(valid_articles)}",
                },
            )
            return valid_articles

    async def _links_from_feed(self, client: httpx.AsyncClient, source_url: str) -> list[str]:
        feed_url = urljoin(source_url.rstrip("/") + "/", "feed/")
        self.logger.info(
            "feed_fetch_started",
            extra={"event": "scraping", "workflow_step": "feed", "url": feed_url},
        )
        response = await client.get(feed_url)
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
        links = [entry.link for entry in parsed.entries if getattr(entry, "link", None)]
        self.logger.info(
            "feed_links_collected",
            extra={
                "event": "scraping",
                "workflow_step": "feed",
                "url": feed_url,
                "count": len(links),
            },
        )
        return links

    async def _links_from_listings(
        self,
        client: httpx.AsyncClient,
        source_url: str,
        paths: list[str],
        max_pages: int,
    ) -> list[str]:
        links: list[str] = []
        normalized_paths = ["/" + path.strip("/") + "/" for path in paths]
        exhausted_paths: set[str] = set()

        for path in normalized_paths:
            self._log_listing_scan(source_url, path, max_pages)

        for page in range(1, max_pages + 1):
            for path in normalized_paths:
                if path in exhausted_paths:
                    continue
                page_links, exhausted = await self._links_from_listing_page(
                    client, source_url, path, page, max_pages
                )
                links.extend(page_links)
                if exhausted:
                    exhausted_paths.add(path)
        return links

    async def _links_from_listing(
        self,
        client: httpx.AsyncClient,
        source_url: str,
        path: str,
        max_pages: int,
    ) -> list[str]:
        links: list[str] = []
        normalized = "/" + path.strip("/") + "/"
        self._log_listing_scan(source_url, normalized, max_pages)
        for page in range(1, max_pages + 1):
            page_links, exhausted = await self._links_from_listing_page(
                client, source_url, normalized, page, max_pages
            )
            links.extend(page_links)
            if exhausted:
                break
        return links

    def _log_listing_scan(self, source_url: str, normalized_path: str, max_pages: int) -> None:
        self.logger.info(
            "listing_scan_started",
            extra={
                "event": "scraping",
                "workflow_step": "listing",
                "url": urljoin(source_url.rstrip("/") + "/", normalized_path.lstrip("/")),
                "detail": f"path={normalized_path}; pages={max_pages}",
            },
        )

    async def _links_from_listing_page(
        self,
        client: httpx.AsyncClient,
        source_url: str,
        normalized_path: str,
        page: int,
        max_pages: int,
    ) -> tuple[list[str], bool]:
        page_path = normalized_path if page == 1 else f"{normalized_path}page/{page}/"
        page_url = urljoin(source_url.rstrip("/") + "/", page_path.lstrip("/"))
        self.logger.info(
            "listing_page_fetch_started",
            extra={
                "event": "scraping",
                "workflow_step": "listing",
                "url": page_url,
                "page_index": page,
                "page_total": max_pages,
            },
        )
        try:
            response = await client.get(page_url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 404 and page > 1:
                self.logger.info(
                    "listing_pagination_exhausted",
                    extra={
                        "event": "scraping",
                        "workflow_step": "listing",
                        "url": page_url,
                        "status_code": status_code,
                        "page_index": page,
                        "page_total": max_pages,
                        "detail": f"path={normalized_path}; last_available_page={page - 1}",
                    },
                )
                return [], True
            self.logger.warning(
                "listing_page_failed",
                extra={
                    "event": "scraping",
                    "workflow_step": "listing",
                    "url": page_url,
                    "error": str(exc),
                    "status_code": status_code,
                    "page_index": page,
                    "page_total": max_pages,
                },
            )
            return [], False
        except Exception as exc:
            self.logger.warning(
                "listing_page_failed",
                extra={
                    "event": "scraping",
                    "workflow_step": "listing",
                    "url": page_url,
                    "error": str(exc),
                    "page_index": page,
                    "page_total": max_pages,
                },
            )
            return [], False

        soup = BeautifulSoup(response.text, "html.parser")
        page_links = [
            urljoin(page_url, anchor.get("href"))
            for anchor in soup.select(
                ".post a[href], article a[href], main a[href], h2 a[href], h3 a[href]"
            )
            if anchor.get("href")
        ]
        filtered = [link for link in page_links if _looks_like_article(source_url, link)]
        self.logger.info(
            "listing_links_collected",
            extra={
                "event": "scraping",
                "workflow_step": "listing",
                "url": page_url,
                "count": len(filtered),
                "page_index": page,
                "page_total": max_pages,
                "detail": f"raw_links={len(page_links)}, article_links={len(filtered)}",
            },
        )
        return filtered, False

    async def _fetch_article(
        self,
        client: httpx.AsyncClient,
        url: str,
        source_name: str,
        source_url: str,
    ) -> ArticleIn | None:
        response = await client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        tags = _extract_tags(soup)

        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "aside"]):
            tag.decompose()

        title = _first_text(
            [
                soup.select_one("h1"),
                soup.select_one("meta[property='og:title']"),
                soup.select_one("title"),
            ]
        )
        canonical = soup.select_one("link[rel='canonical']")
        canonical_url = canonical.get("href") if canonical and canonical.get("href") else url

        container = (
            soup.select_one("article .entry-content")
            or soup.select_one("article .post-content")
            or soup.select_one("article")
            or soup.select_one("main")
            or soup.body
        )
        if not container:
            self.logger.info(
                "article_parse_skipped",
                extra={
                    "event": "scraping",
                    "workflow_step": "article_parse",
                    "url": url,
                    "detail": "no readable article container",
                },
            )
            return None

        paragraphs = [
            " ".join(node.get_text(" ", strip=True).split())
            for node in container.select("p, li")
            if len(node.get_text(" ", strip=True)) > 35
        ]
        text = "\n".join(_dedupe(paragraphs))
        if len(text) < 120:
            self.logger.info(
                "article_parse_skipped",
                extra={
                    "event": "scraping",
                    "workflow_step": "article_parse",
                    "url": url,
                    "detail": f"text too short ({len(text)} chars)",
                },
            )
            return None

        summary_node = soup.select_one("meta[name='description'], meta[property='og:description']")
        summary = (
            summary_node.get("content") if summary_node and summary_node.get("content") else None
        )
        author = _first_text([soup.select_one("[rel='author']"), soup.select_one(".author")])
        published_at = _published_at(soup)

        return ArticleIn(
            url=canonical_url,
            title=title or canonical_url,
            source_name=source_name,
            source_url=source_url,
            author=author,
            published_at=published_at,
            summary=summary,
            text=text,
            tags=tags,
        )


def _first_text(nodes: Iterable) -> str | None:
    for node in nodes:
        if not node:
            continue
        if getattr(node, "name", "") == "meta":
            value = node.get("content")
        else:
            value = node.get_text(" ", strip=True)
        if value:
            return " ".join(value.split())
    return None


def _published_at(soup: BeautifulSoup) -> datetime | None:
    candidates = [
        soup.select_one("time[datetime]"),
        soup.select_one("meta[property='article:published_time']"),
        soup.select_one("meta[name='date']"),
    ]
    for node in candidates:
        if not node:
            continue
        value = node.get("datetime") or node.get("content")
        if not value:
            continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                return parsedate_to_datetime(value)
            except Exception:
                continue
    return None


def _extract_tags(soup: BeautifulSoup) -> list[str]:
    tag_texts = [
        label.strip("#")
        for tag in soup.select(
            "a[rel='tag'], #tagOverview a, #tagOverview [data-tag], .tags a, .post-tags a"
        )
        if (label := _tag_label(tag))
    ]
    tag_texts.extend(
        node.get("content", "").strip("#")
        for node in soup.select('meta[property="article:tag"], meta[name="keywords"]')
        if node.get("content")
    )

    tags: list[str] = []
    for value in tag_texts:
        tags.extend(part.strip() for part in value.split(","))
    return _dedupe(tag for tag in tags if tag)


def _tag_label(node) -> str:
    return node.get_text(" ", strip=True) or node.get("data-tag", "").strip()


def _same_host(source_url: str, link: str) -> bool:
    return urlparse(source_url).netloc.replace("www.", "") == urlparse(link).netloc.replace(
        "www.", ""
    )


def _looks_like_article(source_url: str, link: str) -> bool:
    parsed = urlparse(link)
    if not _same_host(source_url, link):
        return False
    blocked = [
        "/tag/",
        "/category/",
        "/author/",
        "/ressort/",
        "/page/",
        "/jobs",
        "/feed",
        "/wp-content/",
        "#",
    ]
    if any(part in link for part in blocked):
        return False
    path = parsed.path.strip("/")
    return bool(path) and any(char.isdigit() for char in path)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
