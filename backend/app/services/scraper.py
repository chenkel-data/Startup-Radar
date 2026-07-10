import asyncio
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
import re
from typing import Iterable
import unicodedata
from urllib.parse import urljoin, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup, Tag

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.extraction import (
    ArticleCleaningReport,
    ArticleContentRemoval,
    ArticleIn,
)
from app.services.progress import article_fields


_DEUTSCHE_STARTUPS_HOST = "deutsche-startups.de"
_DEUTSCHE_STARTUPS_LISTING_SELECTOR = (
    "#archiveOverview:not(.categoryPart) > .part > .part > .post > a[href]"
)
_DEUTSCHE_STARTUPS_CONTENT_SELECTOR = "#post .wysiwyg"
_GENERIC_LISTING_SELECTOR = ".post a[href], article a[href], main a[href], h2 a[href], h3 a[href]"


@dataclass(frozen=True)
class _ArticleBlock:
    kind: str
    text: str
    hrefs: tuple[str, ...] = ()
    heading_level: int | None = None


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
        links = await self.collect_links(
            source_url=source_url,
            source_name=source_name,
            max_pages=max_pages,
            include_feed=include_feed,
            paths=paths,
        )
        return await self.fetch_articles(
            source_url=source_url,
            source_name=source_name,
            links=links,
        )

    async def collect_links(
        self,
        *,
        source_url: str,
        source_name: str,
        max_pages: int,
        include_feed: bool,
        paths: list[str],
    ) -> list[str]:
        timeout = httpx.Timeout(self.settings.scrape_timeout_seconds)
        use_feed = include_feed and not _is_deutsche_startups_url(source_url)
        async with httpx.AsyncClient(
            headers=self._headers, timeout=timeout, follow_redirects=True
        ) as client:
            self.logger.info(
                "scrape_collection_started",
                extra={
                    "event": "scraping",
                    "workflow_step": "collect",
                    "url": source_url,
                    "summary": (
                        f"Collecting article links from {source_name}: "
                        f"{'include' if use_feed else 'skip'} the feed and scan "
                        f"{len(paths)} listing paths with up to {max_pages} pages each."
                    ),
                },
            )
            links: list[str] = []
            if use_feed:
                links.extend(await self._links_from_feed(client, source_url))

            links.extend(await self._links_from_listings(client, source_url, paths, max_pages))

            links = _dedupe([link for link in links if _same_host(source_url, link)])
            summary = f"Found {len(links)} unique article links; all were selected for this run."
            self.logger.info(
                "article_links_collected",
                extra={
                    "event": "scraping",
                    "workflow_step": "collect",
                    "url": source_url,
                    "summary": summary,
                    "article_links_found": len(links),
                    "article_links_selected": len(links),
                },
            )
            return links

    async def fetch_articles(
        self,
        *,
        source_url: str,
        source_name: str,
        links: list[str],
    ) -> list[ArticleIn]:
        timeout = httpx.Timeout(self.settings.scrape_timeout_seconds)
        async with httpx.AsyncClient(
            headers=self._headers, timeout=timeout, follow_redirects=True
        ) as client:
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
                                    "detail": _article_cleaning_log_detail(article),
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
            return [article for article in articles if article]

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
        selector = (
            _DEUTSCHE_STARTUPS_LISTING_SELECTOR
            if _is_deutsche_startups_url(source_url)
            else _GENERIC_LISTING_SELECTOR
        )
        page_links = [
            urljoin(page_url, anchor.get("href"))
            for anchor in soup.select(selector)
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
        primary_type = _first_text([soup.select_one("#post .postHead .topHeadline")])

        if _is_deutsche_startups_url(source_url) and _is_primary_eventtipp(primary_type):
            self.logger.info(
                "article_parse_skipped",
                extra={
                    "event": "scraping",
                    "workflow_step": "article_parse",
                    "url": url,
                    "primary_type": primary_type,
                    "detail": "primary article type is #Eventtipp",
                },
            )
            return None

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

        selected_container, container = _select_article_container(
            soup,
            strict_deutsche_startups=_is_deutsche_startups_url(source_url),
        )
        if not container:
            self.logger.info(
                "article_parse_skipped",
                extra={
                    "event": "scraping",
                    "workflow_step": "article_parse",
                    "url": url,
                    "primary_type": primary_type,
                    "detail": (
                        f"required article container missing ({selected_container}); "
                        "whole-page fallback disabled"
                    ),
                },
            )
            return None

        if _is_deutsche_startups_url(source_url):
            text, cleaning = _clean_deutsche_startups_article(
                container,
                article_url=canonical_url,
                tags=tags,
            )
        else:
            text, cleaning = _extract_generic_article(container, selected_container)

        self.logger.info(
            "article_content_cleaned",
            extra={
                "event": "scraping",
                "workflow_step": "article_parse",
                "url": canonical_url,
                "primary_type": primary_type,
                "selected_container": cleaning.selected_container,
                "blocks_before": cleaning.blocks_before,
                "blocks_after": cleaning.blocks_after,
                "text_chars_before": cleaning.text_chars_before,
                "text_chars_after": cleaning.text_chars_after,
                "removed_blocks": [
                    block.model_dump(mode="json") for block in cleaning.removed_blocks
                ],
                "remaining_promotion_markers": cleaning.remaining_promotion_markers,
                "detail": (
                    f"container={cleaning.selected_container}; primary_type={primary_type or 'unknown'}; "
                    f"blocks={cleaning.blocks_before}->{cleaning.blocks_after}; "
                    f"chars={cleaning.text_chars_before}->{cleaning.text_chars_after}; "
                    f"removed={len(cleaning.removed_blocks)}; "
                    f"remaining_markers={len(cleaning.remaining_promotion_markers)}"
                ),
            },
        )
        if len(text) < 120:
            self.logger.info(
                "article_parse_skipped",
                extra={
                    "event": "scraping",
                    "workflow_step": "article_parse",
                    "url": url,
                    "primary_type": primary_type,
                    "detail": f"clean article text too short ({len(text)} chars)",
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
            discovered_url=url if url != canonical_url else None,
            title=title or canonical_url,
            source_name=source_name,
            source_url=source_url,
            author=author,
            published_at=published_at,
            summary=summary,
            text=text,
            tags=tags,
            primary_type=primary_type,
            cleaning=cleaning,
        )


def _select_article_container(
    soup: BeautifulSoup,
    *,
    strict_deutsche_startups: bool,
) -> tuple[str, Tag | None]:
    if strict_deutsche_startups:
        return _DEUTSCHE_STARTUPS_CONTENT_SELECTOR, soup.select_one(
            _DEUTSCHE_STARTUPS_CONTENT_SELECTOR
        )

    for selector in (
        "article .entry-content",
        "article .post-content",
        "article",
        "main",
    ):
        if container := soup.select_one(selector):
            return selector, container
    return "article or main", None


def _clean_deutsche_startups_article(
    container: Tag,
    *,
    article_url: str,
    tags: list[str],
) -> tuple[str, ArticleCleaningReport]:
    blocks = _article_blocks(container, article_url)
    return _clean_deutsche_startups_blocks(
        blocks,
        tags=tags,
    )


def _clean_deutsche_startups_blocks(
    blocks: list[_ArticleBlock],
    *,
    tags: list[str],
) -> tuple[str, ArticleCleaningReport]:
    before_text = _render_article_blocks(blocks)
    removal_reasons: dict[int, str] = {}
    current_heading: str | None = None
    seen_paragraphs: set[tuple[str | None, str]] = set()

    for block_index, block in enumerate(blocks, start=1):
        if block.kind == "heading":
            current_heading = block.text
            if reason := _heading_removal_reason(block.text):
                removal_reasons[block_index] = reason
            continue

        reason = _individual_removal_reason(block, tags=tags)
        paragraph_key = block.text.casefold()
        contextual_paragraph_key = (
            current_heading.casefold() if current_heading else None,
            paragraph_key,
        )
        if reason is None and contextual_paragraph_key in seen_paragraphs:
            reason = "duplicate_content"
        if reason:
            removal_reasons[block_index] = reason
        else:
            seen_paragraphs.add(contextual_paragraph_key)

    for block_index, block in enumerate(blocks, start=1):
        if block.kind != "heading" or block_index in removal_reasons:
            continue
        normalized_heading = _fold_for_match(block.text).lstrip("#").strip()
        removes_empty_group = normalized_heading.startswith(("startup-radar", "startup radar"))
        if (block.heading_level or 0) <= 2 and not removes_empty_group:
            continue
        if not _has_retained_content_before_next_heading(
            blocks,
            start_index=block_index,
            heading_level=block.heading_level or 0,
            removal_reasons=removal_reasons,
        ):
            removal_reasons[block_index] = "orphan_heading"

    kept = [
        block
        for block_index, block in enumerate(blocks, start=1)
        if block_index not in removal_reasons
    ]
    removed = [
        _removal_record(
            block,
            removal_reasons[block_index],
            block_index=block_index,
            heading=_heading_before(blocks, block_index),
        )
        for block_index, block in enumerate(blocks, start=1)
        if block_index in removal_reasons
    ]
    text = _render_article_blocks(kept)
    report = ArticleCleaningReport(
        selected_container=_DEUTSCHE_STARTUPS_CONTENT_SELECTOR,
        blocks_before=len(blocks),
        blocks_after=len(kept),
        text_chars_before=len(before_text),
        text_chars_after=len(text),
        removed_blocks=removed,
        remaining_promotion_markers=_remaining_promotion_markers(
            kept,
            tags=tags,
        ),
    )
    return text, report


def _has_retained_content_before_next_heading(
    blocks: list[_ArticleBlock],
    *,
    start_index: int,
    heading_level: int,
    removal_reasons: dict[int, str],
) -> bool:
    for candidate_index, candidate in enumerate(
        blocks[start_index:],
        start=start_index + 1,
    ):
        if candidate.kind == "heading":
            if (candidate.heading_level or 0) <= heading_level:
                return False
            continue
        if candidate_index not in removal_reasons:
            return True
    return False


def _heading_before(blocks: list[_ArticleBlock], block_index: int) -> str | None:
    block = blocks[block_index - 1]
    if block.kind == "heading":
        return block.text
    for candidate in reversed(blocks[: block_index - 1]):
        if candidate.kind == "heading":
            return candidate.text
    return None


def _extract_generic_article(
    container: Tag,
    selected_container: str,
) -> tuple[str, ArticleCleaningReport]:
    paragraphs = [
        _normalize_text(node.get_text(" ", strip=True))
        for node in container.select("p, li")
        if len(_normalize_text(node.get_text(" ", strip=True))) > 35
    ]
    paragraphs = _dedupe(paragraphs)
    text = "\n".join(paragraphs)
    report = ArticleCleaningReport(
        selected_container=selected_container,
        blocks_before=len(paragraphs),
        blocks_after=len(paragraphs),
        text_chars_before=len(text),
        text_chars_after=len(text),
        remaining_promotion_markers=_remaining_text_markers(text),
    )
    return text, report


def _article_blocks(container: Tag, article_url: str) -> list[_ArticleBlock]:
    blocks: list[_ArticleBlock] = []
    semantic_names = {"h2", "h3", "h4", "p", "li"}
    for node in container.descendants:
        if not isinstance(node, Tag):
            continue
        is_semantic_block = node.name in semantic_names
        is_direct_wrapper = (
            node.parent is container
            and not is_semantic_block
            and node.select_one("h2, h3, h4, p, li") is None
        )
        if not is_semantic_block and not is_direct_wrapper:
            continue
        if node.name in {"p", "li"} and _has_selected_text_ancestor(node, container):
            continue

        text = _normalize_text(node.get_text(" ", strip=True))
        if not text:
            continue
        hrefs = tuple(
            urljoin(article_url, href)
            for anchor in node.select("a[href]")
            if (href := anchor.get("href"))
        )
        if node.name in {"p", "li"} or is_direct_wrapper:
            kind = "paragraph"
        else:
            kind = "heading"
        heading_level = int(node.name[1]) if kind == "heading" else None
        blocks.append(
            _ArticleBlock(
                kind=kind,
                text=text,
                hrefs=hrefs,
                heading_level=heading_level,
            )
        )
    return blocks


def _has_selected_text_ancestor(node: Tag, container: Tag) -> bool:
    parent = node.parent
    while isinstance(parent, Tag) and parent is not container:
        if parent.name in {"p", "li"}:
            return True
        parent = parent.parent
    return False


def _render_article_blocks(blocks: list[_ArticleBlock]) -> str:
    return "\n".join(
        f"## {block.text}" if block.kind == "heading" else block.text for block in blocks
    )


def _heading_removal_reason(heading: str) -> str | None:
    normalized = _fold_for_match(heading).lstrip("#").strip()
    if normalized == "jobs":
        return "jobs_heading"
    if normalized == "eventtipp":
        return "event_heading"
    if normalized == "startupland":
        return "startupland_promotion"
    if normalized.startswith("startupland:") and (
        "save the date" in normalized
        or "noch " in normalized
        or "fruhe vogel" in normalized
        or "fruhen vogel" in normalized
        or "ticket" in normalized
    ):
        return "startupland_promotion"
    if normalized in {"kolnbusiness", "koelnbusiness"} or re.search(
        r"\bdu(?:r)?chstarten\b.*\b(?:kolnbusiness|koelnbusiness)\b",
        normalized,
    ):
        return "sponsorship_heading"
    return None


def _individual_removal_reason(
    block: _ArticleBlock,
    *,
    tags: list[str],
) -> str | None:
    normalized = _fold_for_match(block.text)
    jobs_link = any("/startups-jobs/" in urlparse(href).path for href in block.hrefs)
    job_ad_link = any(
        "/startups-jobs/stellenangebote/" in urlparse(href).path for href in block.hrefs
    )
    amazon_link = any(
        urlparse(href).netloc.casefold().removeprefix("www.")
        in {"amazon.de", "amazon.com", "amzn.to"}
        for href in block.hrefs
    )

    if normalized.startswith("startup-jobs") and jobs_link:
        return "startup_jobs_promotion"
    if _is_vacancy_only_block(normalized, job_ad_link=job_ad_link):
        return "job_of_the_day_advertisement"
    if _is_startup_radar_promotion(block, normalized):
        return "startup_radar_promotion"
    if normalized.startswith("was ist zuletzt sonst passiert?") and "startupticker" in normalized:
        return "startupticker_closing_promotion"
    if "diese rubrik wird unterstutzt von" in normalized or (
        normalized.startswith("unser themenschwerpunkt") and "wird prasentiert von" in normalized
    ):
        return "sponsorship_disclosure"
    if (
        normalized.startswith("was gibt")
        and "startupticker" in normalized
        and (
            ("wochenruckblick" in normalized and "schnellen uberblick" in normalized)
            or (
                "kompakte ubersicht" in normalized and "startup-nachrichten des tages" in normalized
            )
        )
    ):
        return "roundup_intro"
    if normalized.startswith("jeden tag entstehen uberall im lande neue startups") and (
        "startup der woche" in normalized
    ):
        return "startup_of_week_intro"
    if normalized.startswith("tipp:") or normalized.startswith("tipp :"):
        if (
            ("startups to watch" in normalized and "rubrik" in normalized)
            or ("startup-investments" in normalized and "dealmonitor" in normalized)
            or ("mehr super angels" in normalized and "ubersicht" in normalized)
        ):
            return "internal_cross_promotion"
    if (
        amazon_link
        and "amazon" in normalized
        and ("buch" in normalized or "werk" in normalized or "lesestoff" in normalized)
    ):
        return "book_promotion"
    if re.match(r"^(?:foto|photo|bild)(?:\s*\(oben\))?\s*:", normalized):
        return "image_credit"
    if _is_startupland_promotion(block, normalized):
        return "startupland_promotion"
    if _is_event_only_promotion(block, tags=tags):
        return "event_promotion"
    return None


def _is_vacancy_only_block(normalized: str, *, job_ad_link: bool) -> bool:
    if "unser job des tages" not in normalized or "sucht" not in normalized:
        return False
    if len(normalized) > 280:
        return False
    has_vacancy_cta = any(
        marker in normalized
        for marker in ("stellenanzeige", "jobborse", "jetzt bewerben", "zur stelle")
    )
    return has_vacancy_cta and (job_ad_link or "jobborse" in normalized)


def _is_startup_radar_promotion(block: _ArticleBlock, normalized: str) -> bool:
    has_newsletter = "newsletter" in normalized and bool(
        re.search(r"\bstart(?:up| up)[ -]?radar\b", normalized)
    )
    has_radar_link = any(
        urlparse(href).netloc.casefold().removeprefix("www.")
        in {"startupradar.substack.com", _DEUTSCHE_STARTUPS_HOST}
        for href in block.hrefs
    )
    if normalized.startswith(("tipp:", "tipp :")):
        return has_newsletter and has_radar_link
    if normalized.startswith("in unserem newsletter startup-radar"):
        return has_newsletter and has_radar_link
    if normalized.startswith("newsletter +++"):
        return has_newsletter and has_radar_link
    if normalized.startswith("brandneu +++"):
        return "mehr im startup-radar" in normalized and has_radar_link
    if normalized.startswith("startup-radar +++"):
        return "mehr im startup-radar" in normalized and has_radar_link
    if normalized.startswith("heute prasentiert deutsche-startups.de") or normalized.startswith(
        "deutsche-startups.de prasentiert"
    ):
        return has_newsletter and has_radar_link
    return False


def _is_startupland_promotion(block: _ArticleBlock, normalized: str) -> bool:
    startupland_link = any(
        urlparse(href).netloc.casefold().removeprefix("www.") == "startupland.de"
        for href in block.hrefs
    )
    if normalized.startswith(
        (
            "startupland is where the future begin",
            "startupland: founders. vcs. visionaries",
            "fomo? absolutely. this is where it all happens",
            "startupland 2025 steigt im",
            "startupland 2027: jetzt ticket",
            "startupland: hier pitchen vcs",
            "sei dabei und buche jetzt schnell deine reise ins startupland",
            "ruckblick: das war die startupland",
            "+++ bei unserem reverse pitch im startupland",
        )
    ):
        return True
    if normalized.startswith("the next unicorn?"):
        return startupland_link and "startupland" in normalized
    if normalized.startswith("+++ kommt mit ins startupland"):
        return startupland_link and "ticket" in normalized
    if normalized.startswith("+++ schnell sein lohnt sich"):
        return "startupland conference" in normalized and "ticket" in normalized
    if re.match(r"^\+\+\+ am \d{1,2}\.\s", normalized):
        return startupland_link and "startupland conference" in normalized
    return False


def _is_event_only_promotion(
    block: _ArticleBlock,
    *,
    tags: list[str],
) -> bool:
    normalized = _fold_for_match(block.text)
    explicit_eventtipp = "eventtipp:" in normalized or "eventtipp :" in normalized
    eventtipp_tag = any(_fold_for_match(tag).lstrip("#").strip() == "eventtipp" for tag in tags)
    if not explicit_eventtipp and not eventtipp_tag:
        return False

    event_signal_patterns = (
        r"\bevents?\b",
        r"\bfestival\w*\b",
        r"\bkonferenz\w*\b",
        r"\bconference\w*\b",
        r"\btickets?\b",
        r"\brabatt\w*\b",
        r"\bdiscount code\b",
        r"\bmit dem code\b",
        r"\banmeld\w*\b",
        r"\bbewerbungsphase\w*\b",
        r"\bspeaker\w*\b",
        r"\bkeynote\w*\b",
        r"\bpanels?\b",
        r"\bnetworking\b",
        r"\bcountdown\b",
        r"\bvormerken\b",
    )
    event_signal_count = sum(
        bool(re.search(pattern, normalized)) for pattern in event_signal_patterns
    )
    has_date = bool(re.search(r"\b\d{1,2}\.\s*[a-z]+(?:\s+20\d{2})?\b", normalized))
    if event_signal_count < 2 and not (event_signal_count and has_date):
        return False

    direct_business_fact = re.search(
        r"\b(gegrundet|grundet|grundete|grundeten|entwickel\w*|betreib\w*|"
        r"betrieben|betrieb(?!\s+(?:der|des|einer?|von)\s+"
        r"(?:konferenz|festival|events?)\b)|sammel\w*|eingesammelt|"
        r"investier\w*|ubernimmt|ubernahm|ubernommen|kauf\w*|fusionier\w*|"
        r"erwirtschaft\w*|umsatz|finanzierungsrunde|wachst|wachsen|wuchs|gewachsen|"
        r"expandier\w*|kooperier\w*|bewert\w*)\b"
        r"|\bbeschaftigt\b.{0,50}\b(?:\d+|mitarbeit(?:er|ende|ern)?)\b"
        r"|\bhat\b.{0,30}\b\d+\s+mitarbeit(?:er|ende|ern)?\b"
        r"|\bstell(?:t|te|ten)\b.{0,50}\b\d+.{0,25}\bmitarbeit(?:er|ende|ern)?\b"
        r".{0,20}\bein\b"
        r"|\bstell(?:t|te|ten)\b.{0,60}\bhardware\w*\b.{0,20}\bher\b"
        r"|\b(?:erziel\w*|verzeichn\w*|mach\w*)\b.{0,60}"
        r"\b(?:umsatz|gewinn|profit|erlos)\w*\b"
        r"|\b(?:gewann|gewinnt|gewonnen)\b.{0,60}"
        r"\b(?:kund\w*|unternehmenskund\w*|neukund\w*|geschaftskund\w*|auftrag\w*)\b"
        r"|\beroffn\w*\b.{0,60}\b(?:standort|buro|werk|filiale)\w*\b"
        r"|\bverkauft\b(?!\s+(?:eintritts)?tickets?\b)"
        r"|\b\w+-grunder\w*\b"
        r"|\b(?:mitgrunder|grunder|ceo|cfo|coo|cto|cpo|geschaftsfuhrer)\w*\b"
        r"\s+(?:von|bei|der|des)\b",
        normalized,
    )
    product_business_fact = re.search(
        r"\b(?:bietet|bot|vertreib\w*|liefer\w*|produzier\w*|launch\w*)\b.{0,100}"
        r"\b\w*(?:saas|plattform|software|hardware|technologie|losung|produkt|dienstleistung)\w*\b"
        r"|\b\w*(?:saas|plattform|software|hardware|technologie|losung|produkt|dienstleistung)\w*\b"
        r".{0,100}\b(?:kund\w*|unternehmenskund\w*|neukund\w*|geschaftskund\w*|"
        r"nutzer\w*|unternehmen)\b",
        normalized,
    )
    generic_event_product = re.search(
        r"\b(?:die|eine|diese|unsere|das|ein|dieses|unser)\s+"
        r"(?:konferenz|conference|festival|event)\b(?!-).{0,80}"
        r"\b(?:bietet|bot|ist|stellt|produziert|zeigt)\b.{0,100}"
        r"\b\w*(?:plattform|software|hardware|technologie|losung|produkt|dienstleistung)\w*\b",
        normalized,
    )
    if direct_business_fact is None and generic_event_product is None:
        direct_business_fact = product_business_fact
    receives_funding = re.search(
        r"\b(erhalt\w*|erhielt|bekomm\w*|bekam|sichert\w*|sicherte|nahm|nimmt|"
        r"warb|wirbt|holte|holt|floss\w*)\b.{0,80}"
        r"\b(million|euro|geld|investor|finanz|kapital|investment)\w*\b"
        r"|\b(million|geld|investor|finanz|kapital|investment)\w*\b.{0,80}"
        r"\b(erhalt\w*|erhielt|bekomm\w*|bekam|sichert\w*|sicherte|nahm|nimmt|"
        r"warb|wirbt|holte|holt|floss\w*)\b",
        normalized,
    )
    discount_offer = re.search(
        r"\b(?:bekomm\w*\s+ihr|sichert\s+euch)\b.{0,100}\brabatt\w*\b"
        r"|\brabatt\w*\b.{0,100}\b(?:bekomm\w*\s+ihr|sichert\s+euch)\b",
        normalized,
    )
    if discount_offer:
        receives_funding = None
    return direct_business_fact is None and receives_funding is None


def _remaining_promotion_markers(
    blocks: list[_ArticleBlock],
    *,
    tags: list[str],
) -> list[str]:
    markers = _remaining_text_markers(_render_article_blocks(blocks))
    for block in blocks:
        if block.kind == "heading" and (reason := _heading_removal_reason(block.text)):
            markers.append(reason)
        elif block.kind == "paragraph" and (
            reason := _individual_removal_reason(
                block,
                tags=tags,
            )
        ):
            markers.append(reason)
    return _dedupe(markers)


def _remaining_text_markers(text: str) -> list[str]:
    normalized = _fold_for_match(text)
    signatures = (
        ("startup_jobs_promotion", "startup-jobs"),
        ("job_of_the_day_advertisement", "unser job des tages"),
        ("startup_radar_promotion", "in unserem newsletter startup-radar"),
        ("startupticker_closing_promotion", "was ist zuletzt sonst passiert?"),
        ("sponsorship_disclosure", "diese rubrik wird unterstutzt von"),
    )
    return [name for name, signature in signatures if signature in normalized]


def _removal_record(
    block: _ArticleBlock,
    reason: str,
    *,
    block_index: int,
    heading: str | None,
) -> ArticleContentRemoval:
    preview = block.text if len(block.text) <= 500 else f"{block.text[:497]}..."
    return ArticleContentRemoval(
        reason=reason,
        block_index=block_index,
        heading=heading,
        text_chars=len(block.text),
        text_preview=preview,
    )


def _is_primary_eventtipp(primary_type: str | None) -> bool:
    if not primary_type:
        return False
    return _fold_for_match(primary_type).lstrip("#").strip() == "eventtipp"


def _is_deutsche_startups_url(value: str) -> bool:
    host = urlparse(value).netloc.casefold().removeprefix("www.")
    return host == _DEUTSCHE_STARTUPS_HOST


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("\xad", "").split())


def _fold_for_match(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", _normalize_text(value))
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_marks.casefold()


def _article_cleaning_log_detail(article: ArticleIn) -> str:
    if not article.cleaning:
        return f"text_chars={len(article.text)}, tags={len(article.tags)}"
    return (
        f"text_chars={len(article.text)}, tags={len(article.tags)}, "
        f"primary_type={article.primary_type or 'unknown'}, "
        f"container={article.cleaning.selected_container}, "
        f"removed={len(article.cleaning.removed_blocks)}, "
        f"remaining_markers={len(article.cleaning.remaining_promotion_markers)}"
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
