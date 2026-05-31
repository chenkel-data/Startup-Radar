import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger

from app.core.config import Settings


class TerminalFormatter(logging.Formatter):
    RESET = "\033[0m"
    BOLD_RED = "\033[1;31m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_MAGENTA = "\033[1;35m"
    STAGE_WIDTH = 31
    STAGES = {
        ("ingestion", "ingest"): (None, None, "Ingestion"),
        ("ingestion", "scraping"): (1, 5, "Collect articles"),
        ("scraping", "collect"): (1, 5, "Collect articles"),
        ("scraping", "feed"): (1, 5, "Collect article links"),
        ("scraping", "listing"): (1, 5, "Collect article links"),
        ("scraping", "article_fetch"): (1, 5, "Fetch article text"),
        ("scraping", "article_parse"): (1, 5, "Fetch article text"),
        ("ingestion", "resolution_registry"): (2, 5, "Load entity registry"),
        ("resolution", "registry_load"): (2, 5, "Load entity registry"),
        ("ingestion", "extraction"): (3, 5, "Extract entities"),
        ("extraction", "article_extraction"): (3, 5, "Extract entities"),
        ("extraction", "llm"): (3, 5, "Extract entities"),
        ("extraction", "llm_configuration"): (3, 5, "Extract entities"),
        ("extraction", "llm_request"): (3, 5, "Extract entities"),
        ("extraction", "llm_retry"): (3, 5, "Extract entities"),
        ("extraction", "llm_failure"): (3, 5, "Extract entities"),
        ("extraction", "llm_output"): (3, 5, "Extract entities"),
        ("extraction", "llm_output_raw"): (3, 5, "Extract entities"),
        ("ingestion", "article_process"): (4, 5, "Normalize entities"),
        ("ingestion", "evidence_gate"): (4, 5, "Validate evidence"),
        ("ingestion", "resolution"): (4, 5, "Normalize entities"),
        ("resolution", "entity_resolution"): (4, 5, "Normalize entities"),
        ("ingestion", "graph_ingest"): (5, 5, "Write graph"),
        ("schema", "apply"): (None, None, "Schema"),
        ("startup", "database"): (None, None, "Startup"),
        ("search", "fulltext"): (None, None, "Search"),
        ("http", "request"): (None, None, "HTTP"),
    }
    ACTIONS = {
        "step_started": "run started",
        "step_completed": "run completed",
        "step_failed": "run failed",
        "workflow_stage_started": "stage started",
        "workflow_stage_completed": "stage completed",
        "scrape_collection_started": "collecting article sources",
        "feed_fetch_started": "fetching feed",
        "feed_links_collected": "feed links collected",
        "listing_scan_started": "scanning listing",
        "listing_page_fetch_started": "fetching listing page",
        "listing_links_collected": "listing page scanned",
        "listing_pagination_exhausted": "listing pagination exhausted",
        "listing_page_failed": "listing page failed",
        "article_links_collected": "article links ready",
        "article_fetch_started": "fetching article",
        "article_fetch_completed": "article fetched",
        "article_fetch_skipped": "article skipped",
        "article_fetch_failed": "article fetch failed",
        "articles_scraped": "article fetch stage completed",
        "article_parse_skipped": "article skipped during parsing",
        "entity_registry_loaded": "existing entities loaded",
        "article_extraction_started": "extracting article",
        "article_extraction_completed": "article extraction completed",
        "article_extraction_failed": "article extraction permanently failed",
        "article_processing_started": "preparing article for graph",
        "evidence_gate_applied": "evidence gate applied",
        "entity_resolution_started": "resolving article entities",
        "entity_resolution_completed": "article entities resolved",
        "graph_ingest_started": "writing article to graph",
        "article_ingested": "article written to graph",
        "article_ingestion_failed": "article graph write failed",
        "ingest_completed": "ingestion completed",
        "llm_request_started": "waiting for LLM response",
        "llm_extraction_retry": "LLM request failed; retry scheduled",
        "llm_extraction_failed": "LLM extraction failed",
        "llm_extraction_missing_api_key": "missing OpenAI API key",
        "article_extraction_requested": "article prepared for LLM",
        "llm_extraction_completed": "LLM extraction completed",
        "llm_output_received": "LLM output received",
        "llm_raw_output_preview": "raw LLM output preview",
        "entity_merged_exact": "entity merged by exact alias",
        "entity_merged_heuristic": "entity merged by heuristic match",
        "entity_merged_embedding": "entity merged by embedding match",
        "entity_embedding_no_match": "embedding match not found",
        "entity_created": "new entity created",
        "http_request": "HTTP request",
        "neo4j_connected": "Neo4j connected",
        "neo4j_unavailable": "Neo4j unavailable",
        "schema_applied": "schema applied",
        "geography_topics_deleted": "geography topics deleted",
        "topic_categories_cleared": "topic categories cleared",
        "fulltext_search_failed": "full-text search failed",
    }
    FIELD_ORDER = (
        "article_title",
        "task_id",
        "mode",
        "model",
        "entity_type",
        "duration_ms",
        "count",
        "failed_count",
        "detail",
        "url",
        "error",
    )
    FIELD_LABELS = {
        "article_title": "title",
        "detail": "detail",
        "duration_ms": "duration",
        "entity_type": "type",
        "failed_count": "failed",
    }
    FIELD_LIMITS = {
        "article_title": 120,
        "detail": 520,
        "url": 180,
        "error": 300,
    }

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        stage = _stage_label(record, self.STAGES)
        message = self.ACTIONS.get(
            record.getMessage(),
            _clean(record.getMessage()).replace("_", " "),
        )

        line = f"{timestamp} {record.levelname:<7} {stage:<{self.STAGE_WIDTH}} | {message}"

        fields = []
        progress = _progress(record)
        if progress:
            fields.append(progress)
        attempt = _attempt(record)
        if attempt:
            fields.append(attempt)

        for field in self.FIELD_ORDER:
            value = getattr(record, field, None)
            if _is_empty(value):
                continue
            if field == "failed_count" and progress:
                continue
            label = self.FIELD_LABELS.get(field, field)
            if field == "duration_ms":
                rendered = f"{label}={value}ms"
            else:
                rendered = f"{label}={_truncate(value, self.FIELD_LIMITS.get(field, 420))}"
            fields.append(rendered)

        if fields:
            line = f"{line} | " + " | ".join(fields)

        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return self._colorize(record, line)

    def _colorize(self, record: logging.LogRecord, line: str) -> str:
        color = None
        message = record.getMessage()
        if message == "llm_extraction_retry":
            color = self.BOLD_MAGENTA
        elif record.levelno >= logging.ERROR:
            color = self.BOLD_RED
        elif record.levelno >= logging.WARNING:
            color = self.BOLD_YELLOW
        return f"{color}{line}{self.RESET}" if color else line


def setup_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(settings.log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    if settings.log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(event)s %(component)s "
            "%(workflow_step)s %(duration_ms)s %(count)s %(task_id)s %(url)s "
            "%(entity_type)s %(detail)s %(article_title)s %(article_index)s "
            "%(article_total)s %(page_index)s %(page_total)s %(completed_count)s "
            "%(remaining)s %(attempt_index)s %(attempt_total)s %(retry_delay_seconds)s "
            "%(failed_count)s %(mode)s %(model)s %(error)s"
        )
    else:
        formatter = TerminalFormatter()
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class ComponentLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        call_extra = kwargs.get("extra", {})
        kwargs["extra"] = {**self.extra, **call_extra}
        return msg, kwargs


def get_logger(component: str) -> logging.LoggerAdapter:
    logger = logging.getLogger(component)
    return ComponentLoggerAdapter(logger, {"component": component})


def _context(record: logging.LogRecord) -> str:
    event = getattr(record, "event", None)
    step = getattr(record, "workflow_step", None)
    parts = [_clean(value) for value in (event, step) if not _is_empty(value)]
    deduped = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return "/".join(deduped)


def _stage_label(
    record: logging.LogRecord,
    stages: dict[tuple[str | None, str | None], tuple[int | None, int | None, str]],
) -> str:
    event = getattr(record, "event", None)
    step = getattr(record, "workflow_step", None)
    stage = stages.get((event, step)) or stages.get((event, None)) or stages.get((None, step))
    if stage:
        index, total, label = stage
        if index is not None and total is not None:
            return f"Step {index}/{total} {label}"
        return label

    context = _context(record)
    if context:
        return context.replace("/", " / ")
    return _clean(getattr(record, "component", record.name))


def _progress(record: logging.LogRecord) -> str | None:
    article_index = getattr(record, "article_index", None)
    article_total = getattr(record, "article_total", None)
    completed_count = getattr(record, "completed_count", None)
    remaining = getattr(record, "remaining", None)
    failed_count = getattr(record, "failed_count", None)
    page_index = getattr(record, "page_index", None)
    page_total = getattr(record, "page_total", None)

    if not _is_empty(article_index) and not _is_empty(article_total):
        value = f"article={article_index}/{article_total}"
        if not _is_empty(completed_count):
            value = f"{value}, done={completed_count}/{article_total}"
        if not _is_empty(failed_count):
            value = f"{value}, failed={failed_count}/{article_total}"
    elif not _is_empty(completed_count) and not _is_empty(article_total):
        value = f"articles={completed_count}/{article_total} done"
        if not _is_empty(failed_count):
            value = f"{value}, failed={failed_count}/{article_total}"
    elif not _is_empty(page_index) and not _is_empty(page_total):
        value = f"page={page_index}/{page_total}"
    else:
        value = None

    if value and not _is_empty(remaining):
        value = f"{value}, left={remaining}"
    return value


def _attempt(record: logging.LogRecord) -> str | None:
    attempt_index = getattr(record, "attempt_index", None)
    attempt_total = getattr(record, "attempt_total", None)
    retry_delay_seconds = getattr(record, "retry_delay_seconds", None)

    value = None
    if not _is_empty(attempt_index) and not _is_empty(attempt_total):
        value = f"attempt={attempt_index}/{attempt_total}"
    if value and not _is_empty(retry_delay_seconds):
        value = f"{value}, retry_in={retry_delay_seconds}s"
    elif not value and not _is_empty(retry_delay_seconds):
        value = f"retry_in={retry_delay_seconds}s"
    return value


def _truncate(value: Any, limit: int = 420) -> str:
    text = _clean(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _clean(value: Any) -> str:
    return " ".join(str(value).split())


def _is_empty(value: Any) -> bool:
    return value is None or value == ""
