import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.models.extraction import ArticleIn, ExtractedEntity, ExtractionResult


def article_fields(
    article: ArticleIn,
    *,
    article_index: int | None = None,
    article_total: int | None = None,
    completed_count: int | None = None,
    remaining: int | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "url": article.url,
        "article_title": truncate(article.title, 140),
    }
    if article_index is not None:
        fields["article_index"] = article_index
    if article_total is not None:
        fields["article_total"] = article_total
    if completed_count is not None:
        fields["completed_count"] = completed_count
    if remaining is not None:
        fields["remaining"] = remaining
    return fields


def extraction_summary(extraction: ExtractionResult) -> str:
    groups = [
        ("startups", extraction.startups),
        ("investors", extraction.investors),
        ("people", extraction.people),
        ("topics", extraction.topics),
        ("companies", extraction.companies),
    ]
    counts = ", ".join(f"{name}={len(items)}" for name, items in groups)
    extracted_names = [_entity_group_summary(name, items, limit=5) for name, items in groups]
    extracted_names = [part for part in extracted_names if part]
    summary = f"entities={extraction.entity_count()} ({counts}); "
    summary += f"relationships={len(extraction.relationships)}"
    if extracted_names:
        summary += "; extracted: " + "; ".join(extracted_names)
    return summary


def raw_model_output_summary(data: Mapping[str, Any]) -> str:
    parts = []
    for key in ("startups", "investors", "people", "topics", "companies"):
        values = data.get(key)
        if isinstance(values, Sequence) and not isinstance(values, str):
            parts.append(_raw_group_summary(key, values))
    for key in ("relationships",):
        values = data.get(key)
        if isinstance(values, Sequence) and not isinstance(values, str):
            parts.append(f"{key}={len(values)}")
    return "; ".join(part for part in parts if part) or "empty extraction object"


def json_preview(data: Any, max_chars: int) -> str:
    try:
        rendered = json.dumps(data, ensure_ascii=False, default=str)
    except TypeError:
        rendered = str(data)
    return truncate(rendered, max_chars)


def truncate(value: Any, max_chars: int = 240) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= max_chars else f"{text[: max_chars - 3]}..."


def _entity_group_summary(name: str, entities: Sequence[ExtractedEntity], *, limit: int) -> str:
    names = [entity.name for entity in entities[:limit]]
    if not names:
        return ""
    suffix = "" if len(entities) <= len(names) else f" +{len(entities) - len(names)} more"
    return f"{name}=[{', '.join(names)}{suffix}]"


def _raw_group_summary(name: str, values: Sequence[Any]) -> str:
    names: list[str] = []
    for value in values[:5]:
        if isinstance(value, Mapping):
            raw_name = value.get("name") or value.get("startup") or value.get("source_name")
            if raw_name:
                names.append(str(raw_name))
    suffix = "" if len(values) <= len(names) else f" +{len(values) - len(names)} more"
    if names:
        return f"{name}={len(values)} ({', '.join(names)}{suffix})"
    return f"{name}={len(values)}"
