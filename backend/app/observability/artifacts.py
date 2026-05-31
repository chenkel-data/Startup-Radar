"""Helpers that build the JSON / Markdown artifacts attached to one ingestion Run."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any


def build_ingestion_summary_md(
    *,
    job_run_id: str,
    source_name: str,
    metrics: dict[str, float],
    failures_sample: Iterable[dict[str, Any]] = (),
) -> str:
    """Human-readable summary for the Run's `ingestion_summary.md` artifact."""
    now = datetime.now(UTC).isoformat()
    lines = [
        f"# Ingestion summary — `{source_name}`",
        "",
        f"- **Run id**: `{job_run_id}`",
        f"- **Generated**: {now}",
        "",
        "## Metrics",
        "",
    ]
    if metrics:
        lines.extend(f"- `{key}`: **{_fmt(value)}**" for key, value in metrics.items())
    else:
        lines.append("_no metrics recorded_")

    failure_rows = list(failures_sample)
    if failure_rows:
        lines += ["", "## Sample failures", ""]
        for row in failure_rows[:5]:
            url = row.get("article_url") or row.get("url") or "(unknown)"
            stage = row.get("stage") or "?"
            error = (row.get("error") or "")[:200]
            lines.append(f"- **{stage}** — `{url}`\n  - {error}")

    return "\n".join(lines) + "\n"


def build_dedup_report(
    *,
    outcome_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-entity resolution outcomes by method and entity type."""
    by_method: dict[str, int] = {"exact": 0, "fuzzy": 0, "embedding": 0, "new": 0}
    by_type: dict[str, dict[str, int]] = {}
    top_fuzzy: list[dict[str, Any]] = []
    for row in outcome_rows:
        method = row.get("method", "new")
        by_method[method] = by_method.get(method, 0) + 1
        entity_type = row.get("entity_type") or row.get("label") or "Entity"
        bucket = by_type.setdefault(entity_type, {"exact": 0, "fuzzy": 0, "embedding": 0, "new": 0})
        bucket[method] = bucket.get(method, 0) + 1
        if method == "fuzzy" and row.get("similarity_min") is not None:
            top_fuzzy.append(
                {
                    "candidate": row.get("candidate"),
                    "canonical": row.get("canonical"),
                    "similarity_min": row.get("similarity_min"),
                    "article_url": row.get("article_url"),
                }
            )
    top_fuzzy.sort(key=lambda r: r.get("similarity_min") or 0.0)
    total = sum(by_method.values()) or 1
    dedup_rate = (by_method["exact"] + by_method["fuzzy"] + by_method["embedding"]) / total
    return {
        "total_outcomes": sum(by_method.values()),
        "by_method": by_method,
        "by_type": by_type,
        "dedup_rate": round(dedup_rate, 4),
        "top_borderline_fuzzy": top_fuzzy[:10],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.6f}"
        return f"{value:.3f}"
    return str(value)
