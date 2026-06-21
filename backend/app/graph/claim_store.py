import json
from datetime import UTC, datetime
from typing import Any

from app.db.neo4j import Neo4jClient
from app.graph.common import (
    DIRECTIONAL_CONFLICT_RELATIONSHIPS,
    DOMAIN_RELATIONSHIPS,
    RELATIONSHIPS,
    jsonable,
    public_relationship_properties,
)
from app.models.extraction import ExtractedRelationship


INVERSE_CONFLICT_QUERY = """
MATCH (left_source)-[left]->(left_target)
MATCH (left_target)-[right]->(left_source)
WHERE type(left) IN $directional_relationships
  AND type(right) = type(left)
  AND elementId(left) < elementId(right)
  AND coalesce(left.lifecycle_status, "supported") = "supported"
  AND coalesce(right.lifecycle_status, "supported") = "supported"
  AND coalesce(left.review_status, "unreviewed") <> "rejected"
  AND coalesce(right.review_status, "unreviewed") <> "rejected"
SET left.review_status = "needs_review",
    right.review_status = "needs_review",
    left.review_reasons = CASE
      WHEN "inverse_direction" IN coalesce(left.review_reasons, []) THEN left.review_reasons
      ELSE coalesce(left.review_reasons, []) + ["inverse_direction"]
    END,
    right.review_reasons = CASE
      WHEN "inverse_direction" IN coalesce(right.review_reasons, []) THEN right.review_reasons
      ELSE coalesce(right.review_reasons, []) + ["inverse_direction"]
    END
RETURN count(left) AS conflicts
"""

TRANSACTION_CONFLICT_QUERY = """
MATCH (first)-[acquired:ACQUIRED]->(second)
MATCH (merge_source)-[merged:MERGED_WITH]->(merge_target)
WHERE (
    (first = merge_source AND second = merge_target)
    OR (first = merge_target AND second = merge_source)
  )
  AND coalesce(acquired.lifecycle_status, "supported") = "supported"
  AND coalesce(merged.lifecycle_status, "supported") = "supported"
  AND coalesce(acquired.review_status, "unreviewed") <> "rejected"
  AND coalesce(merged.review_status, "unreviewed") <> "rejected"
SET acquired.review_status = "needs_review",
    merged.review_status = "needs_review",
    acquired.review_reasons = CASE
      WHEN "competing_transaction_type" IN coalesce(acquired.review_reasons, [])
      THEN acquired.review_reasons
      ELSE coalesce(acquired.review_reasons, []) + ["competing_transaction_type"]
    END,
    merged.review_reasons = CASE
      WHEN "competing_transaction_type" IN coalesce(merged.review_reasons, [])
      THEN merged.review_reasons
      ELSE coalesce(merged.review_reasons, []) + ["competing_transaction_type"]
    END
RETURN count(acquired) AS conflicts
"""


class ClaimStore:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

    async def initialize_claim_state(self) -> int:
        query = """
        MATCH (source)-[r]->(target)
        WHERE type(r) IN $claim_relationships
          AND NOT "Article" IN labels(source)
          AND NOT "Article" IN labels(target)
        SET r.active_article_urls = coalesce(r.active_article_urls, r.article_urls, []),
            r.lifecycle_status = coalesce(r.lifecycle_status, "supported"),
            r.review_status = coalesce(r.review_status, "unreviewed"),
            r.review_reasons = coalesce(r.review_reasons, []),
            r.support_changed = coalesce(r.support_changed, false)
        RETURN count(r) AS initialized
        """
        async with self.neo4j.session() as session:
            result = await session.run(query, claim_relationships=DOMAIN_RELATIONSHIPS)
            record = await result.single()
            return int(record["initialized"]) if record else 0

    async def refresh_claim_conflicts(self) -> int:
        async with self.neo4j.session() as session:
            inverse = await session.run(
                INVERSE_CONFLICT_QUERY,
                directional_relationships=DIRECTIONAL_CONFLICT_RELATIONSHIPS,
            )
            inverse_record = await inverse.single()
            transactions = await session.run(TRANSACTION_CONFLICT_QUERY)
            transaction_record = await transactions.single()
        return int(inverse_record["conflicts"] if inverse_record else 0) + int(
            transaction_record["conflicts"] if transaction_record else 0
        )

    async def node_claims(self, node_id: str) -> dict[str, Any] | None:
        node_query = "MATCH (node {id: $node_id}) RETURN node LIMIT 1"
        claim_query = """
        MATCH (source)-[r]->(target)
        WHERE (source.id = $node_id OR target.id = $node_id)
          AND type(r) IN $relationships
        RETURN source, r, target
        """
        async with self.neo4j.session() as session:
            node_result = await session.run(node_query, node_id=node_id)
            node_record = await node_result.single()
            if not node_record:
                return None
            result = await session.run(
                claim_query,
                node_id=node_id,
                relationships=list(RELATIONSHIPS),
            )
            rows = [record async for record in result]

        claims: list[dict[str, Any]] = []
        mentions: list[dict[str, Any]] = []
        for record in rows:
            source = record["source"]
            relationship = record["r"]
            target = record["target"]
            rel_type = relationship.type
            source_id = source.get("id") or source.element_id
            target_id = target.get("id") or target.element_id
            properties = public_relationship_properties(dict(relationship))
            active_article_urls_value = properties.get("active_article_urls")
            if active_article_urls_value is None:
                active_article_urls_value = properties.get("article_urls") or []
            active_article_urls = _unique_strings(active_article_urls_value)
            assertions = _claim_assertions(properties.get("provenance"))
            payload = {
                "edge_id": relationship.element_id,
                "relationship": rel_type,
                "direction": (
                    "undirected"
                    if rel_type == "MERGED_WITH"
                    else "outgoing"
                    if source_id == node_id
                    else "incoming"
                ),
                "counterparty": _node_reference(target if source_id == node_id else source),
                "lifecycle_status": properties.get("lifecycle_status", "supported"),
                "review_status": properties.get("review_status", "unreviewed"),
                "review_reasons": properties.get("review_reasons", []),
                "review_comment": properties.get("review_comment"),
                "reviewed_by": properties.get("reviewed_by"),
                "reviewed_at": properties.get("reviewed_at"),
                "review_history": _review_history(properties.get("review_history")),
                "support_changed": properties.get("support_changed", False),
                "active_support_count": len(active_article_urls),
                "active_article_urls": active_article_urls,
                "source_articles": _source_articles(
                    assertions=assertions,
                    active_article_urls=active_article_urls,
                    review_status=properties.get("review_status") or "unreviewed",
                    review_reasons=properties.get("review_reasons") or [],
                ),
                "assertions": assertions,
                "source_id": source_id,
                "target_id": target_id,
            }
            source_is_article = "Article" in source.labels
            target_is_article = "Article" in target.labels
            if rel_type in DOMAIN_RELATIONSHIPS and not source_is_article and not target_is_article:
                claims.append(payload)
            elif rel_type in {"MENTIONS", "HAS_TOPIC"}:
                mentions.append(payload)

        claims.sort(key=_claim_sort_key)
        mentions.sort(key=_claim_sort_key)
        return {
            "node_id": node_id,
            "claims": claims,
            "mentions": mentions,
        }

    async def review_claim(
        self,
        *,
        source_id: str,
        relationship: str,
        target_id: str,
        decision: str,
        comment: str | None,
        reviewer: str | None,
    ) -> bool:
        review_event = json.dumps(
            {
                "decision": decision,
                "comment": comment,
                "reviewer": reviewer,
                "reviewed_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        query = """
        MATCH (source {id: $source_id})-[r]->(target {id: $target_id})
        WHERE type(r) = $relationship
        SET r.review_status = $decision,
            r.review_reasons = CASE
              WHEN $decision IN ["accepted", "rejected"] THEN []
              ELSE coalesce(r.review_reasons, [])
            END,
            r.review_comment = $comment,
            r.reviewed_by = $reviewer,
            r.reviewed_at = datetime(),
            r.review_history = coalesce(r.review_history, []) + [$review_event]
        RETURN count(r) AS reviewed
        """
        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                source_id=source_id,
                relationship=relationship,
                target_id=target_id,
                decision=decision,
                comment=comment,
                reviewer=reviewer,
                review_event=review_event,
            )
            record = await result.single()
            return bool(record and record["reviewed"])


async def mark_claim_conflicts_tx(tx) -> None:
    await tx.run(
        INVERSE_CONFLICT_QUERY,
        directional_relationships=DIRECTIONAL_CONFLICT_RELATIONSHIPS,
    )
    await tx.run(TRANSACTION_CONFLICT_QUERY)


def has_valid_claim_direction(relationship: ExtractedRelationship) -> bool:
    source_type = relationship.source_type
    target_type = relationship.target_type
    if relationship.type == "INVESTED_IN":
        return source_type in {"Investor", "Company", "Person"} and target_type == "Startup"
    if relationship.type == "FOUNDED_BY":
        return source_type == "Startup" and target_type == "Person"
    if relationship.type == "EMPLOYED_BY":
        return source_type == "Person" and target_type in {"Startup", "Company"}
    if relationship.type == "ACQUIRED":
        return source_type in {"Startup", "Company"} and target_type in {"Startup", "Company"}
    if relationship.type == "HAS_TOPIC":
        return source_type != "Topic" and target_type == "Topic"
    return True


def claim_endpoint_ids(relationship_type: str, source_id: str, target_id: str) -> tuple[str, str]:
    if relationship_type == "MERGED_WITH" and target_id < source_id:
        return target_id, source_id
    return source_id, target_id


def _node_reference(node) -> dict[str, Any]:
    labels = list(node.labels)
    return {
        "id": node.get("id") or node.element_id,
        "label": node.get("name") or node.get("title") or node.get("id"),
        "type": labels[0] if labels else "Node",
    }


def _claim_assertions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    assertions: list[dict[str, Any]] = []
    for serialized in value:
        if not isinstance(serialized, str):
            continue
        try:
            record = json.loads(serialized)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        record["event"] = record.get("event") or "asserted"
        assertions.append(record)
    assertions.sort(
        key=lambda record: str(record.get("processed_at") or record.get("published_at") or ""),
        reverse=True,
    )
    return assertions


def _source_articles(
    *,
    assertions: list[dict[str, Any]],
    active_article_urls: list[str],
    review_status: str,
    review_reasons: list[str],
) -> list[dict[str, Any]]:
    active_keys = {key for url in active_article_urls if (key := _article_key(url))}
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, assertion in enumerate(assertions):
        key = (
            _article_key(assertion.get("article_url"))
            or _article_key(assertion.get("article_id"))
            or _article_key(assertion.get("article_title"))
            or f"unknown:{index}"
        )
        groups.setdefault(key, []).append(assertion)

    rows: list[dict[str, Any]] = []
    for key, article_assertions in groups.items():
        latest = article_assertions[0]
        latest_asserted = next(
            (item for item in article_assertions if item.get("event") == "asserted"),
            None,
        )
        evidence_source = (
            latest
            if latest.get("event") == "asserted" and latest.get("evidence")
            else latest_asserted
            if latest_asserted and latest_asserted.get("evidence")
            else None
        )
        status = _source_article_status(latest, key in active_keys)
        row = {
            "article_url": latest.get("article_url") or (latest_asserted or {}).get("article_url"),
            "article_title": latest.get("article_title")
            or (latest_asserted or {}).get("article_title"),
            "published_at": latest.get("published_at")
            or (latest_asserted or {}).get("published_at"),
            "latest_processed_at": latest.get("processed_at"),
            "status": status,
            "evidence": (evidence_source or {}).get("evidence"),
            "trace_url": latest.get("mlflow_trace_url"),
            "processing_count": len(article_assertions),
            "review_source": _is_review_source(status, review_status, review_reasons),
        }
        rows.append(jsonable(row))

    rows.sort(key=_source_article_sort_key)
    return rows


def _source_article_status(assertion: dict[str, Any], active: bool) -> str:
    event = assertion.get("event")
    if event == "not_reproduced":
        return "no_longer_current"
    if event == "direction_changed":
        return "direction_changed"
    if event == "asserted" and active:
        return "current"
    return "historical"


def _is_review_source(status: str, review_status: str, review_reasons: list[str]) -> bool:
    if review_status != "needs_review":
        return False
    if status == "no_longer_current":
        return "not_reproduced_same_article" in review_reasons
    if status == "direction_changed":
        return "direction_changed_same_article" in review_reasons
    return False


def _source_article_sort_key(article: dict[str, Any]) -> tuple[int, float]:
    status = str(article.get("status") or "")
    if article.get("review_source"):
        rank = 0
    elif status == "current":
        rank = 1
    elif status in {"no_longer_current", "direction_changed"}:
        rank = 2
    else:
        rank = 3
    return rank, -_timestamp_value(
        article.get("latest_processed_at") or article.get("published_at")
    )


def _timestamp_value(value: Any) -> float:
    if not value:
        return 0
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        normalized = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return 0


def _article_key(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text or None


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _review_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    events: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            try:
                record = json.loads(item)
            except json.JSONDecodeError:
                continue
        elif isinstance(item, dict):
            record = item
        else:
            continue
        if isinstance(record, dict):
            events.append(jsonable(record))
    return events


def _claim_sort_key(claim: dict[str, Any]) -> tuple[int, str, str]:
    review_rank = 0 if claim["review_status"] == "needs_review" else 1
    counterpart = claim["counterparty"]
    return review_rank, str(claim["relationship"]), str(counterpart["label"])
