import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from neo4j.graph import Node, Relationship

from app.models.extraction import GraphEdge, GraphNode, GraphResponse


LABELS = {
    "Startup": "Startup",
    "Investor": "Investor",
    "Person": "Person",
    "Topic": "Topic",
    "Company": "Company",
    "Article": "Article",
    "Source": "Source",
    "ProfileEvidence": "ProfileEvidence",
}

RELATIONSHIPS = {
    "INVESTED_IN",
    "FOUNDED_BY",
    "EMPLOYED_BY",
    "PARTNERED_WITH",
    "MERGED_WITH",
    "HAS_TOPIC",
    "FROM_SOURCE",
    "ACQUIRED",
    "MENTIONS",
}

LANDSCAPE_RELATIONSHIPS = [
    "INVESTED_IN",
    "FOUNDED_BY",
    "EMPLOYED_BY",
    "PARTNERED_WITH",
    "MERGED_WITH",
    "ACQUIRED",
    "HAS_TOPIC",
]
ARTICLE_FEED_RELATIONSHIPS = ["HAS_TOPIC", "FROM_SOURCE", "MENTIONS"]
DOMAIN_RELATIONSHIPS = [
    "INVESTED_IN",
    "FOUNDED_BY",
    "EMPLOYED_BY",
    "PARTNERED_WITH",
    "MERGED_WITH",
    "ACQUIRED",
    "HAS_TOPIC",
]
PROFILE_ENTITY_LABELS = ["Startup", "Investor", "Company", "Person", "Topic"]
DIRECTIONAL_CONFLICT_RELATIONSHIPS = [
    "INVESTED_IN",
    "FOUNDED_BY",
    "EMPLOYED_BY",
    "ACQUIRED",
]
EVIDENCE_MAX_CHARS = 1200
PROFILE_EVIDENCE_MAX_CHARS = 1600


def safe_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Unsupported label: {label}")
    return LABELS[label]


def sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def profile_evidence_id(*parts: str) -> str:
    return f"profile-evidence:{sha1('|'.join(parts))}"


def is_geography_topic_node(node: Node) -> bool:
    return "Topic" in node.labels and is_geography_category(node.get("category"))


def is_geography_category(value: Any) -> bool:
    return str(value or "").casefold() == "geography"


def graph_response(nodes: list[Node], rels: list[Relationship]) -> GraphResponse:
    node_models: dict[str, GraphNode] = {}
    for node in nodes:
        if not node:
            continue
        if is_geography_topic_node(node):
            continue
        node_id = node.get("id") or node.element_id
        labels = list(node.labels)
        node_type = labels[0] if labels else "Node"
        title = node.get("name") or node.get("title") or node_id
        properties = public_node_properties(dict(node))
        if node_type == "Topic":
            properties.pop("category", None)
        node_models[node_id] = GraphNode(
            id=node_id,
            label=title,
            type=node_type,
            properties=properties,
        )

    edge_models: dict[str, GraphEdge] = {}
    for rel in rels:
        if not rel:
            continue
        if rel.get("review_status") == "rejected":
            continue
        source_id = rel.start_node.get("id") or rel.start_node.element_id
        target_id = rel.end_node.get("id") or rel.end_node.element_id
        if source_id not in node_models or target_id not in node_models:
            continue
        edge_id = rel.element_id
        edge_models[edge_id] = GraphEdge(
            id=edge_id,
            source=source_id,
            target=target_id,
            label=rel.type,
            properties=public_relationship_properties(dict(rel)),
        )
    return GraphResponse(nodes=list(node_models.values()), edges=list(edge_models.values()))


def public_node_properties(properties: dict[str, Any]) -> dict[str, Any]:
    result = jsonable(properties)
    if "description_curation_history" in result:
        result["description_curation_history"] = _json_object_list(
            result.get("description_curation_history")
        )
    result.pop("embedding", None)
    result.pop("confidence", None)
    return result


def public_relationship_properties(properties: dict[str, Any]) -> dict[str, Any]:
    result = jsonable(properties)
    result.pop("confidence", None)
    return result


def jsonable(properties: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in properties.items():
        result[key] = jsonable_value(value)
    return result


def jsonable_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if hasattr(value, "to_native"):
        native = value.to_native()
        if isinstance(native, datetime):
            return native.astimezone(UTC).isoformat()
        return native.isoformat() if hasattr(native, "isoformat") else native
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if isinstance(value, list):
        return [jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable_value(item) for key, item in value.items()}
    return value


def _json_object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            rows.append(jsonable(item))
            continue
        if not isinstance(item, str):
            continue
        try:
            parsed = json.loads(item)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(jsonable(parsed))
    return rows
