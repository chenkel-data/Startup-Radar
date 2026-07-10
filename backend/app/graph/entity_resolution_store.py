from typing import Any

from app.db.neo4j import Neo4jClient
from app.graph.claim_store import mark_claim_conflicts_tx
from app.graph.common import DOMAIN_RELATIONSHIPS, RELATIONSHIPS, jsonable, safe_label
from app.services.entity_resolution import NameNormalizer


ALIAS_ENTITY_LABELS = {"Startup", "Investor", "Company", "Person"}
_MERGE_RELATIONSHIPS = RELATIONSHIPS | {"HAS_PROFILE_EVIDENCE"}
_EVIDENCE_STATUS_RANK = {"unsure": 0, "attributed": 1, "stated": 2}
_EXPLICIT_REVIEW_STATUSES = {"accepted", "rejected"}


class AmbiguousAliasError(ValueError):
    pass


class EntityResolutionStore:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

    async def list_entities(self, label: str) -> list[dict[str, Any]]:
        label = safe_label(label)
        query = f"""
        MATCH (n:{label})
        RETURN n.id AS id,
               n.name AS name,
               n.canonical_name AS canonical_name,
               n.aliases AS aliases,
               n.primary_type_basis AS primary_type_basis,
               n.observed_types AS observed_types,
               n.type_conflict AS type_conflict,
               n.evidence_status AS evidence_status,
               n.description AS description,
               n.descriptions AS descriptions,
               n.embedding AS embedding
        LIMIT 50000
        """
        async with self.neo4j.session() as session:
            result = await session.run(query)
            return [jsonable(dict(record)) async for record in result]

    async def has_relationship_to_named_entity(
        self,
        target_id: str,
        candidate_label: str,
        candidate_names: list[str],
    ) -> bool:
        label = safe_label(candidate_label)
        name_variants = {
            variant for name in candidate_names for variant in _name_match_variants(name) if variant
        }
        key_variants = {key for name in candidate_names if (key := NameNormalizer.key(name, label))}
        candidate_ids = {
            f"{label.lower()}:{NameNormalizer.slug(name)}"
            for name in candidate_names
            if NameNormalizer.display(name)
        }
        if not name_variants and not key_variants and not candidate_ids:
            return False

        query = f"""
        MATCH (target {{id: $target_id}})
        MATCH (candidate:{label})
        WHERE coalesce(candidate.id, "") <> $target_id
        WITH target, candidate,
             [raw IN [coalesce(candidate.name, ""), coalesce(candidate.canonical_name, "")]
               + coalesce(candidate.aliases, []) | replace(toLower(raw), "-", " ")] AS names
        WHERE coalesce(candidate.id, "") IN $candidate_ids
           OR any(name IN names WHERE name IN $name_variants OR name IN $key_variants)
           OR any(name IN names WHERE any(key IN $key_variants
                WHERE size(key) >= 4 AND name STARTS WITH key + " "))
        MATCH (target)-[rel]-(candidate)
        WHERE type(rel) IN $relationships
          AND coalesce(rel.lifecycle_status, "supported") = "supported"
          AND coalesce(rel.review_status, "unreviewed") <> "rejected"
        RETURN count(rel) > 0 AS blocked
        LIMIT 1
        """
        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                target_id=target_id,
                candidate_ids=list(candidate_ids),
                name_variants=list(name_variants),
                key_variants=list(key_variants),
                relationships=list(DOMAIN_RELATIONSHIPS),
            )
            record = await result.single()
            return bool(record and record["blocked"])

    async def save_entity_embedding(self, entity_id: str, embedding: list[float]) -> None:
        query = "MATCH (n {id: $id}) SET n.embedding = $embedding"
        async with self.neo4j.session() as session:
            await session.run(query, id=entity_id, embedding=embedding)

    async def add_alias(self, node_id: str, alias: str) -> dict[str, Any] | None:
        async with self.neo4j.session() as session:
            return await session.execute_write(_add_alias_tx, node_id, alias)

    async def remove_alias(self, node_id: str, alias: str) -> dict[str, Any] | None:
        async with self.neo4j.session() as session:
            return await session.execute_write(_remove_alias_tx, node_id, alias)

    async def find_nearest_entity(
        self,
        label: str,
        embedding: list[float],
        min_score: float,
        k: int = 5,
        exclude_ids: set[str] | None = None,
    ) -> tuple[str | None, float]:
        """Return (entity_id, score) of the closest entity via Neo4j vector index, or (None, 0)."""
        index_name = f"{label.lower()}_embedding"
        query = """
        CALL db.index.vector.queryNodes($index_name, $k, $embedding)
        YIELD node, score
        WHERE score >= $min_score
          AND NOT coalesce(node.id, "") IN $exclude_ids
        RETURN node.id AS id, score
        ORDER BY score DESC
        LIMIT 1
        """
        async with self.neo4j.session() as session:
            try:
                result = await session.run(
                    query,
                    index_name=index_name,
                    k=k,
                    embedding=embedding,
                    min_score=min_score,
                    exclude_ids=list(exclude_ids or set()),
                )
                record = await result.single()
                if record:
                    return record["id"], float(record["score"])
            except Exception:
                pass
        return None, 0.0


async def _add_alias_tx(tx, node_id: str, alias: str) -> dict[str, Any] | None:
    target_result = await tx.run(
        """
        MATCH (target {id: $node_id})
        RETURN target, labels(target) AS labels
        LIMIT 1
        """,
        node_id=node_id,
    )
    target_record = await target_result.single()
    if not target_record:
        return None

    target = target_record["target"]
    label = next(
        (value for value in target_record["labels"] if value in ALIAS_ENTITY_LABELS),
        None,
    )
    if label is None:
        raise ValueError("Aliases can only be added to startups, investors, companies, and people")

    display_alias = NameNormalizer.display(alias)
    alias_key = NameNormalizer.key(display_alias, label)
    if not alias_key:
        raise ValueError("Alias must contain a usable entity name")

    safe_entity_label = safe_label(label)
    candidates_result = await tx.run(
        f"""
        MATCH (candidate:{safe_entity_label})
        WHERE candidate.id <> $node_id
        RETURN candidate
        """,
        node_id=node_id,
    )
    candidates = [
        record["candidate"]
        async for record in candidates_result
        if _node_has_key(record["candidate"], alias_key, label)
    ]
    if len(candidates) > 1:
        raise AmbiguousAliasError(
            f'Alias "{display_alias}" matches multiple existing {label} nodes'
        )

    target_properties = dict(target)
    merged_node_ids: list[str] = []
    if candidates:
        source = candidates[0]
        source_id = str(source.get("id") or "")
        if not source_id:
            raise ValueError("Matching entity has no stable id")
        target_properties = _merge_node_properties(
            target_properties,
            dict(source),
            display_alias,
            label,
        )
        await _merge_entity_relationships_tx(
            tx,
            source_id=source_id,
            target_id=node_id,
        )
        await tx.run(
            """
            MATCH (e:ProfileEvidence)
            WHERE e.entity_id = $source_id OR e.other_entity_id = $source_id
            SET e.entity_id = CASE
                  WHEN e.entity_id = $source_id THEN $target_id ELSE e.entity_id
                END,
                e.other_entity_id = CASE
                  WHEN e.other_entity_id = $source_id THEN $target_id ELSE e.other_entity_id
                END
            """,
            source_id=source_id,
            target_id=node_id,
        )
        await tx.run(
            """
            MATCH (source {id: $source_id})
            DETACH DELETE source
            """,
            source_id=source_id,
        )
        merged_node_ids.append(source_id)
    else:
        target_properties["aliases"] = _unique_aliases(
            [*(target_properties.get("aliases") or []), display_alias],
            label,
        )

    await tx.run(
        """
        MATCH (target {id: $node_id})
        SET target = $properties,
            target.updated_at = datetime()
        """,
        node_id=node_id,
        properties=target_properties,
    )
    if merged_node_ids:
        await mark_claim_conflicts_tx(tx)

    return {
        "node_id": node_id,
        "name": str(
            target_properties.get("name") or target_properties.get("canonical_name") or node_id
        ),
        "aliases": list(target_properties.get("aliases") or []),
        "merged_node_ids": merged_node_ids,
    }


async def _remove_alias_tx(tx, node_id: str, alias: str) -> dict[str, Any] | None:
    target_result = await tx.run(
        """
        MATCH (target {id: $node_id})
        RETURN target, labels(target) AS labels
        LIMIT 1
        """,
        node_id=node_id,
    )
    target_record = await target_result.single()
    if not target_record:
        return None

    target = target_record["target"]
    label = next(
        (value for value in target_record["labels"] if value in ALIAS_ENTITY_LABELS),
        None,
    )
    if label is None:
        raise ValueError(
            "Aliases can only be removed from startups, investors, companies, and people"
        )

    alias_key = NameNormalizer.key(NameNormalizer.display(alias), label)
    if not alias_key:
        raise ValueError("Alias must contain a usable entity name")

    aliases = _unique_aliases(
        [
            value
            for value in target.get("aliases") or []
            if NameNormalizer.key(str(value or ""), label) != alias_key
        ],
        label,
    )
    await tx.run(
        """
        MATCH (target {id: $node_id})
        SET target.aliases = $aliases,
            target.updated_at = datetime()
        """,
        node_id=node_id,
        aliases=aliases,
    )
    return {
        "node_id": node_id,
        "name": str(target.get("name") or target.get("canonical_name") or node_id),
        "aliases": aliases,
        "merged_node_ids": [],
    }


async def _merge_entity_relationships_tx(
    tx,
    *,
    source_id: str,
    target_id: str,
) -> None:
    relationships_result = await tx.run(
        """
        MATCH (source {id: $source_id})-[relationship]->(neighbor)
        RETURN elementId(relationship) AS relationship_id,
               type(relationship) AS relationship_type,
               relationship,
               true AS outgoing,
               neighbor.id AS neighbor_id
        UNION ALL
        MATCH (neighbor)-[relationship]->(source {id: $source_id})
        RETURN elementId(relationship) AS relationship_id,
               type(relationship) AS relationship_type,
               relationship,
               false AS outgoing,
               neighbor.id AS neighbor_id
        """,
        source_id=source_id,
    )
    relationships = [record async for record in relationships_result]

    for record in relationships:
        relationship_id = record["relationship_id"]
        relationship_type = str(record["relationship_type"])
        neighbor_id = record["neighbor_id"]
        if relationship_type not in _MERGE_RELATIONSHIPS:
            raise ValueError(
                f"Unsupported relationship type during entity merge: {relationship_type}"
            )
        if not neighbor_id:
            raise ValueError("Cannot merge a relationship whose neighboring node has no stable id")
        if neighbor_id == target_id:
            await tx.run(
                "MATCH ()-[relationship]->() WHERE elementId(relationship) = $relationship_id DELETE relationship",
                relationship_id=relationship_id,
            )
            continue

        left_id = target_id if record["outgoing"] else neighbor_id
        right_id = neighbor_id if record["outgoing"] else target_id
        existing_result = await tx.run(
            f"""
            MATCH (left {{id: $left_id}})
            MATCH (right {{id: $right_id}})
            OPTIONAL MATCH (left)-[existing:{relationship_type}]->(right)
            RETURN existing
            LIMIT 1
            """,
            left_id=left_id,
            right_id=right_id,
        )
        existing_record = await existing_result.single()
        existing = existing_record["existing"] if existing_record else None
        properties = _merge_relationship_properties(
            dict(existing) if existing is not None else {},
            dict(record["relationship"]),
        )
        await tx.run(
            f"""
            MATCH ()-[old_relationship]->()
            WHERE elementId(old_relationship) = $relationship_id
            MATCH (left {{id: $left_id}})
            MATCH (right {{id: $right_id}})
            MERGE (left)-[merged:{relationship_type}]->(right)
            SET merged = $properties,
                merged.updated_at = datetime()
            DELETE old_relationship
            """,
            relationship_id=relationship_id,
            left_id=left_id,
            right_id=right_id,
            properties=properties,
        )


def _node_has_key(node, alias_key: str, label: str) -> bool:
    names = [
        node.get("name"),
        node.get("canonical_name"),
        *(node.get("aliases") or []),
    ]
    return any(NameNormalizer.key(str(name), label) == alias_key for name in names if name)


def _merge_node_properties(
    target: dict[str, Any],
    source: dict[str, Any],
    alias: str,
    label: str,
) -> dict[str, Any]:
    merged = dict(target)
    protected = {
        "id",
        "name",
        "canonical_name",
        "description",
        "description_source",
        "embedding",
        "created_at",
    }
    for key, value in source.items():
        if key in protected or value is None:
            continue
        current = merged.get(key)
        if isinstance(current, list) and isinstance(value, list):
            merged[key] = _unique_values([*current, *value])
        elif key == "type_conflict":
            merged[key] = bool(current) or bool(value)
        elif not _has_value(current):
            merged[key] = value

    merged["aliases"] = _unique_aliases(
        [
            *(target.get("aliases") or []),
            alias,
            source.get("name"),
            source.get("canonical_name"),
            *(source.get("aliases") or []),
        ],
        label,
    )
    merged["descriptions"] = _unique_values(
        [
            *(target.get("descriptions") or []),
            target.get("description"),
            *(source.get("descriptions") or []),
            source.get("description"),
        ]
    )
    return merged


def _merge_relationship_properties(
    target: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(target)
    for key, value in source.items():
        current = merged.get(key)
        if isinstance(current, list) and isinstance(value, list):
            merged[key] = _unique_values([*current, *value])
        elif not _has_value(current):
            merged[key] = value

    target_status = str(target.get("evidence_status") or "unsure")
    source_status = str(source.get("evidence_status") or "unsure")
    if _EVIDENCE_STATUS_RANK.get(source_status, 0) > _EVIDENCE_STATUS_RANK.get(
        target_status,
        0,
    ):
        merged["evidence_status"] = source_status
        if source.get("evidence"):
            merged["evidence"] = source["evidence"]

    if (
        target.get("lifecycle_status") == "supported"
        or source.get("lifecycle_status") == "supported"
    ):
        merged["lifecycle_status"] = "supported"
    merged["support_changed"] = bool(target.get("support_changed")) or bool(
        source.get("support_changed")
    )

    target_review = str(target.get("review_status") or "unreviewed")
    source_review = str(source.get("review_status") or "unreviewed")
    if {target_review, source_review} == _EXPLICIT_REVIEW_STATUSES:
        merged["review_status"] = "needs_review"
        merged["review_reasons"] = _unique_values(
            [
                *(merged.get("review_reasons") or []),
                "manual_entity_merge_conflict",
            ]
        )
    elif target_review in _EXPLICIT_REVIEW_STATUSES:
        merged["review_status"] = target_review
    elif source_review in _EXPLICIT_REVIEW_STATUSES:
        merged["review_status"] = source_review
        for key in ("review_comment", "reviewed_by", "reviewed_at"):
            if source.get(key) is not None:
                merged[key] = source[key]
    elif "needs_review" in {target_review, source_review}:
        merged["review_status"] = "needs_review"
    else:
        merged["review_status"] = "unreviewed"
    return merged


def _unique_aliases(values: list[Any], label: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        display = NameNormalizer.display(str(value or ""))
        key = NameNormalizer.key(display, label) if display else ""
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(display)
    return result


def _unique_values(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value is None or value == "":
            continue
        if value not in result:
            result.append(value)
    return result


def _has_value(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _name_match_variants(name: str) -> set[str]:
    display = NameNormalizer.display(name)
    lowered = display.casefold()
    hyphen_normalized = lowered.replace("-", " ")
    return {
        lowered,
        hyphen_normalized,
        " ".join(hyphen_normalized.split()),
    }
