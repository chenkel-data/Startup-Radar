from typing import Any

from app.db.neo4j import Neo4jClient
from app.graph.common import DOMAIN_RELATIONSHIPS, jsonable, safe_label
from app.services.entity_resolution import NameNormalizer


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


def _name_match_variants(name: str) -> set[str]:
    display = NameNormalizer.display(name)
    lowered = display.casefold()
    hyphen_normalized = lowered.replace("-", " ")
    return {
        lowered,
        hyphen_normalized,
        " ".join(hyphen_normalized.split()),
    }
