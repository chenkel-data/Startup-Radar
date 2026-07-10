from __future__ import annotations

from collections.abc import Mapping


ORGANIZATION_TYPES: frozenset[str] = frozenset({"Startup", "Investor", "Company"})
NON_TOPIC_TYPES: frozenset[str] = frozenset({*ORGANIZATION_TYPES, "Person"})

RELATIONSHIP_SIGNATURES: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "INVESTED_IN": (frozenset({*ORGANIZATION_TYPES, "Person"}), ORGANIZATION_TYPES),
    "ACQUIRED": (frozenset({*ORGANIZATION_TYPES, "Person"}), ORGANIZATION_TYPES),
    "MERGED_WITH": (ORGANIZATION_TYPES, ORGANIZATION_TYPES),
    "FOUNDED_BY": (ORGANIZATION_TYPES, frozenset({"Person"})),
    "EMPLOYED_BY": (frozenset({"Person"}), ORGANIZATION_TYPES),
    "PARTNERED_WITH": (ORGANIZATION_TYPES, ORGANIZATION_TYPES),
    "HAS_TOPIC": (NON_TOPIC_TYPES, frozenset({"Topic"})),
}


def is_organization_type(entity_type: str) -> bool:
    return entity_type in ORGANIZATION_TYPES


def has_valid_relationship_signature(
    relationship_type: str,
    source_type: str,
    target_type: str,
) -> bool:
    signature = RELATIONSHIP_SIGNATURES.get(relationship_type)
    if signature is None:
        return False
    source_types, target_types = signature
    return source_type in source_types and target_type in target_types
