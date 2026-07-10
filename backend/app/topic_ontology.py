from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from app.models.extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    NormalizedEntity,
    SourceAttribution,
    strongest_evidence_status,
    strongest_type_basis,
)

TOPIC_ONTOLOGY_PATH = Path(__file__).with_name("topic_ontology.json")
EXPECTED_TOPIC_COUNT = 30
_CONCEPT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def normalize_topic_key(value: str) -> str:
    """Normalize harmless spelling differences without semantic matching."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.casefold().replace("&", " and ").replace("-", " ")
    normalized = re.sub(r"[^\w\s+#.]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip(" .")


class TopicConcept(BaseModel):
    model_config = ConfigDict(frozen=True)

    concept_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    definition: str
    semantic_boundary: str

    @field_validator("concept_id", "name", "definition", "semantic_boundary")
    @classmethod
    def require_non_empty_value(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Topic ontology values must not be empty")
        return cleaned

    @field_validator("concept_id")
    @classmethod
    def validate_concept_id(cls, value: str) -> str:
        if not _CONCEPT_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "Topic concept_id must contain lowercase letters, digits, and single hyphens"
            )
        return value

    @field_validator("aliases")
    @classmethod
    def clean_aliases(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Topic aliases must be unique within a concept")
        return cleaned


class TopicOntology(BaseModel):
    model_config = ConfigDict(frozen=True)

    ontology_version: str
    topics: list[TopicConcept]
    _by_key: dict[str, TopicConcept] = PrivateAttr(default_factory=dict)
    _by_id: dict[str, TopicConcept] = PrivateAttr(default_factory=dict)
    _content_hash: str = PrivateAttr(default="")

    @field_validator("ontology_version")
    @classmethod
    def require_version(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Topic ontology version must not be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_unique_concepts(self) -> TopicOntology:
        concept_ids: set[str] = set()
        keys: dict[str, str] = {}
        for concept in self.topics:
            if concept.concept_id in concept_ids:
                raise ValueError(f"Duplicate topic concept_id: {concept.concept_id}")
            concept_ids.add(concept.concept_id)
            for candidate in (concept.name, *concept.aliases):
                key = normalize_topic_key(candidate)
                if not key:
                    raise ValueError(
                        f"Topic name or alias normalizes to an empty value: {candidate!r}"
                    )
                owner = keys.get(key)
                if owner is not None and owner != concept.concept_id:
                    raise ValueError(
                        f"Topic name/alias collision for {candidate!r}: "
                        f"{owner} and {concept.concept_id}"
                    )
                keys[key] = concept.concept_id
        return self

    def model_post_init(self, __context: Any) -> None:
        self._by_id = {concept.concept_id: concept for concept in self.topics}
        self._by_key = {
            normalize_topic_key(candidate): concept
            for concept in self.topics
            for candidate in (concept.name, *concept.aliases)
        }
        canonical_payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._content_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

    @property
    def content_hash(self) -> str:
        return self._content_hash

    def lookup(self, value: str) -> TopicConcept | None:
        return self._by_key.get(normalize_topic_key(value))

    def concept(self, concept_id: str) -> TopicConcept:
        return self._by_id[concept_id]

    def prompt_catalog(self) -> str:
        blocks = []
        for concept in self.topics:
            aliases = "; ".join(concept.aliases) if concept.aliases else "none"
            blocks.append(
                "\n".join(
                    (
                        concept.name,
                        f"Aliases: {aliases}",
                        f"Definition: {concept.definition}",
                        f"Boundary: {concept.semantic_boundary}",
                    )
                )
            )
        return "\n\n".join(blocks)

    def enforce(self, extraction: ExtractionResult) -> TopicOntologyReport:
        report = TopicOntologyReport(
            ontology_version=self.ontology_version,
            ontology_hash=self.content_hash,
            topic_candidates=len(extraction.topics),
            has_topic_candidates=sum(
                relationship.type == "HAS_TOPIC" for relationship in extraction.relationships
            ),
        )

        topics_by_concept: dict[str, ExtractedEntity] = {}
        for topic in extraction.topics:
            concept = self.lookup(topic.name)
            if concept is None:
                report.topics_rejected += 1
                report.rejections.append(
                    {
                        "kind": "topic_entity",
                        "input_name": topic.name,
                        "reason": "unknown_topic",
                    }
                )
                continue
            if topic.name != concept.name:
                report.topic_names_canonicalized += 1
            canonical = topic.model_copy(
                update={
                    "name": concept.name,
                    "aliases": list(concept.aliases),
                    "type_basis": "explicit",
                    "description": concept.definition,
                }
            )
            previous = topics_by_concept.get(concept.concept_id)
            if previous is None:
                topics_by_concept[concept.concept_id] = canonical
            else:
                report.duplicate_topics_merged += 1
                topics_by_concept[concept.concept_id] = _merge_topic_entities(
                    previous,
                    canonical,
                )

        relationships: list[ExtractedRelationship] = []
        has_topic_by_key: dict[tuple[str, str, str], int] = {}
        supported_concepts: set[str] = set()
        relationship_status_by_concept: dict[str, ExtractedRelationship] = {}

        for relationship in extraction.relationships:
            if relationship.type != "HAS_TOPIC":
                relationships.append(relationship)
                continue

            concept = self.lookup(relationship.target_name)
            if concept is None:
                report.has_topic_rejected += 1
                report.rejections.append(
                    {
                        "kind": "has_topic_relationship",
                        "input_name": relationship.target_name,
                        "source_name": relationship.source_name,
                        "reason": "unknown_topic",
                    }
                )
                continue
            if relationship.target_name != concept.name:
                report.topic_names_canonicalized += 1
            canonical = relationship.model_copy(
                update={
                    "target_name": concept.name,
                    "target_type": "Topic",
                }
            )
            relationship_key = (
                relationship.source_type,
                _normalized_entity_key(relationship.source_name),
                concept.concept_id,
            )
            existing_index = has_topic_by_key.get(relationship_key)
            if existing_index is not None:
                report.duplicate_has_topic_merged += 1
                relationships[existing_index] = _stronger_relationship(
                    relationships[existing_index],
                    canonical,
                )
            else:
                has_topic_by_key[relationship_key] = len(relationships)
                relationships.append(canonical)
            supported_concepts.add(concept.concept_id)
            previous_relationship = relationship_status_by_concept.get(concept.concept_id)
            relationship_status_by_concept[concept.concept_id] = (
                canonical
                if previous_relationship is None
                else _stronger_relationship(previous_relationship, canonical)
            )

        canonical_topics: list[ExtractedEntity] = []
        for concept in self.topics:
            if concept.concept_id not in supported_concepts:
                if concept.concept_id in topics_by_concept:
                    report.orphan_topics_removed += 1
                continue
            topic = topics_by_concept.get(concept.concept_id)
            supporting_relationship = relationship_status_by_concept[concept.concept_id]
            if topic is None:
                topic = ExtractedEntity(
                    name=concept.name,
                    aliases=list(concept.aliases),
                    type_basis="explicit",
                    evidence_status=supporting_relationship.evidence_status,
                    evidence_status_defaulted=(supporting_relationship.evidence_status_defaulted),
                    description=concept.definition,
                    source=SourceAttribution(evidence=supporting_relationship.evidence),
                )
            elif (
                strongest_evidence_status(
                    topic.evidence_status,
                    supporting_relationship.evidence_status,
                )
                != topic.evidence_status
            ):
                topic = topic.model_copy(
                    update={
                        "evidence_status": supporting_relationship.evidence_status,
                        "evidence_status_defaulted": (
                            supporting_relationship.evidence_status_defaulted
                        ),
                    }
                )
            canonical_topics.append(topic)

        extraction.topics = canonical_topics
        extraction.relationships = relationships
        report.topics_kept = len(canonical_topics)
        report.has_topic_kept = sum(
            relationship.type == "HAS_TOPIC" for relationship in relationships
        )
        return report

    def normalized_entity(
        self,
        concept: TopicConcept,
        extracted: ExtractedEntity,
    ) -> NormalizedEntity:
        return NormalizedEntity(
            id=f"topic:{concept.concept_id}",
            label="Topic",
            canonical_name=concept.name,
            name=concept.name,
            aliases=list(concept.aliases),
            primary_type_basis="explicit",
            observed_types=["Topic"],
            type_conflict=False,
            evidence_status=extracted.evidence_status,
            description=concept.definition,
            descriptions=[concept.definition],
            concept_id=concept.concept_id,
            ontology_version=self.ontology_version,
            ontology_hash=self.content_hash,
            semantic_boundary=concept.semantic_boundary,
        )


@dataclass
class TopicOntologyReport:
    ontology_version: str
    ontology_hash: str
    topic_candidates: int = 0
    topics_kept: int = 0
    topics_rejected: int = 0
    has_topic_candidates: int = 0
    has_topic_kept: int = 0
    has_topic_rejected: int = 0
    topic_names_canonicalized: int = 0
    orphan_topics_removed: int = 0
    duplicate_topics_merged: int = 0
    duplicate_has_topic_merged: int = 0
    rejections: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ontology_version": self.ontology_version,
            "ontology_hash": self.ontology_hash,
            "topic_candidates": self.topic_candidates,
            "topics_kept": self.topics_kept,
            "topics_rejected": self.topics_rejected,
            "has_topic_candidates": self.has_topic_candidates,
            "has_topic_kept": self.has_topic_kept,
            "has_topic_rejected": self.has_topic_rejected,
            "topic_names_canonicalized": self.topic_names_canonicalized,
            "orphan_topics_removed": self.orphan_topics_removed,
            "duplicate_topics_merged": self.duplicate_topics_merged,
            "duplicate_has_topic_merged": self.duplicate_has_topic_merged,
            "rejections": list(self.rejections),
        }


def load_topic_ontology(
    path: Path = TOPIC_ONTOLOGY_PATH,
    *,
    expected_topic_count: int | None = EXPECTED_TOPIC_COUNT,
) -> TopicOntology:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ontology = TopicOntology.model_validate(payload)
    if expected_topic_count is not None and len(ontology.topics) != expected_topic_count:
        raise ValueError(
            f"Expected {expected_topic_count} production topics, found {len(ontology.topics)}"
        )
    return ontology


@lru_cache(maxsize=1)
def get_topic_ontology() -> TopicOntology:
    return load_topic_ontology()


def _merge_topic_entities(
    left: ExtractedEntity,
    right: ExtractedEntity,
) -> ExtractedEntity:
    strongest_status = strongest_evidence_status(
        left.evidence_status,
        right.evidence_status,
    )
    strongest_basis = strongest_type_basis(left.type_basis, right.type_basis)
    preferred = left if strongest_status == left.evidence_status else right
    return preferred.model_copy(
        update={
            "type_basis": strongest_basis,
            "evidence_status": strongest_status,
            "aliases": list(left.aliases),
            "description": left.description,
        }
    )


def _stronger_relationship(
    left: ExtractedRelationship,
    right: ExtractedRelationship,
) -> ExtractedRelationship:
    strongest_status = strongest_evidence_status(
        left.evidence_status,
        right.evidence_status,
    )
    return left if strongest_status == left.evidence_status else right


def _normalized_entity_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()
