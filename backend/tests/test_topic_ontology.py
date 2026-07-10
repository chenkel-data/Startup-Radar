import pytest
from pydantic import ValidationError

from app.models.extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from app.topic_ontology import (
    TopicOntology,
    get_topic_ontology,
)


def test_production_ontology_is_closed_and_resolves_only_reviewed_aliases() -> None:
    ontology = get_topic_ontology()

    assert ontology.ontology_version == "source-only-broad-topics-v1"
    ai = ontology.lookup("AI")
    assert ai is not None
    assert ai.concept_id == "ai"
    assert ai.definition
    assert ai.semantic_boundary
    assert ontology.lookup("KI").name == "AI"
    assert ontology.lookup("Robotik").name == "Robotics"
    assert ontology.lookup("ConTech").name == "ConstructionTech"
    assert ontology.lookup("AI Automation Platform") is None
    assert ontology.lookup("Data Analytics") is None


def test_ontology_rejects_alias_collisions_between_concepts() -> None:
    with pytest.raises(ValidationError, match="collision"):
        TopicOntology.model_validate(
            {
                "ontology_version": "test-v1",
                "topics": [
                    {
                        "concept_id": "ai",
                        "name": "AI",
                        "aliases": ["KI"],
                        "definition": "Artificial intelligence.",
                        "semantic_boundary": "Explicit AI only.",
                    },
                    {
                        "concept_id": "other-ai",
                        "name": "Other AI",
                        "aliases": ["KI"],
                        "definition": "Another concept.",
                        "semantic_boundary": "Never overlaps.",
                    },
                ],
            }
        )


def test_enforcement_canonicalizes_deduplicates_and_rejects_unknown_topics() -> None:
    ontology = get_topic_ontology()
    extraction = ExtractionResult(
        startups=[ExtractedEntity(name="Nova", evidence_status="stated")],
        topics=[
            ExtractedEntity(name="KI", evidence_status="stated"),
            ExtractedEntity(name="AI", evidence_status="attributed"),
            ExtractedEntity(name="Data Analytics", evidence_status="stated"),
            ExtractedEntity(name="FinTech", evidence_status="stated"),
        ],
        relationships=[
            ExtractedRelationship(
                type="HAS_TOPIC",
                source_name="Nova",
                source_type="Startup",
                target_name="KI",
                target_type="Topic",
                evidence_status="stated",
                evidence="Nova nutzt KI.",
            ),
            ExtractedRelationship(
                type="HAS_TOPIC",
                source_name="Nova",
                source_type="Startup",
                target_name="AI",
                target_type="Topic",
                evidence_status="attributed",
                evidence="Nova beschreibt seine KI.",
            ),
            ExtractedRelationship(
                type="HAS_TOPIC",
                source_name="Nova",
                source_type="Startup",
                target_name="Data Analytics",
                target_type="Topic",
                evidence_status="stated",
                evidence="Nova analysiert Daten.",
            ),
        ],
    )

    report = ontology.enforce(extraction)

    assert [topic.name for topic in extraction.topics] == ["AI"]
    assert extraction.topics[0].description == ontology.lookup("AI").definition
    assert [
        (relationship.source_name, relationship.target_name)
        for relationship in extraction.relationships
    ] == [("Nova", "AI")]
    assert extraction.relationships[0].evidence == "Nova nutzt KI."
    assert report.topics_rejected == 1
    assert report.duplicate_topics_merged == 1
