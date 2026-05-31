from app.models.extraction import (
    ArticleIn,
    RawEntityRecord,
    RawExtractionResult,
    RawRelationshipRecord,
)
from app.prompts.extraction import COMPLETION_DELIMITER, TUPLE_DELIMITER
from app.services.llm import _raw_to_extraction_result, parse_extraction_output


def test_parse_extraction_output_reads_entities_and_relationships() -> None:
    raw = "\n".join(
        [
            f"entity{TUPLE_DELIMITER}SAP{TUPLE_DELIMITER}company"
            f"{TUPLE_DELIMITER}stated{TUPLE_DELIMITER}SAP ist der Kaeufer.",
            f"entity{TUPLE_DELIMITER}Prior Labs{TUPLE_DELIMITER}startup"
            f"{TUPLE_DELIMITER}stated{TUPLE_DELIMITER}Prior Labs ist ein KI-Startup.",
            f"relation{TUPLE_DELIMITER}SAP{TUPLE_DELIMITER}Prior Labs"
            f"{TUPLE_DELIMITER}ACQUIRED{TUPLE_DELIMITER}stated"
            f"{TUPLE_DELIMITER}Uebernahme{TUPLE_DELIMITER}"
            "SAP kauft das junge KI-Startup Prior Labs.",
            COMPLETION_DELIMITER,
        ]
    )

    parsed = parse_extraction_output(raw)

    assert [
        (entity.name, entity.entity_type, entity.evidence_status) for entity in parsed.entities
    ] == [
        ("SAP", "company", "stated"),
        ("Prior Labs", "startup", "stated"),
    ]
    assert len(parsed.relationships) == 1
    relationship = parsed.relationships[0]
    assert relationship.source == "SAP"
    assert relationship.target == "Prior Labs"
    assert relationship.rel_type == "ACQUIRED"
    assert relationship.evidence_status == "stated"
    assert relationship.keywords == "Uebernahme"
    assert relationship.description == "SAP kauft das junge KI-Startup Prior Labs."


def test_parse_extraction_output_defaults_missing_or_invalid_evidence_status_to_unsure() -> None:
    raw = "\n".join(
        [
            f"entity{TUPLE_DELIMITER}Leegle{TUPLE_DELIMITER}startup"
            f"{TUPLE_DELIMITER}Leegle ist ein LegalTech.",
            f"relation{TUPLE_DELIMITER}Christian Lindner{TUPLE_DELIMITER}Leegle"
            f"{TUPLE_DELIMITER}INVESTED_IN{TUPLE_DELIMITER}Finanzierung"
            f"{TUPLE_DELIMITER}Christian Lindner investiert in Leegle.",
            f"entity{TUPLE_DELIMITER}Cohere{TUPLE_DELIMITER}startup"
            f"{TUPLE_DELIMITER}maybe{TUPLE_DELIMITER}Cohere ist ein KI-Unternehmen.",
            COMPLETION_DELIMITER,
        ]
    )

    parsed = parse_extraction_output(raw)

    assert parsed.entities[0].name == "Leegle"
    assert parsed.entities[0].evidence_status == "unsure"
    assert parsed.entities[0].evidence_status_defaulted is True
    assert parsed.entities[1].name == "Cohere"
    assert parsed.entities[1].evidence_status == "unsure"
    assert parsed.entities[1].evidence_status_defaulted is True
    assert parsed.relationships[0].evidence_status == "unsure"
    assert parsed.relationships[0].evidence_status_defaulted is True


def test_raw_records_convert_to_typed_extraction_and_drop_unknown_endpoints() -> None:
    article = ArticleIn(
        url="https://example.test/sap-prior-labs",
        title="SAP kauft Prior Labs",
        source_name="deutsche-startups.de",
        text="SAP kauft das junge KI-Startup Prior Labs und staerkt damit sein KI-Angebot.",
    )
    raw = RawExtractionResult(
        entities=[
            RawEntityRecord(
                name="SAP",
                entity_type="company",
                evidence_status="stated",
                description="SAP ist der Kaeufer von Prior Labs.",
            ),
            RawEntityRecord(
                name="Prior Labs",
                entity_type="startup",
                evidence_status="stated",
                description="Prior Labs ist ein junges KI-Startup.",
            ),
        ],
        relationships=[
            RawRelationshipRecord(
                source="SAP",
                target="Prior Labs",
                rel_type="ACQUIRED",
                evidence_status="stated",
                keywords="Uebernahme",
                description="SAP kauft das junge KI-Startup Prior Labs.",
            ),
            RawRelationshipRecord(
                source="SAP",
                target="Missing Startup",
                rel_type="ACQUIRED",
                evidence_status="stated",
                keywords="Uebernahme",
                description="SAP kauft Missing Startup.",
            ),
        ],
    )

    result = _raw_to_extraction_result(raw, article)

    assert [company.name for company in result.companies] == ["SAP"]
    assert [startup.name for startup in result.startups] == ["Prior Labs"]
    assert len(result.relationships) == 1
    relationship = result.relationships[0]
    assert relationship.source_name == "SAP"
    assert relationship.source_type == "Company"
    assert relationship.target_name == "Prior Labs"
    assert relationship.target_type == "Startup"
    assert relationship.type == "ACQUIRED"
    assert relationship.evidence == "SAP kauft das junge KI-Startup Prior Labs."
