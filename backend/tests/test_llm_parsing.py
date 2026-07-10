import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.models.extraction import (
    ArticleIn,
    RawEntityRecord,
    RawExtractionResult,
    RawRelationshipRecord,
)
from app.prompts.extraction import (
    build_article_prompt_input,
)
from app.services.llm import (
    LLMExtractionService,
    _attach_extraction_to_span,
    _merge_gleaning_pass,
    _parse_extraction_response,
    _raw_to_extraction_result,
    _raw_to_extraction_result_with_diagnostics,
    parse_extraction_output,
)


def test_extraction_trace_exposes_evidence_for_accepted_and_rejected_facts(
    monkeypatch,
) -> None:
    article = ArticleIn(
        url="https://example.test/nova",
        title="Nova",
        source_name="deutsche-startups.de",
        text=("Nova entwickelt Kliniksoftware. Fund X investiert in Nova."),
    )
    raw = RawExtractionResult(
        entities=[
            RawEntityRecord(
                name="Nova",
                entity_type="startup",
                evidence_status="stated",
                description="Nova entwickelt Kliniksoftware.",
                evidence="Nova entwickelt Kliniksoftware.",
            ),
            RawEntityRecord(
                name="Fund X",
                entity_type="investor",
                evidence_status="stated",
                description="Fund X ist ein Investor.",
                evidence="Fund X ist ein Investor.",
            ),
        ],
        relationships=[
            RawRelationshipRecord(
                source="Fund X",
                target="Nova",
                rel_type="INVESTED_IN",
                evidence_status="stated",
                keywords="Finanzierung",
                description="Fund X investiert in Nova.",
                evidence="Fund X investiert in Nova.",
            ),
        ],
    )

    class FakeSpan:
        inputs = None
        outputs = None
        attributes = None

        def set_inputs(self, inputs) -> None:
            self.inputs = inputs

        def set_outputs(self, outputs) -> None:
            self.outputs = outputs

        def set_attributes(self, attributes) -> None:
            self.attributes = attributes

    span = FakeSpan()
    monkeypatch.setattr(
        "app.services.llm.mlflow.get_current_active_span",
        lambda: span,
    )
    parser = getattr(
        _parse_extraction_response,
        "__wrapped__",
        _parse_extraction_response,
    )

    parsed = parser(raw, article, raw_text="raw model output")
    result = parsed["extraction"]
    evidence_validation = parsed["evidence_validation"]

    assert result.startups[0].source.evidence == ("Nova entwickelt Kliniksoftware.")
    assert result.relationships[0].evidence == "Fund X investiert in Nova."
    facts = evidence_validation["facts"]
    rejected_entity = next(fact for fact in facts if fact["fact"].get("name") == "Fund X")
    assert rejected_entity["evidence_check"] == "not_in_article"
    assert rejected_entity["accepted"] is False
    assert rejected_entity["rejection_reasons"] == ["evidence_not_in_article"]
    assert rejected_entity["recovery"]["mode"] == "restored"
    assert rejected_entity["recovery"]["evidence"] == "Fund X investiert in Nova."
    assert [investor.name for investor in result.investors] == ["Fund X"]
    assert result.investors[0].description == "Fund X investiert in Nova."
    assert result.investors[0].source.evidence == "Fund X investiert in Nova."
    rejected_relationship = next(
        fact for fact in facts if fact["fact"].get("relationship_type") == "INVESTED_IN"
    )
    assert rejected_relationship["evidence"] == "Fund X investiert in Nova."
    assert rejected_relationship["evidence_check"] == "matched"
    assert rejected_relationship["accepted"] is True
    assert rejected_relationship["rejection_reasons"] == []
    _attach_extraction_to_span(result, evidence_validation=evidence_validation)

    assert span.outputs["extraction"]["relationships"][0]["evidence"] == (
        "Fund X investiert in Nova."
    )
    assert span.outputs["evidence_validation"]["facts"] == facts
    assert span.attributes["evidence_facts_accepted"] == 2
    assert span.attributes["evidence_facts_rejected"] == 1


def test_parse_extraction_output_reads_entities_and_relationships() -> None:
    raw = json.dumps(
        {
            "entities": [
                {
                    "name": "SAP",
                    "entity_type": "company",
                    "type_basis": "fallback",
                    "evidence_status": "stated",
                    "description": "SAP ist der Käufer.",
                    "evidence_ref": "B001.S001",
                },
                {
                    "name": "Prior Labs",
                    "entity_type": "startup",
                    "type_basis": "explicit",
                    "evidence_status": "stated",
                    "description": "Prior Labs ist ein KI-Startup.",
                    "evidence_ref": "B001.S001",
                },
            ],
            "relationships": [
                {
                    "source": "SAP",
                    "target": "Prior Labs",
                    "relationship_type": "ACQUIRED",
                    "evidence_status": "stated",
                    "keywords": "Übernahme",
                    "evidence_ref": "B001.S001",
                }
            ],
        }
    )

    parsed = parse_extraction_output(raw)

    assert [
        (
            entity.name,
            entity.entity_type,
            entity.type_basis,
            entity.evidence_status,
            entity.evidence_ref,
        )
        for entity in parsed.entities
    ] == [
        (
            "SAP",
            "company",
            "fallback",
            "stated",
            "B001.S001",
        ),
        (
            "Prior Labs",
            "startup",
            "explicit",
            "stated",
            "B001.S001",
        ),
    ]
    assert len(parsed.relationships) == 1
    relationship = parsed.relationships[0]
    assert relationship.source == "SAP"
    assert relationship.target == "Prior Labs"
    assert relationship.rel_type == "ACQUIRED"
    assert relationship.evidence_status == "stated"
    assert relationship.keywords == "Übernahme"
    assert relationship.description == ""
    assert relationship.evidence_ref == "B001.S001"


@pytest.mark.parametrize(
    "payload",
    [
        {"entities": []},
        {"entities": [], "relationships": [], "extra": True},
        {
            "entities": [
                {
                    "name": "Nova",
                    "entity_type": "invalid",
                    "type_basis": "explicit",
                    "evidence_status": "stated",
                    "description": "Nova",
                    "evidence_ref": "B001.S001",
                }
            ],
            "relationships": [],
        },
    ],
)
def test_parse_extraction_output_rejects_missing_extra_and_invalid_fields(payload) -> None:
    with pytest.raises(ValidationError):
        parse_extraction_output(json.dumps(payload))


@pytest.mark.asyncio
async def test_extraction_uses_strict_schema_cap_and_restores_full_evidence() -> None:
    article = ArticleIn(
        url="https://example.test/nova",
        title="Nova",
        text="Nova ist ein Startup.",
    )
    response_json = json.dumps(
        {
            "entities": [
                {
                    "name": "Nova",
                    "entity_type": "startup",
                    "type_basis": "explicit",
                    "evidence_status": "stated",
                    "description": "Nova ist ein Startup.",
                    "evidence_ref": "B001.S001",
                }
            ],
            "relationships": [],
        }
    )

    class Completions:
        kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=response_json, refusal=None),
                    )
                ],
                usage=None,
            )

    completions = Completions()
    service = LLMExtractionService(
        Settings(
            openai_api_key="test-key",
            openai_model="gpt-test",
            llm_gleaning_passes=0,
            mlflow_use_prompt_registry=False,
        )
    )
    service._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result, raw, _usage, _audit, _evidence = await service._extract_with_openai(
        article,
        system_prompt="Return JSON.",
        user_prompt="Extract.",
    )

    assert raw == response_json
    assert result.startups[0].source.evidence == article.text
    response_format = completions.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "refusal"),
    [("length", None), ("stop", "Cannot process this article.")],
)
async def test_truncated_or_refused_extraction_is_not_retried(
    finish_reason: str,
    refusal: str | None,
) -> None:
    article = ArticleIn(
        url="https://example.test/nova",
        title="Nova",
        text="Nova ist ein Startup.",
    )

    class Completions:
        calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=finish_reason,
                        message=SimpleNamespace(content="{}", refusal=refusal),
                    )
                ],
                usage=None,
            )

    completions = Completions()
    service = LLMExtractionService(
        Settings(
            openai_api_key="test-key",
            openai_model="gpt-test",
            llm_gleaning_passes=0,
            llm_retry_attempts=3,
            mlflow_use_prompt_registry=False,
        )
    )
    service._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    traced = getattr(
        service._extract_article_traced,
        "__wrapped__",
        service._extract_article_traced,
    )

    with pytest.raises(RuntimeError, match="stopped without retry"):
        await traced(service, article)

    assert completions.calls == 1


@pytest.mark.asyncio
async def test_empty_gleaning_result_stops_additional_paid_passes(monkeypatch) -> None:
    service = llm_policy_service(
        openai_temperature=0.0,
        openai_seed=None,
        llm_gleaning_passes=3,
    )
    service._client = object()
    service.logger = SimpleNamespace(
        debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    responses = iter(
        [
            '{"entities":[],"relationships":[]}',
            '{"entities":[],"relationships":[]}',
        ]
    )
    calls = 0

    async def fake_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(service, "_call_llm_raw", fake_call)
    article = ArticleIn(
        url="https://example.test/empty",
        title="Keine Meldung",
        text="Heute gibt es keine neue Startup-Meldung.",
    )

    result = await service.extract_with_gleaning(article)

    assert result.entities == []
    assert result.relationships == []
    assert calls == 2


def test_raw_records_convert_to_typed_extraction_and_report_dropped_relationships() -> None:
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
                evidence=article.text,
            ),
            RawEntityRecord(
                name="Prior Labs",
                entity_type="startup",
                evidence_status="stated",
                description="Prior Labs ist ein junges KI-Startup.",
                evidence=article.text,
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
                evidence=article.text,
            ),
            RawRelationshipRecord(
                source="SAP",
                target="Missing Startup",
                rel_type="ACQUIRED",
                evidence_status="stated",
                keywords="Uebernahme",
                description="SAP kauft Missing Startup.",
                evidence=article.text,
            ),
            RawRelationshipRecord(
                source="SAP",
                target="Prior Labs",
                rel_type="BOUGHT",
                evidence_status="stated",
                keywords="Uebernahme",
                description="SAP kauft Prior Labs.",
                evidence=article.text,
            ),
        ],
    )

    result, diagnostics = _raw_to_extraction_result_with_diagnostics(raw, article)

    assert [company.name for company in result.companies] == ["SAP"]
    assert [startup.name for startup in result.startups] == ["Prior Labs"]
    assert len(result.relationships) == 1
    relationship = result.relationships[0]
    assert relationship.source_name == "SAP"
    assert relationship.source_type == "Company"
    assert relationship.target_name == "Prior Labs"
    assert relationship.target_type == "Startup"
    assert relationship.type == "ACQUIRED"
    assert relationship.evidence == article.text
    assert diagnostics["relationships_dropped"] == 2
    assert diagnostics["relationships_missing_target_endpoint"] == 1
    assert diagnostics["relationship_validation_errors"] == 1


def test_relationship_typing_accepts_only_one_contract_valid_endpoint_pair() -> None:
    def convert(source: str, source_types: list[str]):
        article = ArticleIn(
            url=f"https://example.test/{source.lower().replace(' ', '-')}",
            title="Planetary financing",
            source_name="example.test",
            text=f"{source} invested in Planetary.",
        )
        entities = [
            RawEntityRecord(
                name=source,
                entity_type=source_type,
                evidence_status="stated",
                description=article.text,
                evidence=article.text,
            )
            for source_type in source_types
        ]
        entities.append(
            RawEntityRecord(
                name="Planetary",
                entity_type="startup",
                evidence_status="stated",
                description=article.text,
                evidence=article.text,
            )
        )
        raw = RawExtractionResult(
            entities=entities,
            relationships=[
                RawRelationshipRecord(
                    source=source,
                    target="Planetary",
                    rel_type="INVESTED_IN",
                    evidence_status="stated",
                    keywords="investment",
                    description=article.text,
                    evidence=article.text,
                )
            ],
        )
        return _raw_to_extraction_result_with_diagnostics(raw, article)

    unique, _diagnostics = convert(
        "AgriFoodTech Venture Alliance",
        ["investor", "topic"],
    )
    assert [
        (relationship.source_type, relationship.target_type)
        for relationship in unique.relationships
    ] == [("Investor", "Startup")]

    invalid, invalid_diagnostics = convert(
        "AgriFoodTech Venture Alliance",
        ["topic"],
    )
    assert invalid.relationships == []
    assert invalid_diagnostics["fact_evidence_checks"][-1]["rejection_reasons"] == [
        "invalid_relationship_signature"
    ]

    ambiguous, ambiguous_diagnostics = convert("Acme", ["startup", "investor"])
    assert ambiguous.relationships == []
    assert ambiguous_diagnostics["fact_evidence_checks"][-1]["rejection_reasons"] == [
        "ambiguous_endpoint_types"
    ]


def test_conversion_rejects_missing_or_non_verbatim_evidence() -> None:
    article = ArticleIn(
        url="https://example.test/bayshore",
        title="Investoren finanzieren bayshore",
        source_name="deutsche-startups.de",
        text=(
            "Der Berliner Geldgeber Earlybird Venture Capital investiert "
            "6,9 Millionen Euro in bayshore."
        ),
    )
    raw = RawExtractionResult(
        entities=[
            RawEntityRecord(
                name="bayshore",
                entity_type="startup",
                evidence_status="stated",
                description="bayshore ist ein Startup aus Berlin.",
                evidence="",
            ),
            RawEntityRecord(
                name="Earlybird Venture Capital",
                entity_type="investor",
                evidence_status="stated",
                description="Earlybird Venture Capital ist ein Berliner Geldgeber.",
                evidence="Earlybird Venture Capital ist ein Berliner Investor.",
            ),
        ],
        relationships=[
            RawRelationshipRecord(
                source="Earlybird Venture Capital",
                target="bayshore",
                rel_type="INVESTED_IN",
                evidence_status="stated",
                keywords="Finanzierung",
                description="Earlybird Venture Capital investiert in bayshore.",
                evidence="Earlybird Venture Capital investiert in bayshore.",
            )
        ],
    )

    result, diagnostics = _raw_to_extraction_result_with_diagnostics(raw, article)

    assert result.entity_count() == 0
    assert result.relationships == []
    assert diagnostics["entities_missing_evidence"] == 1
    assert diagnostics["entities_unmatched_evidence"] == 1
    assert diagnostics["relationships_unmatched_evidence"] == 1
    assert diagnostics["relationships_dropped"] == 1
    assert diagnostics["entity_recoveries"] == []


def test_invalid_evidence_ref_drops_only_the_affected_fact() -> None:
    article = ArticleIn(
        url="https://example.test/nova",
        title="Nova",
        text="Nova ist ein Startup. Mia Kern gründete Nova.",
    )
    raw = RawExtractionResult(
        entities=[
            RawEntityRecord(
                name="Nova",
                entity_type="startup",
                type_basis="explicit",
                evidence_status="stated",
                description="Nova ist ein Startup.",
                evidence_ref="B001.S001",
            ),
            RawEntityRecord(
                name="Mia Kern",
                entity_type="person",
                type_basis="explicit",
                evidence_status="stated",
                description="Mia Kern gründete Nova.",
                evidence_ref="B001.S999",
            ),
        ]
    )

    result, diagnostics = _raw_to_extraction_result_with_diagnostics(raw, article)

    assert [entity.name for entity in result.startups] == ["Nova"]
    assert result.people == []
    assert diagnostics["entities_unmatched_evidence"] == 1
    rejected = diagnostics["fact_evidence_checks"][1]
    assert rejected["rejection_reasons"] == ["invalid_evidence_ref:unknown_ref"]


def test_gleaning_merges_missing_founder_entities_and_relationships() -> None:
    base = RawExtractionResult(
        entities=[
            RawEntityRecord(
                name="Isar Aerospace",
                entity_type="startup",
                evidence_status="stated",
                description="Isar Aerospace ist ein Raumfahrt-Startup.",
                evidence=(
                    "Isar Aerospace wurde von Daniel Metzler und Josef Fleischmann gegruendet."
                ),
            )
        ]
    )
    extra = RawExtractionResult(
        entities=[
            RawEntityRecord(
                name="Daniel Metzler",
                entity_type="person",
                evidence_status="stated",
                description="Daniel Metzler ist Mitgruender von Isar Aerospace.",
                evidence=(
                    "Isar Aerospace wurde von Daniel Metzler und Josef Fleischmann gegruendet."
                ),
            ),
            RawEntityRecord(
                name="Josef Fleischmann",
                entity_type="person",
                evidence_status="stated",
                description="Josef Fleischmann ist Mitgruender von Isar Aerospace.",
                evidence=(
                    "Isar Aerospace wurde von Daniel Metzler und Josef Fleischmann gegruendet."
                ),
            ),
        ],
        relationships=[
            RawRelationshipRecord(
                source="Isar Aerospace",
                target="Daniel Metzler",
                rel_type="FOUNDED_BY",
                evidence_status="stated",
                keywords="Gruendung",
                description="Isar Aerospace wurde von Daniel Metzler gegruendet.",
                evidence=(
                    "Isar Aerospace wurde von Daniel Metzler und Josef Fleischmann gegruendet."
                ),
            ),
            RawRelationshipRecord(
                source="Isar Aerospace",
                target="Josef Fleischmann",
                rel_type="FOUNDED_BY",
                evidence_status="stated",
                keywords="Gruendung",
                description="Isar Aerospace wurde von Josef Fleischmann gegruendet.",
                evidence=(
                    "Isar Aerospace wurde von Daniel Metzler und Josef Fleischmann gegruendet."
                ),
            ),
        ],
    )

    _merge_gleaning_pass(
        base,
        extra,
        pass_index=1,
        raw_response_chars=500,
    )

    assert [(entity.name, entity.entity_type) for entity in base.entities] == [
        ("Isar Aerospace", "startup"),
        ("Daniel Metzler", "person"),
        ("Josef Fleischmann", "person"),
    ]
    assert [
        (relationship.source, relationship.rel_type, relationship.target)
        for relationship in base.relationships
    ] == [
        ("Isar Aerospace", "FOUNDED_BY", "Daniel Metzler"),
        ("Isar Aerospace", "FOUNDED_BY", "Josef Fleischmann"),
    ]
    article = ArticleIn(
        url="https://example.test/isar",
        title="Isar Aerospace bekommt 270 Millionen",
        source_name="deutsche-startups.de",
        text="Isar Aerospace wurde von Daniel Metzler und Josef Fleischmann gegruendet.",
    )
    extraction = _raw_to_extraction_result(base, article)
    assert [(person.name) for person in extraction.people] == [
        "Daniel Metzler",
        "Josef Fleischmann",
    ]
    assert [(rel.source_name, rel.type, rel.target_name) for rel in extraction.relationships] == [
        ("Isar Aerospace", "FOUNDED_BY", "Daniel Metzler"),
        ("Isar Aerospace", "FOUNDED_BY", "Josef Fleischmann"),
    ]


def test_cached_registry_prompt_receives_title_type_and_complete_clean_body() -> None:
    class CachedPrompt:
        variables = {"input_text"}
        name = "article_extraction"
        version = 7
        uri = "prompts:/article_extraction/7"

        def __init__(self) -> None:
            self.input_text: str | None = None

        def format(self, *, input_text: str) -> list[dict[str, str]]:
            self.input_text = input_text
            return [
                {"role": "system", "content": "registry system"},
                {"role": "user", "content": f"registry user\n{input_text}"},
            ]

    article = ArticleIn(
        url="https://example.test/ticker",
        title="Nova sammelt zwölf Millionen Euro ein",
        source_name="deutsche-startups.de",
        primary_type="#StartupTicker",
        text="## Nova\n" + "Nova entwickelt eine Plattform für Industriekunden. " * 400,
    )
    cached_prompt = CachedPrompt()
    service = llm_policy_service(openai_temperature=0.0, openai_seed=None)
    service.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    service._extraction_prompt = cached_prompt

    _system_prompt, user_prompt, _metadata = service._render_extraction_prompts(article)

    expected_input = build_article_prompt_input(article)
    assert cached_prompt.input_text == expected_input
    assert expected_input in user_prompt
    assert "[B001.S400] Nova entwickelt eine Plattform für Industriekunden." in user_prompt


def llm_policy_service(
    *,
    openai_temperature: float,
    openai_seed: int | None,
    llm_gleaning_passes: int = 1,
) -> LLMExtractionService:
    service = object.__new__(LLMExtractionService)
    service.settings = Settings(
        openai_model="gpt-test",
        openai_temperature=openai_temperature,
        openai_seed=openai_seed,
        llm_gleaning_passes=llm_gleaning_passes,
        mlflow_use_prompt_registry=False,
    )
    service._extraction_prompt = None
    service._gleaning_prompt = None
    return service
