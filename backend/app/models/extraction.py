from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from app.relationship_contract import has_valid_relationship_signature

EntityType = Literal["Startup", "Investor", "Person", "Topic", "Company"]
EntityTypeBasis = Literal["explicit", "contextual", "fallback", "legacy"]
EvidenceStatus = Literal["stated", "attributed", "unsure"]
RelationshipType = Literal[
    "INVESTED_IN",
    "FOUNDED_BY",
    "EMPLOYED_BY",
    "PARTNERED_WITH",
    "MERGED_WITH",
    "HAS_TOPIC",
    "ACQUIRED",
]

ADMITTED_EVIDENCE_STATUSES: frozenset[EvidenceStatus] = frozenset({"stated", "attributed"})
_EVIDENCE_RANK: dict[EvidenceStatus, int] = {
    "unsure": 0,
    "attributed": 1,
    "stated": 2,
}
_TYPE_BASIS_RANK: dict[EntityTypeBasis, int] = {
    "fallback": 0,
    "contextual": 1,
    "legacy": 2,
    "explicit": 3,
}


def strongest_evidence_status(left: EvidenceStatus, right: EvidenceStatus) -> EvidenceStatus:
    return left if _EVIDENCE_RANK[left] >= _EVIDENCE_RANK[right] else right


def strongest_type_basis(left: EntityTypeBasis, right: EntityTypeBasis) -> EntityTypeBasis:
    return left if _TYPE_BASIS_RANK[left] >= _TYPE_BASIS_RANK[right] else right


class SourceAttribution(BaseModel):
    article_url: str | None = None
    article_title: str | None = None
    evidence: str | None = None


class ArticleContentRemoval(BaseModel):
    reason: str
    block_index: int | None = None
    heading: str | None = None
    text_chars: int
    text_preview: str


class ArticleCleaningReport(BaseModel):
    selected_container: str
    blocks_before: int
    blocks_after: int
    text_chars_before: int
    text_chars_after: int
    removed_blocks: list[ArticleContentRemoval] = Field(default_factory=list)
    remaining_promotion_markers: list[str] = Field(default_factory=list)


class ArticleIn(BaseModel):
    url: str
    discovered_url: str | None = None
    title: str
    source_name: str = "deutsche-startups.de"
    source_url: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    summary: str | None = None
    text: str
    tags: list[str] = Field(default_factory=list)
    primary_type: str | None = None
    cleaning: ArticleCleaningReport | None = None

    @field_validator("text")
    @classmethod
    def text_has_signal(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError("article text is too short")
        return value.strip()


class ExtractedEntity(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    type_basis: EntityTypeBasis = "legacy"
    evidence_status: EvidenceStatus = "unsure"
    evidence_status_defaulted: bool = False
    description: str | None = None
    source: SourceAttribution = Field(default_factory=SourceAttribution)


class ExtractedRelationship(BaseModel):
    type: RelationshipType
    source_name: str
    source_type: EntityType
    target_name: str
    target_type: EntityType
    keywords: str | None = None
    evidence_status: EvidenceStatus = "unsure"
    evidence_status_defaulted: bool = False
    evidence: str | None = None

    @model_validator(mode="after")
    def endpoint_types_match_relationship(self) -> "ExtractedRelationship":
        if not has_valid_relationship_signature(
            self.type,
            self.source_type,
            self.target_type,
        ):
            raise ValueError(
                f"invalid {self.type} signature: {self.source_type} -> {self.target_type}"
            )
        return self


class ExtractionResult(BaseModel):
    startups: list[ExtractedEntity] = Field(default_factory=list)
    investors: list[ExtractedEntity] = Field(default_factory=list)
    people: list[ExtractedEntity] = Field(default_factory=list)
    topics: list[ExtractedEntity] = Field(default_factory=list)
    companies: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    raw_model_output: dict[str, Any] | None = None

    @field_validator("topics", mode="before")
    @classmethod
    def drop_legacy_geography_topics(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [
            item
            for item in value
            if not (
                isinstance(item, dict) and str(item.get("category") or "").casefold() == "geography"
            )
        ]

    def entity_count(self) -> int:
        return (
            len(self.startups)
            + len(self.investors)
            + len(self.people)
            + len(self.topics)
            + len(self.companies)
        )


class NormalizedEntity(BaseModel):
    id: str
    label: EntityType
    canonical_name: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    primary_type_basis: EntityTypeBasis = "legacy"
    observed_types: list[EntityType] = Field(default_factory=list)
    type_conflict: bool = False
    evidence_status: EvidenceStatus = "unsure"
    description: str | None = None
    descriptions: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    concept_id: str | None = None
    ontology_version: str | None = None
    ontology_hash: str | None = None
    semantic_boundary: str | None = None


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class SearchResult(BaseModel):
    id: str
    name: str
    type: str
    score: float = 0
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    roles: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    source_url: HttpUrl | str = "https://www.deutsche-startups.de"
    source_name: str = "deutsche-startups.de"
    max_pages: int = Field(default=2, ge=1, le=50)
    include_feed: bool = False
    force_rescrape: bool = False
    paths: list[str] = Field(
        default_factory=lambda: ["/ressort/startups/", "/ressort/deals/", "/tag/startupticker/"]
    )


FeedbackLabel = Literal["good", "bad", "wrong_merge", "missed_entity", "other"]
FeedbackTarget = Literal["extraction", "resolution", "overall"]
ClaimReviewDecision = Literal["accepted", "rejected", "unreviewed"]
EntityDescriptionReviewDecision = Literal["accepted", "rejected", "unreviewed"]


class FeedbackIn(BaseModel):
    """Payload for `POST /traces/{trace_id}/feedback`."""

    label: FeedbackLabel
    target: FeedbackTarget = "overall"
    comment: str | None = Field(default=None, max_length=2000)
    reviewer: str | None = Field(default=None, max_length=200)


class EntityAliasIn(BaseModel):
    alias: str = Field(min_length=1, max_length=200)

    @field_validator("alias")
    @classmethod
    def clean_alias(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("alias must not be empty")
        return cleaned


class EntityAliasResult(BaseModel):
    node_id: str
    name: str
    aliases: list[str]
    merged_node_ids: list[str] = Field(default_factory=list)


class EntityDescriptionReviewIn(BaseModel):
    decision: EntityDescriptionReviewDecision
    description: str | None = Field(default=None, max_length=4000)
    comment: str | None = Field(default=None, max_length=2000)
    reviewer: str | None = Field(default=None, max_length=200)

    @field_validator("description")
    @classmethod
    def clean_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("description must not be empty")
        return cleaned

    @model_validator(mode="after")
    def edited_description_is_accepted(self) -> "EntityDescriptionReviewIn":
        if self.description is not None and self.decision != "accepted":
            raise ValueError("an edited description must be accepted")
        return self


class EntityDescriptionReviewResult(BaseModel):
    node_id: str
    description: str
    description_source: str | None = None
    human_review_status: EntityDescriptionReviewDecision
    reviewed_at: datetime | None = None


class ClaimReviewIn(BaseModel):
    source_id: str
    relationship: RelationshipType
    target_id: str
    decision: ClaimReviewDecision
    comment: str | None = Field(default=None, max_length=2000)
    reviewer: str | None = Field(default=None, max_length=200)


class IngestStats(BaseModel):
    source_name: str
    articles_found: int = 0
    articles_cached: int = 0
    articles_scraped: int = 0
    articles_skipped: int = 0
    articles_processed: int = 0
    articles_failed: int = 0
    entities_extracted: int = 0
    relationships_created: int = 0
    topic_candidates: int = 0
    topics_kept: int = 0
    topics_rejected: int = 0
    has_topic_relationships_rejected: int = 0
    duration_ms: float = 0
    article_extraction_policy_hash: str | None = None
    cache_message: str | None = None


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    name: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Raw extraction records (pre-normalization)
# ---------------------------------------------------------------------------


class RawEntityRecord(BaseModel):
    name: str
    entity_type: str  # lowercase: startup | investor | person | topic
    type_basis: EntityTypeBasis = "legacy"
    evidence_status: EvidenceStatus = "unsure"
    evidence_status_defaulted: bool = False
    description: str
    evidence: str = ""
    evidence_ref: str = ""


class RawRelationshipRecord(BaseModel):
    source: str
    target: str
    rel_type: str
    evidence_status: EvidenceStatus = "unsure"
    evidence_status_defaulted: bool = False
    keywords: str
    description: str
    evidence: str = ""
    evidence_ref: str = ""


class RawExtractionResult(BaseModel):
    entities: list[RawEntityRecord] = Field(default_factory=list)
    relationships: list[RawRelationshipRecord] = Field(default_factory=list)


class StructuredEntityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    entity_type: Literal["startup", "investor", "person", "topic", "company"]
    type_basis: Literal["explicit", "contextual", "fallback"]
    evidence_status: EvidenceStatus
    description: str
    evidence_ref: str


class StructuredRelationshipRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relationship_type: RelationshipType
    evidence_status: EvidenceStatus
    keywords: str
    evidence_ref: str


class StructuredExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[StructuredEntityRecord]
    relationships: list[StructuredRelationshipRecord]

    def to_raw(self) -> RawExtractionResult:
        return RawExtractionResult(
            entities=[
                RawEntityRecord(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    type_basis=entity.type_basis,
                    evidence_status=entity.evidence_status,
                    description=entity.description,
                    evidence_ref=entity.evidence_ref,
                )
                for entity in self.entities
            ],
            relationships=[
                RawRelationshipRecord(
                    source=relationship.source,
                    target=relationship.target,
                    rel_type=relationship.relationship_type,
                    evidence_status=relationship.evidence_status,
                    keywords=relationship.keywords,
                    description="",
                    evidence_ref=relationship.evidence_ref,
                )
                for relationship in self.relationships
            ],
        )
