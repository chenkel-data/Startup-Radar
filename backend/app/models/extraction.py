from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

EntityType = Literal["Startup", "Investor", "Person", "Topic", "Company"]
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


def strongest_evidence_status(left: EvidenceStatus, right: EvidenceStatus) -> EvidenceStatus:
    return left if _EVIDENCE_RANK[left] >= _EVIDENCE_RANK[right] else right


class SourceAttribution(BaseModel):
    article_url: str | None = None
    article_title: str | None = None
    evidence: str | None = None


class ArticleIn(BaseModel):
    url: str
    title: str
    source_name: str = "deutsche-startups.de"
    source_url: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    summary: str | None = None
    text: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def text_has_signal(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError("article text is too short")
        return value.strip()


class ExtractedEntity(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
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
    evidence_status: EvidenceStatus = "unsure"
    description: str | None = None
    descriptions: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


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


class IngestRequest(BaseModel):
    source_url: HttpUrl | str = "https://www.deutsche-startups.de"
    source_name: str = "deutsche-startups.de"
    max_pages: int = Field(default=2, ge=1, le=50)
    include_feed: bool = False
    paths: list[str] = Field(
        default_factory=lambda: ["/ressort/startups/", "/ressort/deals/", "/tag/startupticker/"]
    )


FeedbackLabel = Literal["good", "bad", "wrong_merge", "missed_entity", "other"]
FeedbackTarget = Literal["extraction", "resolution", "overall"]
ClaimReviewDecision = Literal["accepted", "rejected", "unreviewed"]


class FeedbackIn(BaseModel):
    """Payload for `POST /traces/{trace_id}/feedback`."""

    label: FeedbackLabel
    target: FeedbackTarget = "overall"
    comment: str | None = Field(default=None, max_length=2000)
    reviewer: str | None = Field(default=None, max_length=200)


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
    articles_processed: int = 0
    articles_failed: int = 0
    entities_extracted: int = 0
    relationships_created: int = 0
    duration_ms: float = 0


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
# Raw LightRAG-style extraction records (pre-normalization)
# ---------------------------------------------------------------------------


class RawEntityRecord(BaseModel):
    name: str
    entity_type: str  # lowercase: startup | investor | person | topic
    evidence_status: EvidenceStatus = "unsure"
    evidence_status_defaulted: bool = False
    description: str


class RawRelationshipRecord(BaseModel):
    source: str
    target: str
    rel_type: str
    evidence_status: EvidenceStatus = "unsure"
    evidence_status_defaulted: bool = False
    keywords: str
    description: str


class RawExtractionResult(BaseModel):
    entities: list[RawEntityRecord] = Field(default_factory=list)
    relationships: list[RawRelationshipRecord] = Field(default_factory=list)
