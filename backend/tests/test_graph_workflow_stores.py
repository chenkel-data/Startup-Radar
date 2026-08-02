import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.api.routes import review_entity_description
from app.graph.admin_store import AdminStore
from app.graph.claim_store import ClaimStore
from app.graph.entity_profile_store import EntityProfileStore
from app.graph.entity_resolution_store import EntityResolutionStore
from app.models.extraction import EntityDescriptionReviewIn


RecordBatch = dict[str, Any] | list[dict[str, Any]] | None


class FakeResult:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = records or []
        self._index = 0

    async def single(self) -> dict[str, Any] | None:
        return self.records[0] if self.records else None

    def __aiter__(self) -> "FakeResult":
        self._index = 0
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self.records):
            raise StopAsyncIteration
        record = self.records[self._index]
        self._index += 1
        return record


class FakeSession:
    def __init__(self, records: list[RecordBatch]) -> None:
        self.records = records
        self.runs: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def run(self, query: str, **params: Any) -> FakeResult:
        self.runs.append((query, params))
        batch = self.records.pop(0) if self.records else None
        if isinstance(batch, list):
            return FakeResult(batch)
        if isinstance(batch, dict):
            return FakeResult([batch])
        return FakeResult()


class FakeNeo4j:
    def __init__(self, records: list[RecordBatch] | None = None) -> None:
        self.records = list(records or [])
        self.sessions: list[FakeSession] = []

    def session(self) -> FakeSession:
        session = FakeSession(self.records)
        self.sessions.append(session)
        return session


class FakeNode(dict):
    def __init__(self, properties: dict[str, Any], labels: set[str]) -> None:
        super().__init__(properties)
        self.labels = labels
        self.element_id = properties.get("id", "node-element-id")


class FakeRelationship(dict):
    def __init__(self, rel_type: str, properties: dict[str, Any]) -> None:
        super().__init__(properties)
        self.type = rel_type
        self.element_id = properties.get("element_id", "relationship-element-id")


class FakeClaimStore:
    def __init__(self) -> None:
        self.initialized = False
        self.conflicts_refreshed = False

    async def initialize_claim_state(self) -> int:
        self.initialized = True
        return 2

    async def refresh_claim_conflicts(self) -> int:
        self.conflicts_refreshed = True
        return 1


class FakeProfileReviewStore:
    def __init__(self) -> None:
        self.saved: dict[str, Any] | None = None

    async def description_review_input(self, entity_id: str) -> dict[str, Any]:
        return {
            "id": entity_id,
            "label": "Company",
            "name": "SAP",
            "canonical_name": "SAP",
            "aliases": ["SAP SE"],
            "current_description": "SAP ist ein Softwareunternehmen.",
        }

    async def review_description(self, **kwargs: Any) -> dict[str, Any]:
        self.saved = kwargs
        return {
            "node_id": kwargs["entity_id"],
            "description": kwargs["description"],
            "description_source": "human",
            "human_review_status": kwargs["decision"],
            "reviewed_at": "2026-07-28T09:00:00+00:00",
        }


class FakeEmbedding:
    def __init__(self) -> None:
        self.text: str | None = None

    async def embed_one(self, text: str) -> list[float]:
        self.text = text
        return [0.1, 0.2]


@pytest.mark.asyncio
async def test_admin_apply_schema_runs_schema_and_claim_maintenance(monkeypatch) -> None:
    neo4j = FakeNeo4j()
    claims = FakeClaimStore()
    store = AdminStore(neo4j, claims)  # type: ignore[arg-type]
    deleted_geography = False
    cleared_categories = False

    def fake_schema_statements(embedding_provider: str) -> list[str]:
        assert embedding_provider == "openai"
        return ["CREATE INDEX test_index IF NOT EXISTS"]

    async def fake_delete_geography_topics() -> int:
        nonlocal deleted_geography
        deleted_geography = True
        return 0

    async def fake_clear_topic_categories() -> int:
        nonlocal cleared_categories
        cleared_categories = True
        return 0

    monkeypatch.setattr("app.graph.admin_store.build_schema_statements", fake_schema_statements)
    monkeypatch.setattr(store, "delete_geography_topics", fake_delete_geography_topics)
    monkeypatch.setattr(store, "clear_topic_categories", fake_clear_topic_categories)

    await store.apply_schema("openai")

    assert neo4j.sessions[0].runs == [("CREATE INDEX test_index IF NOT EXISTS", {})]
    assert deleted_geography is True
    assert cleared_categories is True
    assert claims.initialized is True
    assert claims.conflicts_refreshed is True


@pytest.mark.asyncio
async def test_entity_resolution_relationship_check_normalizes_candidate_names() -> None:
    neo4j = FakeNeo4j(records=[{"blocked": True}])
    store = EntityResolutionStore(neo4j)

    blocked = await store.has_relationship_to_named_entity(
        target_id="startup:prior-labs",
        candidate_label="Company",
        candidate_names=["SAP-SE"],
    )

    _query, params = neo4j.sessions[0].runs[0]
    assert blocked is True
    assert params["target_id"] == "startup:prior-labs"
    assert "sap se" in params["name_variants"]
    assert "company:sap-se" in params["candidate_ids"]


@pytest.mark.asyncio
async def test_profile_review_inputs_can_scope_to_resolved_entity_ids() -> None:
    neo4j = FakeNeo4j(records=[[]])
    store = EntityProfileStore(neo4j)

    rows = await store.profile_review_inputs(
        profile_curation_policy_hash="policy-1",
        entity_ids=["company:sap"],
    )

    query, params = neo4j.sessions[0].runs[0]
    assert rows == []
    assert params["entity_ids"] == ["company:sap"]
    assert params["profile_curation_policy_hash"] == "policy-1"
    assert "n.id IN $entity_ids" in query
    assert 'coalesce(n.description_source, "") <> "human"' in query
    assert "NOT evidence.id IN considered_evidence_ids" in query
    assert "evidence.ingested_profile_curation_policy_hashes" in query


@pytest.mark.asyncio
async def test_profile_revision_persists_article_policy_hash() -> None:
    """Remember updated article evidence so the same policy does not review it again."""
    neo4j = FakeNeo4j(records=[{"revision": 2}])
    store = EntityProfileStore(neo4j)

    await store.save_profile_revision(
        entity_id="company:sap",
        description="SAP ist ein Softwareunternehmen.",
        evidence_ids_considered=["profile-evidence:1"],
        used_evidence_ids=[],
        article_ids_considered=["article:1"],
        profile_curation_policy_hash="policy-1",
        review={
            "decision": "update_profile",
            "confidence": "high",
            "reason": "Stable profile detail.",
        },
        curation={
            "description": "SAP ist ein Softwareunternehmen.",
            "confidence": "high",
            "used_evidence_refs": [],
            "limitations": [],
        },
        model="gpt-test",
    )

    _save_query, save_params = neo4j.sessions[0].runs[0]
    _mark_query, mark_params = neo4j.sessions[0].runs[1]
    assert save_params["profile_curation_policy_hash"] == "policy-1"
    assert save_params["article_policy_keys"] == ["article:1::policy-1"]
    assert mark_params["profile_curation_policy_hash"] == "policy-1"


@pytest.mark.asyncio
async def test_description_edit_refreshes_embedding_before_it_is_saved() -> None:
    store = FakeProfileReviewStore()
    embedding = FakeEmbedding()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                entity_profile_store=store,
                embedding=embedding,
                settings=SimpleNamespace(
                    embedding_provider="openai",
                    embedding_model="embedding-test",
                    embedding_st_model="unused",
                ),
            )
        )
    )

    result = await review_entity_description(
        request,  # type: ignore[arg-type]
        "company:sap",
        EntityDescriptionReviewIn(
            decision="accepted",
            description="SAP entwickelt Unternehmenssoftware.",
        ),
    )

    assert result.human_review_status == "accepted"
    assert embedding.text is not None
    assert "Public profile: SAP entwickelt Unternehmenssoftware." in embedding.text
    assert store.saved is not None
    assert store.saved["embedding"]
    assert store.saved["embedding_input_hash"]
    assert store.saved["embedding_model"] == "embedding-test"


@pytest.mark.asyncio
async def test_claim_review_writes_decision_history() -> None:
    neo4j = FakeNeo4j(records=[{"reviewed": 1}])
    store = ClaimStore(neo4j)

    reviewed = await store.review_claim(
        source_id="company:sap",
        relationship="ACQUIRED",
        target_id="startup:prior-labs",
        decision="accepted",
        comment="confirmed",
        reviewer="admin",
    )

    _query, params = neo4j.sessions[0].runs[0]
    review_event = json.loads(params["review_event"])
    assert reviewed is True
    assert params["decision"] == "accepted"
    assert review_event["decision"] == "accepted"
    assert review_event["comment"] == "confirmed"
    assert review_event["reviewer"] == "admin"


@pytest.mark.asyncio
async def test_node_claims_preserves_empty_active_article_urls() -> None:
    source = FakeNode({"id": "company:sap", "name": "SAP"}, {"Company"})
    target = FakeNode({"id": "startup:prior-labs", "name": "Prior Labs"}, {"Startup"})
    relationship = FakeRelationship(
        "ACQUIRED",
        {
            "article_urls": ["https://example.test/historical-support"],
            "active_article_urls": [],
            "lifecycle_status": "unsupported_by_latest_source_processing",
            "review_status": "needs_review",
            "review_reasons": ["not_reproduced_same_article"],
            "support_changed": True,
            "provenance": [
                json.dumps(
                    {
                        "event": "not_reproduced",
                        "article_url": "https://example.test/historical-support",
                        "article_title": "Historical support",
                        "processed_at": "2026-06-10T09:00:00Z",
                    }
                ),
                json.dumps(
                    {
                        "event": "asserted",
                        "article_url": "https://example.test/historical-support",
                        "article_title": "Historical support",
                        "processed_at": "2026-06-09T09:00:00Z",
                        "evidence": "Earlier relationship evidence.",
                    }
                ),
            ],
        },
    )
    neo4j = FakeNeo4j(
        records=[{"node": source}, [{"source": source, "r": relationship, "target": target}]]
    )
    store = ClaimStore(neo4j)

    claims = await store.node_claims("company:sap")

    assert claims is not None
    claim = claims["claims"][0]
    assert claim["active_support_count"] == 0
    assert claim["active_article_urls"] == []
    assert claim["source_articles"] == [
        {
            "article_url": "https://example.test/historical-support",
            "article_title": "Historical support",
            "published_at": None,
            "latest_processed_at": "2026-06-10T09:00:00Z",
            "status": "no_longer_current",
            "evidence": "Earlier relationship evidence.",
            "trace_url": None,
            "processing_count": 2,
            "review_source": True,
        }
    ]
