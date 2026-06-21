from types import SimpleNamespace

import pytest

from app.services.entity_curation import (
    EntityProfileCurationService,
)


class FakeGraph:
    def __init__(self, candidates):
        self.candidates = candidates
        self.requested_entity_ids = []
        self.requested_policy_hashes = []
        self.kept = []
        self.revisions = []
        self.embeddings = []

    async def profile_review_inputs(self, *, profile_curation_policy_hash, entity_ids=None):
        self.requested_entity_ids.append(entity_ids)
        self.requested_policy_hashes.append(profile_curation_policy_hash)
        if entity_ids is None:
            return self.candidates
        return [candidate for candidate in self.candidates if candidate["id"] in entity_ids]

    async def save_profile_review_keep(self, **kwargs):
        self.kept.append(kwargs)

    async def save_profile_revision(self, **kwargs):
        self.revisions.append(kwargs)

    async def save_profile_embedding(self, **kwargs):
        self.embeddings.append(kwargs)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def _call_llm_raw(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.responses.pop(0)


class FakeEmbedding:
    async def embed_one(self, text):
        return [0.1, 0.2, 0.3]


def settings():
    return SimpleNamespace(
        entity_curation_max_concurrency=2,
        openai_model="gpt-test",
    )


def entity_candidate():
    return {
        "id": "company:sap",
        "label": "Company",
        "name": "SAP",
        "aliases": [],
        "current_description": "SAP ist ein Softwareunternehmen.",
        "profile_revision": 1,
        "has_embedding": False,
        "embedding_input_hash": None,
        "new_evidence": [
            {
                "id": "profile-evidence:1",
                "article_id": "article:1",
                "article_url": "https://example.test/sap",
                "kind": "entity_description",
                "text": "SAP entwickelt Unternehmenssoftware fuer Firmenkunden.",
                "published_at": "2026-06-01T08:00:00+00:00",
                "created_at": "2026-06-01T09:00:00+00:00",
            }
        ],
    }


@pytest.mark.asyncio
async def test_curation_review_can_keep_profile_and_update_missing_embedding() -> None:
    graph = FakeGraph([entity_candidate()])
    llm = FakeLLM(
        [
            """
            {
              "decision": "keep_profile",
              "confidence": "high",
              "reason": "The new evidence is already covered.",
              "evidence_refs_considered": ["E1"],
              "update_instructions": []
            }
            """
        ]
    )
    service = EntityProfileCurationService(
        settings=settings(),
        llm=llm,
        embedding=FakeEmbedding(),
        profile_store=graph,
    )

    rows = await service.curate_profiles([{"entity_id": "company:sap", "entity_type": "Company"}])

    assert rows[0]["status"] == "kept"
    assert rows[0]["embedding_updated"] is True
    assert graph.kept[0]["entity_id"] == "company:sap"
    assert graph.kept[0]["evidence_ids"] == ["profile-evidence:1"]
    assert graph.kept[0]["article_ids"] == ["article:1"]
    assert graph.kept[0]["profile_curation_policy_hash"] == service.profile_curation_policy_hash
    assert graph.kept[0]["embedding"] == [0.1, 0.2, 0.3]
    assert graph.revisions == []


@pytest.mark.asyncio
async def test_curation_updates_profile_only_after_update_review() -> None:
    graph = FakeGraph([entity_candidate()])
    llm = FakeLLM(
        [
            """
            {
              "decision": "update_profile",
              "confidence": "high",
              "reason": "The new evidence adds stable product information.",
              "evidence_refs_considered": ["E1"],
              "update_instructions": ["Describe the company generally."]
            }
            """,
            """
            {
              "description": "SAP ist ein Softwareunternehmen mit Fokus auf Unternehmensanwendungen.",
              "confidence": "high",
              "used_evidence_refs": ["E1"],
              "limitations": []
            }
            """,
        ]
    )
    service = EntityProfileCurationService(
        settings=settings(),
        llm=llm,
        embedding=FakeEmbedding(),
        profile_store=graph,
    )

    rows = await service.curate_profiles([{"entity_id": "company:sap", "entity_type": "Company"}])

    assert rows[0]["status"] == "updated"
    assert graph.revisions[0]["description"].startswith("SAP ist ein Softwareunternehmen")
    assert graph.revisions[0]["evidence_ids_considered"] == ["profile-evidence:1"]
    assert graph.revisions[0]["article_ids_considered"] == ["article:1"]
    assert (
        graph.revisions[0]["profile_curation_policy_hash"] == service.profile_curation_policy_hash
    )
    assert graph.kept == []


@pytest.mark.asyncio
async def test_curation_can_scope_candidates_to_resolved_entity_ids() -> None:
    other = {
        **entity_candidate(),
        "id": "company:other",
        "name": "Other",
    }
    graph = FakeGraph([entity_candidate(), other])
    llm = FakeLLM(
        [
            """
            {
              "decision": "keep_profile",
              "confidence": "high",
              "reason": "The new evidence is already covered.",
              "evidence_refs_considered": ["E1"],
              "update_instructions": []
            }
            """
        ]
    )
    service = EntityProfileCurationService(
        settings=settings(),
        llm=llm,
        embedding=FakeEmbedding(),
        profile_store=graph,
    )

    rows = await service.curate_entity_ids(["company:sap"], job_run_id="job-1")

    assert graph.requested_entity_ids == [["company:sap"]]
    assert graph.requested_policy_hashes == [service.profile_curation_policy_hash]
    assert [row["entity_id"] for row in rows] == ["company:sap"]
    assert len(llm.calls) == 1
    assert graph.kept[0]["entity_id"] == "company:sap"


@pytest.mark.asyncio
async def test_curation_skips_llm_when_all_new_evidence_is_exact_duplicate() -> None:
    entity = {
        **entity_candidate(),
        "has_embedding": True,
        "embedding_input_hash": None,
        "considered_evidence": [
            {
                "id": "profile-evidence:old",
                "article_id": "article:old",
                "text": "SAP ist ein Softwareunternehmen.",
            }
        ],
        "new_evidence": [
            {
                "id": "profile-evidence:new",
                "article_id": "article:new",
                "kind": "entity_description",
                "text": " sap ist ein softwareunternehmen. ",
                "published_at": "2026-06-02T08:00:00+00:00",
                "created_at": "2026-06-02T09:00:00+00:00",
            }
        ],
    }
    graph = FakeGraph([entity])
    llm = FakeLLM([])
    service = EntityProfileCurationService(
        settings=settings(),
        llm=llm,
        embedding=FakeEmbedding(),
        profile_store=graph,
    )

    rows = await service.curate_profiles([{"entity_id": "company:sap", "entity_type": "Company"}])

    assert rows[0]["status"] == "skipped_exact_duplicate_evidence"
    assert rows[0]["review_decision"] == "exact_duplicate_evidence"
    assert llm.calls == []
    assert graph.kept[0]["evidence_ids"] == ["profile-evidence:new"]
    assert graph.kept[0]["article_ids"] == ["article:new"]
    assert graph.kept[0]["review"]["decision"] == "exact_duplicate_evidence"
    assert graph.revisions == []


@pytest.mark.asyncio
async def test_curation_backfills_missing_embeddings_when_no_entities_were_touched() -> None:
    graph = FakeGraph(
        [
            {
                **entity_candidate(),
                "new_evidence": [],
                "has_embedding": False,
            }
        ]
    )
    service = EntityProfileCurationService(
        settings=settings(),
        llm=FakeLLM([]),
        embedding=FakeEmbedding(),
        profile_store=graph,
    )

    rows = await service.curate_profiles([])

    assert rows[0]["entity_id"] == "company:sap"
    assert graph.embeddings[0]["entity_id"] == "company:sap"
