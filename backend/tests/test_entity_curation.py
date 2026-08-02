from types import SimpleNamespace

import pytest

from app.services.entity_curation import (
    PROFILE_CURATION_MIN_NEW_EVIDENCE,
    EntityProfileCurationService,
)


class FakeGraph:
    def __init__(self, candidates):
        self.candidates = candidates
        self.requested_policy_hashes = []
        self.kept = []
        self.revisions = []

    async def profile_review_inputs(self, *, profile_curation_policy_hash, entity_ids=None):
        self.requested_policy_hashes.append(profile_curation_policy_hash)
        return self.candidates

    async def save_profile_review_keep(self, **kwargs):
        self.kept.append(kwargs)
        self._mark_considered(kwargs["entity_id"], kwargs["evidence_ids"])

    async def save_profile_revision(self, **kwargs):
        self.revisions.append(kwargs)
        self._mark_considered(kwargs["entity_id"], kwargs["evidence_ids_considered"])

    def _mark_considered(self, entity_id, evidence_ids):
        considered = set(evidence_ids)
        for candidate in self.candidates:
            if candidate["id"] != entity_id:
                continue
            candidate["new_evidence"] = [
                evidence
                for evidence in candidate["new_evidence"]
                if evidence["id"] not in considered
            ]


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def _call_llm_raw(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.responses.pop(0)

    def pop_openai_rate_limit_stats(self):
        return {"events": 0}


class FakeEmbedding:
    async def embed_one(self, text):
        return [0.1, 0.2, 0.3]


def settings():
    return SimpleNamespace(
        entity_curation_max_concurrency=1,
        openai_model="gpt-test",
    )


def evidence(index: int, *, article_id: str = "article:1", text: str | None = None):
    return {
        "id": f"profile-evidence:{index}",
        "article_id": article_id,
        "article_url": f"https://example.test/{article_id}/{index}",
        "kind": "entity_description",
        "text": text or f"Eigenständige Profilinformation Nummer {index}.",
        "published_at": f"2026-06-{index:02d}T08:00:00+00:00",
        "created_at": f"2026-06-{index:02d}T09:00:00+00:00",
    }


def entity_candidate(
    evidence_count: int = PROFILE_CURATION_MIN_NEW_EVIDENCE,
    *,
    entity_id: str = "company:sap",
    label: str = "Company",
    name: str = "SAP",
    current_description: str = "SAP ist ein Softwareunternehmen.",
    article_id: str = "article:1",
):
    candidate_evidence = [
        evidence(index, article_id=article_id) for index in range(1, evidence_count + 1)
    ]
    for item in candidate_evidence:
        item["id"] = f"{entity_id}:evidence:{item['id'].rsplit(':', 1)[-1]}"
    return {
        "id": entity_id,
        "label": label,
        "name": name,
        "aliases": [],
        "current_description": current_description,
        "profile_revision": 1,
        "has_embedding": False,
        "embedding_input_hash": None,
        "new_evidence": candidate_evidence,
    }


def service_for(graph, responses):
    return EntityProfileCurationService(
        settings=settings(),
        llm=FakeLLM(responses),
        embedding=FakeEmbedding(),
        profile_store=graph,
    )


@pytest.mark.asyncio
async def test_pending_summary_waits_until_three_distinct_evidence_texts() -> None:
    candidate = entity_candidate(evidence_count=2)
    candidate["new_evidence"].append(
        evidence(
            3,
            article_id="article:2",
            text=" sap ist ein softwareunternehmen. ",
        )
    )
    graph = FakeGraph([candidate])
    service = service_for(graph, [])

    summary = await service.pending_summary()

    assert summary["threshold"] == 3
    assert summary["ready_entities"] == 0
    assert summary["waiting_entities"] == 1
    assert summary["waiting_evidence"] == 2
    assert service.llm.calls == []


@pytest.mark.asyncio
async def test_manual_curation_reviews_ready_profile_and_marks_evidence_considered() -> None:
    graph = FakeGraph([entity_candidate()])
    service = service_for(
        graph,
        [
            """
            {
              "decision": "keep_profile",
              "confidence": "high",
              "reason": "The new evidence is already covered.",
              "evidence_refs_considered": ["E1", "E2", "E3"],
              "update_instructions": []
            }
            """
        ],
    )

    result = await service.curate_pending_profiles()

    assert result["kept"] == 1
    assert result["ready_entities_remaining"] == 0
    assert len(graph.kept[0]["evidence_ids"]) == 3
    assert graph.kept[0]["article_ids"] == ["article:1"]
    assert graph.kept[0]["profile_curation_policy_hash"] == service.profile_curation_policy_hash


@pytest.mark.asyncio
async def test_manual_curation_updates_existing_and_missing_profiles_after_evidence_gate() -> None:
    graph = FakeGraph(
        [
            entity_candidate(),
            entity_candidate(
                entity_id="startup:nova",
                label="Startup",
                name="Nova",
                current_description="",
                article_id="article:2",
            ),
        ]
    )
    service = service_for(
        graph,
        [
            """
            {
              "decision": "update_profile",
              "confidence": "high",
              "reason": "The new evidence adds stable product information.",
              "evidence_refs_considered": ["E1", "E2", "E3"],
              "update_instructions": ["Describe the company generally."]
            }
            """,
            """
            {
              "description": "SAP ist ein Softwareunternehmen mit Fokus auf Unternehmensanwendungen.",
              "confidence": "high",
              "used_evidence_refs": ["E1", "E2", "E3"],
              "limitations": []
            }
            """,
            """
            {
              "description": "Nova entwickelt Software für Industrieunternehmen.",
              "confidence": "high",
              "used_evidence_refs": ["E1", "E2", "E3"],
              "limitations": []
            }
            """,
        ],
    )

    result = await service.curate_pending_profiles()

    assert result["updated"] == 2
    assert len(service.llm.calls) == 3
    revisions = {revision["entity_id"]: revision for revision in graph.revisions}
    assert revisions["company:sap"]["description"].startswith("SAP ist ein Softwareunternehmen")
    assert revisions["company:sap"]["article_ids_considered"] == ["article:1"]
    assert revisions["startup:nova"]["description"].startswith("Nova entwickelt Software")
    assert revisions["startup:nova"]["article_ids_considered"] == ["article:2"]
