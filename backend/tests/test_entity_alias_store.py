from typing import Any

import pytest

from app.graph.entity_resolution_store import (
    AmbiguousAliasError,
    _add_alias_tx,
    _merge_relationship_properties,
    _remove_alias_tx,
)


class FakeResult:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    async def single(self) -> dict[str, Any] | None:
        return self.records[0] if self.records else None

    def __aiter__(self):
        async def records():
            for record in self.records:
                yield record

        return records()


class FakeAliasTx:
    def __init__(
        self,
        *,
        target: dict[str, Any],
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        self.target = target
        self.candidates = candidates or []
        self.runs: list[tuple[str, dict[str, Any]]] = []

    async def run(self, query: str, **params: Any) -> FakeResult:
        self.runs.append((query, params))
        if "RETURN target, labels(target) AS labels" in query:
            return FakeResult([{"target": self.target, "labels": ["Investor"]}])
        if "RETURN candidate" in query:
            return FakeResult([{"candidate": candidate} for candidate in self.candidates])
        if "RETURN elementId(relationship)" in query:
            return FakeResult([])
        return FakeResult([])


@pytest.mark.asyncio
async def test_alias_correction_preserves_entity_and_relationship_history() -> None:
    merge_tx = FakeAliasTx(
        target={
            "id": "investor:earlybird-venture-capital",
            "name": "Earlybird Venture Capital",
            "canonical_name": "Earlybird Venture Capital",
            "aliases": ["Earlybird Venture Capital"],
            "description": "Canonical profile",
        },
        candidates=[
            {
                "id": "investor:earlybird",
                "name": "Earlybird",
                "canonical_name": "Earlybird",
                "aliases": ["Earlybird"],
                "description": "Earlier profile",
            }
        ],
    )

    result = await _add_alias_tx(
        merge_tx,
        "investor:earlybird-venture-capital",
        "Earlybird",
    )

    assert result is not None
    assert result["merged_node_ids"] == ["investor:earlybird"]
    assert result["aliases"] == ["Earlybird Venture Capital", "Earlybird"]
    update = next(params for query, params in merge_tx.runs if "SET target = $properties" in query)
    assert update["properties"]["description"] == "Canonical profile"
    assert update["properties"]["descriptions"] == ["Canonical profile", "Earlier profile"]

    relationship = _merge_relationship_properties(
        {
            "evidence_status": "attributed",
            "evidence": "Target evidence",
            "article_urls": ["https://example.test/target"],
            "provenance": ["target"],
            "review_status": "accepted",
            "review_history": ["accepted"],
        },
        {
            "evidence_status": "stated",
            "evidence": "Source evidence",
            "article_urls": ["https://example.test/source"],
            "provenance": ["source"],
            "review_status": "rejected",
            "review_history": ["rejected"],
        },
    )
    assert relationship["article_urls"] == [
        "https://example.test/target",
        "https://example.test/source",
    ]
    assert relationship["review_history"] == ["accepted", "rejected"]
    assert relationship["review_status"] == "needs_review"
    assert relationship["review_reasons"] == ["manual_entity_merge_conflict"]

    remove_tx = FakeAliasTx(
        target={
            "id": "investor:earlybird-venture-capital",
            "name": "Earlybird Venture Capital",
            "canonical_name": "Earlybird Venture Capital",
            "aliases": ["Earlybird Venture Capital", "Earlybird", "Earlybird VC"],
        }
    )
    corrected = await _remove_alias_tx(
        remove_tx,
        "investor:earlybird-venture-capital",
        " earlybird ",
    )
    assert corrected is not None
    assert corrected["node_id"] == "investor:earlybird-venture-capital"
    assert corrected["aliases"] == ["Earlybird Venture Capital", "Earlybird VC"]


@pytest.mark.asyncio
async def test_add_alias_rejects_multiple_matching_nodes_without_writing() -> None:
    tx = FakeAliasTx(
        target={
            "id": "investor:canonical",
            "name": "Canonical",
            "canonical_name": "Canonical",
            "aliases": [],
        },
        candidates=[
            {
                "id": "investor:first",
                "name": "Earlybird",
                "canonical_name": "Earlybird",
                "aliases": [],
            },
            {
                "id": "investor:second",
                "name": "Other",
                "canonical_name": "Other",
                "aliases": ["Earlybird"],
            },
        ],
    )

    with pytest.raises(AmbiguousAliasError, match="multiple existing Investor nodes"):
        await _add_alias_tx(tx, "investor:canonical", "Earlybird")

    assert not any("SET target = $properties" in query for query, _ in tx.runs)
