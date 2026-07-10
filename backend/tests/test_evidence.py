from app.evidence import build_evidence_catalog


def test_catalog_resolves_verbatim_passages_and_rejects_invalid_references() -> None:
    text = "## Funding\nNova raises capital. Fund X invests!\n## Team\nMia Kern founded Nova"

    catalog = build_evidence_catalog(text)

    assert "## Funding" in catalog.rendered_body
    assert "[B001.S001] Nova raises capital." in catalog.rendered_body
    assert "[B001.S002] Fund X invests!" in catalog.rendered_body
    assert "## Team" in catalog.rendered_body
    assert catalog.resolve("B001.S002") == ("Fund X invests!", None)
    assert catalog.resolve("B001.S001-B001.S002") == (
        "Nova raises capital. Fund X invests!",
        None,
    )
    assert catalog.resolve("B002.S001") == ("Mia Kern founded Nova", None)

    catalog = build_evidence_catalog("One. Two.\nThree.")

    assert catalog.resolve("") == (None, "missing_ref")
    assert catalog.resolve("B1.S1") == (None, "malformed_ref")
    assert catalog.resolve("B001.S001,B001.S002") == (None, "noncontiguous_ref")
    assert catalog.resolve("B001.S001-B002.S001") == (None, "cross_block_ref")
    assert catalog.resolve("B001.S002-B001.S001") == (None, "reversed_ref")
    assert catalog.resolve("B001.S999") == (None, "unknown_ref")
