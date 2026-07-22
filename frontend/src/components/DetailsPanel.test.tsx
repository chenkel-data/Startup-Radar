import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DetailsPanel } from "./DetailsPanel";
import { api } from "../lib/api";
import type { GraphNode, GraphResponse, NodeClaim, NodeClaimsResponse } from "../types/graph";

const apiMock = vi.hoisted(() => ({
  addEntityAlias: vi.fn(),
  removeEntityAlias: vi.fn(),
  reviewEntityDescription: vi.fn(),
  nodeClaims: vi.fn(),
  reviewClaim: vi.fn(),
}));

vi.mock("../lib/api", () => ({
  api: apiMock,
}));

const sapNode: GraphNode = {
  id: "company:sap",
  label: "SAP",
  type: "Company",
  properties: { description: "SAP is a company." },
};
const priorLabsNode: GraphNode = {
  id: "startup:prior-labs",
  label: "Prior Labs",
  type: "Startup",
  properties: {},
};
const aiNode: GraphNode = {
  id: "topic:kuenstliche-intelligenz",
  label: "Kuenstliche Intelligenz",
  type: "Topic",
  properties: {},
};
const graph: GraphResponse = {
  nodes: [sapNode, priorLabsNode, aiNode],
  edges: [
    {
      id: "edge-review",
      source: "company:sap",
      target: "startup:prior-labs",
      label: "ACQUIRED",
      properties: {},
    },
    {
      id: "edge-supported",
      source: "company:sap",
      target: "topic:kuenstliche-intelligenz",
      label: "HAS_TOPIC",
      properties: {},
    },
  ],
};

const LONG_EVIDENCE =
  "Helsing setzt auf KI-Fähigkeiten für den Sicherheits- und Verteidigungsbereich und wurde 2021 von Torsten Reil, Niklas Köhler und Gundbert Scherf gegründet. Mit dem frischen Kapital möchte das Unternehmen seine Mission beschleunigen und neue KI-Plattformen entwickeln.";

function reviewClaim(): NodeClaim {
  return {
    edge_id: "edge-review",
    relationship: "ACQUIRED",
    direction: "outgoing",
    counterparty: {
      id: "startup:prior-labs",
      label: "Prior Labs",
      type: "Startup",
    },
    lifecycle_status: "supported",
    review_status: "needs_review",
    review_reasons: ["inverse_direction"],
    support_changed: false,
    active_support_count: 1,
    active_article_urls: ["https://example.test/sap-prior-labs"],
    source_articles: [
      {
        article_title: "SAP +++ Prior Labs",
        article_url: "https://example.test/sap-prior-labs",
        latest_processed_at: "2026-05-31T14:35:22Z",
        status: "current",
        evidence: "SAP uebernimmt das junge KI-Startup Prior Labs.",
        processing_count: 1,
      },
    ],
    assertions: [
      {
        event: "asserted",
        article_title: "SAP +++ Prior Labs",
        article_url: "https://example.test/sap-prior-labs",
        processed_at: "2026-05-31T14:35:22Z",
        evidence: "SAP uebernimmt das junge KI-Startup Prior Labs.",
      },
    ],
    source_id: "company:sap",
    target_id: "startup:prior-labs",
  };
}

function supportedClaim(): NodeClaim {
  return {
    edge_id: "edge-supported",
    relationship: "HAS_TOPIC",
    direction: "outgoing",
    counterparty: {
      id: "topic:kuenstliche-intelligenz",
      label: "Kuenstliche Intelligenz",
      type: "Topic",
    },
    lifecycle_status: "supported",
    review_status: "unreviewed",
    review_reasons: [],
    support_changed: false,
    active_support_count: 2,
    active_article_urls: ["https://example.test/sap-ai", "https://example.test/sap-ai-2"],
    source_articles: [
      {
        article_title: "SAP AI article",
        article_url: "https://example.test/sap-ai",
        latest_processed_at: "2026-05-31T14:40:00Z",
        status: "current",
        evidence: "SAP arbeitet an KI-Produkten.",
        processing_count: 1,
      },
      {
        article_title: "Second SAP AI article",
        article_url: "https://example.test/sap-ai-2",
        latest_processed_at: "2026-05-31T14:41:00Z",
        status: "current",
        evidence: "SAP arbeitet an KI-Produkten.",
        processing_count: 1,
      },
    ],
    assertions: [
      {
        event: "asserted",
        article_title: "SAP AI article",
        article_url: "https://example.test/sap-ai",
        processed_at: "2026-05-31T14:40:00Z",
        evidence: "SAP arbeitet an KI-Produkten.",
      },
    ],
    source_id: "company:sap",
    target_id: "topic:kuenstliche-intelligenz",
  };
}

function longEvidenceClaim(): NodeClaim {
  const claim = supportedClaim();
  return {
    ...claim,
    edge_id: "edge-long-evidence",
    source_articles: [
      {
        ...claim.source_articles![0],
        evidence: LONG_EVIDENCE,
      },
    ],
    assertions: [
      {
        ...claim.assertions[0],
        evidence: LONG_EVIDENCE,
      },
    ],
  };
}

function rejectedClaim(): NodeClaim {
  return {
    ...reviewClaim(),
    review_status: "rejected",
    review_reasons: [],
    review_history: [
      {
        decision: "rejected",
        reviewer: "Editor",
        reviewed_at: "2026-05-31T15:00:00Z",
        comment: "Wrong direction.",
      },
    ],
  };
}

function latestEntityMention(): NodeClaim {
  return {
    edge_id: "mention-sap-article",
    relationship: "MENTIONS",
    direction: "incoming",
    counterparty: {
      id: "article:sap-latest",
      label: "SAP extraction source",
      type: "Article",
    },
    lifecycle_status: "supported",
    review_status: "unreviewed",
    review_reasons: [],
    support_changed: false,
    active_support_count: 0,
    source_articles: [],
    assertions: [
      {
        event: "asserted",
        article_title: "Earliest SAP source",
        article_url: "https://example.test/sap-earliest",
        processed_at: "2026-05-26T10:00:00Z",
        trace_id: "tr-earliest-reprocessed",
        mlflow_experiment_id: "2",
        evidence: "Earliest SAP evidence after reprocessing.",
      },
      {
        event: "asserted",
        article_title: "Earliest SAP source",
        article_url: "https://example.test/sap-earliest",
        processed_at: "2026-05-27T10:00:00Z",
        trace_id: "tr-earliest",
        mlflow_experiment_id: "2",
        evidence: "Earliest SAP evidence.",
      },
      {
        event: "asserted",
        article_title: "Oldest SAP source",
        article_url: "https://example.test/sap-oldest",
        processed_at: "2026-05-28T10:00:00Z",
        trace_id: "tr-oldest",
        mlflow_experiment_id: "2",
        evidence: "Oldest SAP evidence.",
      },
      {
        event: "asserted",
        article_title: "Earlier SAP source",
        article_url: "https://example.test/sap-earlier",
        processed_at: "2026-05-29T10:00:00Z",
        trace_id: "tr-earlier",
        mlflow_experiment_id: "2",
        evidence: "Earlier SAP evidence.",
      },
      {
        event: "asserted",
        article_title: "Older SAP source",
        article_url: "https://example.test/sap-old",
        processed_at: "2026-05-30T10:00:00Z",
        trace_id: "tr-old",
        mlflow_experiment_id: "2",
        evidence: "Older SAP evidence.",
      },
      {
        event: "asserted",
        article_title: "Latest SAP source",
        article_url: "https://example.test/sap-latest",
        published_at: "2026-05-16T00:00:00Z",
        processed_at: "2026-06-01T10:00:00Z",
        trace_id: "tr-latest",
        mlflow_trace_url: "http://localhost:5001/#/experiments/2/traces/tr-latest",
        evidence: "Latest SAP evidence.",
      },
    ],
    source_id: "article:sap-latest",
    target_id: "company:sap",
  };
}

function noLongerFoundClaimWithRepeatedArticle(): NodeClaim {
  return {
    ...reviewClaim(),
    review_reasons: ["not_reproduced_same_article"],
    support_changed: true,
    active_support_count: 0,
    active_article_urls: [],
    source_articles: [
      {
        article_title: "Bunch review source",
        article_url: "https://example.test/bunch-source",
        latest_processed_at: "2026-06-10T09:00:00Z",
        status: "no_longer_current",
        evidence: "Bunch was founded by Alice Example.",
        processing_count: 3,
        review_source: true,
      },
    ],
    assertions: [
      {
        event: "not_reproduced",
        article_title: "Bunch review source",
        article_url: "https://example.test/bunch-source",
        processed_at: "2026-06-10T09:00:00Z",
      },
      {
        event: "asserted",
        article_title: "Bunch review source",
        article_url: "https://example.test/bunch-source",
        processed_at: "2026-06-09T09:00:00Z",
        evidence: "Bunch was founded by Alice Example.",
      },
      {
        event: "asserted",
        article_title: "Bunch review source",
        article_url: "https://example.test/bunch-source",
        processed_at: "2026-06-08T09:00:00Z",
        evidence: "Older duplicate Bunch evidence.",
      },
    ],
  };
}

function supportChangedButCurrentlySupportedClaim(): NodeClaim {
  return {
    ...supportedClaim(),
    support_changed: true,
    active_support_count: 1,
    active_article_urls: ["https://example.test/current-support"],
    source_articles: [
      {
        article_title: "Current source",
        article_url: "https://example.test/current-support",
        latest_processed_at: "2026-06-10T09:00:00Z",
        status: "current",
        evidence: "SAP arbeitet weiter an KI-Produkten.",
        processing_count: 1,
      },
      {
        article_title: "Older source",
        article_url: "https://example.test/older-source",
        latest_processed_at: "2026-06-10T08:00:00Z",
        status: "no_longer_current",
        processing_count: 1,
      },
    ],
    assertions: [
      {
        event: "not_reproduced",
        article_title: "Older source",
        article_url: "https://example.test/older-source",
        processed_at: "2026-06-10T08:00:00Z",
      },
      {
        event: "asserted",
        article_title: "Current source",
        article_url: "https://example.test/current-support",
        processed_at: "2026-06-10T09:00:00Z",
        evidence: "SAP arbeitet weiter an KI-Produkten.",
      },
    ],
  };
}

function legacySupportedClaimWithoutSourceArticles(): NodeClaim {
  const claim = supportedClaim();
  const { source_articles: _sourceArticles, ...legacyClaim } = claim;
  return legacyClaim;
}

function claimsResponse(claims: NodeClaim[], mentions: NodeClaim[] = []): NodeClaimsResponse {
  return {
    node_id: "company:sap",
    claims,
    mentions,
  };
}

function renderDetailsPanel() {
  const onOpenGraph = vi.fn();
  const onEntityUpdated = vi.fn();
  const onNodeSelect = vi.fn();
  render(
    <DetailsPanel
      node={sapNode}
      graph={graph}
      visibleTypes={new Set(["Company", "Startup", "Topic"])}
      visibleRelations={new Set(["ACQUIRED", "HAS_TOPIC"])}
      onOpenGraph={onOpenGraph}
      onEntityUpdated={onEntityUpdated}
      onNodeSelect={onNodeSelect}
    />,
  );
  return { onOpenGraph, onEntityUpdated, onNodeSelect };
}

describe("DetailsPanel claim review behavior", () => {
  beforeEach(() => {
    vi.mocked(api.nodeClaims).mockResolvedValue(claimsResponse([reviewClaim(), supportedClaim()]));
    vi.mocked(api.reviewClaim).mockResolvedValue({ status: "ok", decision: "accepted" });
    vi.mocked(api.addEntityAlias).mockResolvedValue({
      node_id: sapNode.id,
      name: sapNode.label,
      aliases: ["SAP SE"],
      merged_node_ids: [],
    });
    vi.mocked(api.removeEntityAlias).mockResolvedValue({
      node_id: sapNode.id,
      name: sapNode.label,
      aliases: [],
      merged_node_ids: [],
    });
    vi.mocked(api.reviewEntityDescription).mockResolvedValue({
      node_id: sapNode.id,
      description: "SAP is a company.",
      description_source: "curated_llm",
      human_review_status: "accepted",
      reviewed_at: "2026-07-28T09:00:00Z",
    });
  });

  it("saves an inline description edit as accepted", async () => {
    const user = userEvent.setup();
    const { onOpenGraph } = renderDetailsPanel();

    await user.click(screen.getByRole("button", { name: "Edit entity description" }));
    const editor = screen.getByRole("textbox", { name: "Entity description" });
    await user.clear(editor);
    await user.type(editor, "SAP entwickelt Unternehmenssoftware.");
    await user.click(screen.getByRole("button", { name: "Save & accept" }));

    await waitFor(() =>
      expect(api.reviewEntityDescription).toHaveBeenCalledWith(
        "company:sap",
        "accepted",
        "SAP entwickelt Unternehmenssoftware.",
      ),
    );
    expect(onOpenGraph).toHaveBeenCalledWith("SAP");
  });

  it("keeps a rejected description visible and allows resetting its review", async () => {
    const user = userEvent.setup();
    const onOpenGraph = vi.fn();
    const node: GraphNode = {
      ...sapNode,
      properties: {
        ...sapNode.properties,
        description_human_review_status: "rejected",
        description_human_reviewed_at: "2026-07-28T09:00:00Z",
      },
    };
    render(
      <DetailsPanel
        node={node}
        graph={{ ...graph, nodes: [node, priorLabsNode, aiNode] }}
        visibleTypes={new Set(["Company", "Startup", "Topic"])}
        visibleRelations={new Set(["ACQUIRED", "HAS_TOPIC"])}
        onOpenGraph={onOpenGraph}
        onEntityUpdated={vi.fn()}
        onNodeSelect={vi.fn()}
      />,
    );

    expect(screen.getByText("SAP is a company.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Reset entity description review" }));

    await waitFor(() =>
      expect(api.reviewEntityDescription).toHaveBeenCalledWith(
        "company:sap",
        "unreviewed",
        undefined,
      ),
    );
    expect(onOpenGraph).toHaveBeenCalledWith("SAP");
  });

  it("adds an alias and asks the app to refresh the entity", async () => {
    const user = userEvent.setup();
    const { onEntityUpdated } = renderDetailsPanel();

    expect(screen.queryByLabelText("New alias")).not.toBeInTheDocument();
    expect(screen.getByLabelText("About aliases")).toHaveAttribute(
      "title",
      expect.stringMatching(/automatisch zusammengeführt/i),
    );
    await user.click(screen.getByRole("button", { name: "Add alias" }));
    await user.type(screen.getByLabelText("New alias"), "SAP SE");
    await user.click(screen.getByRole("button", { name: "Add" }));

    await waitFor(() => expect(api.addEntityAlias).toHaveBeenCalledWith("company:sap", "SAP SE"));
    expect(onEntityUpdated).toHaveBeenCalledWith({
      node_id: "company:sap",
      name: "SAP",
      aliases: ["SAP SE"],
      merged_node_ids: [],
    });
    expect(screen.queryByLabelText("New alias")).not.toBeInTheDocument();
  });

  it("filters relationship claims by review, supported, and all", async () => {
    const user = userEvent.setup();
    renderDetailsPanel();

    await screen.findByText("Conflicting direction");
    expect(screen.getByText(/SAP uebernimmt das junge KI-Startup Prior Labs/i)).toBeInTheDocument();
    expect(screen.queryByText("SAP arbeitet an KI-Produkten.")).not.toBeInTheDocument();

    const filters = screen.getByLabelText("Relationship filter");

    await user.click(within(filters).getByRole("button", { name: /Supported/i }));
    expect(screen.getByText(/SAP arbeitet an KI-Produkten/i)).toBeInTheDocument();
    expect(screen.queryByText("SAP uebernimmt das junge KI-Startup Prior Labs.")).not.toBeInTheDocument();

    await user.click(within(filters).getByRole("button", { name: /All relationships/i }));
    expect(screen.getByText(/SAP arbeitet an KI-Produkten/i)).toBeInTheDocument();
    expect(screen.getByText(/SAP uebernimmt das junge KI-Startup Prior Labs/i)).toBeInTheDocument();
  });

  it("submits a review decision and refreshes claim data", async () => {
    const user = userEvent.setup();
    const { onOpenGraph } = renderDetailsPanel();

    await screen.findByText("Conflicting direction");
    await user.click(screen.getByRole("button", { name: "Accept" }));

    await waitFor(() =>
      expect(api.reviewClaim).toHaveBeenCalledWith(
        "company:sap",
        "ACQUIRED",
        "startup:prior-labs",
        "accepted",
      ),
    );
    expect(api.nodeClaims).toHaveBeenCalledTimes(2);
    expect(onOpenGraph).toHaveBeenCalledWith("SAP");
  });

  it("shows the latest article extraction trace for the selected entity", async () => {
    const user = userEvent.setup();
    vi.mocked(api.nodeClaims).mockResolvedValue(claimsResponse([], [latestEntityMention()]));
    renderDetailsPanel();

    await screen.findByText("Latest SAP source");
    expect(screen.getByRole("heading", { name: "Latest Article Extractions" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /5 source articles/i })).toBeInTheDocument();
    expect(screen.queryByText(/Latest SAP evidence/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Older SAP source")).not.toBeInTheDocument();

    const traceLink = screen.getByRole("link", { name: /Open trace/i });
    expect(traceLink).toHaveAttribute(
      "href",
      "http://localhost:5001/#/experiments/2/traces/tr-latest",
    );
    expect(traceLink).toHaveTextContent(
      /Article extracted 01 Jun 2026, \d{2}:00:00 \| Published 16 May 2026/i,
    );
    expect(within(traceLink).queryByText(/CEST|MENTIONS/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /5 source articles/i }));

    expect(screen.getByText("Older SAP source")).toBeInTheDocument();
    expect(screen.getByText("Earlier SAP source")).toBeInTheDocument();
    expect(screen.getByText("Oldest SAP source")).toBeInTheDocument();
    expect(screen.queryByText("Earliest SAP source")).not.toBeInTheDocument();
  });

  it("shows the review trigger without source history", async () => {
    vi.mocked(api.nodeClaims).mockResolvedValue(claimsResponse([noLongerFoundClaimWithRepeatedArticle()]));
    renderDetailsPanel();

    await screen.findByText("Missing in latest extraction");
    expect(screen.queryByText("No current support")).not.toBeInTheDocument();
    expect(screen.queryByText("1 source article")).not.toBeInTheDocument();
    expect(screen.getByText(/Bunch review source/i)).toBeInTheDocument();
    expect(screen.getByText(/Missing in latest extraction/i)).toBeInTheDocument();
    expect(screen.getByText("Review source")).toBeInTheDocument();
    expect(screen.queryByText(/Latest run did not extract this relationship/i)).not.toBeInTheDocument();
    expect(screen.getByText("Evidence")).toBeInTheDocument();
    expect(screen.queryByText(/Previously extracted/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Bunch was founded by Alice Example/i)).toBeInTheDocument();
    expect(screen.queryByText("Source evidence")).not.toBeInTheDocument();
    expect(screen.queryByText(/Published/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Earlier evidence/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Accept" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reject" })).toBeInTheDocument();
    expect(screen.queryByText(/Technical details/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Older duplicate Bunch evidence.")).not.toBeInTheDocument();
  });

  it("treats support changes with active sources as supported", async () => {
    vi.mocked(api.nodeClaims).mockResolvedValue(claimsResponse([supportChangedButCurrentlySupportedClaim()]));
    renderDetailsPanel();

    await screen.findByText("Supported");
    expect(screen.getByText("0 need review | 1 supported | 0 accepted | 0 rejected")).toBeInTheDocument();
    expect(screen.queryByText("Support changed")).not.toBeInTheDocument();
    expect(screen.queryByText(/Supporting evidence changed/i)).not.toBeInTheDocument();
    expect(screen.getByText("1 source")).toBeInTheDocument();
    expect(screen.getByText("Source")).toBeInTheDocument();
    expect(screen.getByText(/SAP arbeitet weiter an KI-Produkten/i)).toBeInTheDocument();
  });

  it("does not crash when claims are missing source article summaries", async () => {
    vi.mocked(api.nodeClaims).mockResolvedValue(claimsResponse([legacySupportedClaimWithoutSourceArticles()]));
    renderDetailsPanel();

    await screen.findByText("Supported");
    expect(screen.getByText("2 sources")).toBeInTheDocument();
    expect(screen.getByText(/SAP arbeitet an KI-Produkten/i)).toBeInTheDocument();
  });

  it("keeps rejected claims visible as reviewed feedback and allows reset", async () => {
    const user = userEvent.setup();
    vi.mocked(api.nodeClaims).mockResolvedValue(claimsResponse([rejectedClaim()]));
    renderDetailsPanel();

    await screen.findByText("Rejected");
    expect(screen.getByText("Reviewed source")).toBeInTheDocument();
    expect(screen.queryByText("Wrong direction.")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Accept" })).toBeInTheDocument();
    expect(screen.queryByText(/Technical details/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Reset" }));

    await waitFor(() =>
      expect(api.reviewClaim).toHaveBeenCalledWith(
        "company:sap",
        "ACQUIRED",
        "startup:prior-labs",
        "unreviewed",
      ),
    );
  });
});
