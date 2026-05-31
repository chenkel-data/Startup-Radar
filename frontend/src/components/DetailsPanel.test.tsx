import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DetailsPanel } from "./DetailsPanel";
import { api } from "../lib/api";
import type { GraphNode, GraphResponse, NodeClaim, NodeClaimsResponse } from "../types/graph";

const apiMock = vi.hoisted(() => ({
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

function claimsResponse(claims: NodeClaim[]): NodeClaimsResponse {
  return {
    node_id: "company:sap",
    claims,
    mentions: [],
  };
}

function renderDetailsPanel() {
  const onOpenGraph = vi.fn();
  const onNodeSelect = vi.fn();
  render(
    <DetailsPanel
      node={sapNode}
      graph={graph}
      visibleTypes={new Set(["Company", "Startup", "Topic"])}
      visibleRelations={new Set(["ACQUIRED", "HAS_TOPIC"])}
      onOpenGraph={onOpenGraph}
      onNodeSelect={onNodeSelect}
    />,
  );
  return { onOpenGraph, onNodeSelect };
}

describe("DetailsPanel claim review behavior", () => {
  beforeEach(() => {
    vi.mocked(api.nodeClaims).mockResolvedValue(claimsResponse([reviewClaim(), supportedClaim()]));
    vi.mocked(api.reviewClaim).mockResolvedValue({ status: "ok", decision: "accepted" });
  });

  it("filters relationship claims by review, supported, and all", async () => {
    const user = userEvent.setup();
    renderDetailsPanel();

    await screen.findByText("Review required");
    expect(screen.getByText("SAP uebernimmt das junge KI-Startup Prior Labs.")).toBeInTheDocument();
    expect(screen.queryByText("SAP arbeitet an KI-Produkten.")).not.toBeInTheDocument();

    const filters = screen.getByLabelText("Relationship filter");

    await user.click(within(filters).getByRole("button", { name: /Supported/i }));
    expect(screen.getByText("SAP arbeitet an KI-Produkten.")).toBeInTheDocument();
    expect(screen.queryByText("SAP uebernimmt das junge KI-Startup Prior Labs.")).not.toBeInTheDocument();

    await user.click(within(filters).getByRole("button", { name: /All relationships/i }));
    expect(screen.getByText("SAP arbeitet an KI-Produkten.")).toBeInTheDocument();
    expect(screen.getByText("SAP uebernimmt das junge KI-Startup Prior Labs.")).toBeInTheDocument();
  });

  it("submits a review decision and refreshes claim data", async () => {
    const user = userEvent.setup();
    const { onOpenGraph } = renderDetailsPanel();

    await screen.findByText("Review required");
    await user.click(screen.getByText("Review required"));
    await user.click(screen.getByRole("button", { name: /Accept/i }));

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

  it("keeps rejected claims visible as reviewed feedback and allows reset", async () => {
    const user = userEvent.setup();
    vi.mocked(api.nodeClaims).mockResolvedValue(claimsResponse([rejectedClaim()]));
    renderDetailsPanel();

    await screen.findByText("Rejected");
    expect(screen.getByText(/Human feedback:/i)).toBeInTheDocument();
    expect(screen.getByText(/Rejected by Editor/i)).toBeInTheDocument();

    await user.click(screen.getByText("Rejected"));
    expect(screen.getByText("Wrong direction.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Accept/i })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Reset/i }));

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
