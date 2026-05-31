import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./lib/api";
import type { GraphNode, GraphResponse, SearchResult } from "./types/graph";

const apiMock = vi.hoisted(() => ({
  graph: vi.fn(),
  nodeClaims: vi.fn(),
  reviewClaim: vi.fn(),
  search: vi.fn(),
  startIngest: vi.fn(),
  ingestStatus: vi.fn(),
  clearGraph: vi.fn(),
  insights: vi.fn(),
}));

vi.mock("./lib/api", () => ({
  ApiError: class ApiError extends Error {
    status: number;

    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  api: apiMock,
}));

vi.mock("./components/GraphCanvas", () => ({
  GraphCanvas: ({
    graph,
    selectedNode,
    onNodeSelect,
  }: {
    graph: GraphResponse;
    selectedNode?: GraphNode;
    onNodeSelect: (node: GraphNode | undefined) => void;
  }) => (
    <section aria-label="Mock graph">
      <span data-testid="selected-node">{selectedNode?.label ?? "none"}</span>
      {graph.nodes.map((node) => (
        <button key={node.id} type="button" onClick={() => onNodeSelect(node)}>
          Select {node.label}
        </button>
      ))}
    </section>
  ),
}));

vi.mock("./components/InsightsPanel", () => ({
  InsightsPanel: ({ onEntityOpen }: { onEntityOpen: (name: string) => void }) => (
    <section aria-label="Mock insights">
      <button type="button" onClick={() => onEntityOpen("SAP")}>
        Open SAP insight
      </button>
    </section>
  ),
}));

const aveliosNode: GraphNode = {
  id: "startup:avelios-medical",
  label: "Avelios Medical",
  type: "Startup",
  properties: { canonical_name: "Avelios Medical" },
};
const founderNode: GraphNode = {
  id: "person:christopher-muhr",
  label: "Christopher Muhr",
  type: "Person",
  properties: {},
};
const sapNode: GraphNode = {
  id: "company:sap",
  label: "SAP",
  type: "Company",
  properties: {},
};

const emptyGraph: GraphResponse = { nodes: [], edges: [] };
const aveliosGraph: GraphResponse = {
  nodes: [aveliosNode, founderNode, sapNode],
  edges: [
    {
      id: "founder-edge",
      source: "startup:avelios-medical",
      target: "person:christopher-muhr",
      label: "FOUNDED_BY",
      properties: {},
    },
  ],
};
const founderGraph: GraphResponse = {
  nodes: [founderNode, aveliosNode],
  edges: [
    {
      id: "founder-edge",
      source: "startup:avelios-medical",
      target: "person:christopher-muhr",
      label: "FOUNDED_BY",
      properties: {},
    },
  ],
};
const searchResult: SearchResult = {
  id: "startup:avelios-medical",
  name: "Avelios Medical",
  type: "Startup",
  score: 12.5,
  aliases: [],
};

describe("App graph exploration", () => {
  beforeEach(() => {
    vi.mocked(api.graph).mockImplementation(async (entity?: string) => {
      if (entity === "Avelios Medical") return aveliosGraph;
      if (entity === "Christopher Muhr") return founderGraph;
      if (entity === "SAP") return { nodes: [sapNode], edges: [] };
      return emptyGraph;
    });
    vi.mocked(api.search).mockResolvedValue([searchResult]);
    vi.mocked(api.nodeClaims).mockResolvedValue({ node_id: "", claims: [], mentions: [] });
    vi.mocked(api.clearGraph).mockResolvedValue({ status: "ok", deleted_nodes: 0 });
  });

  it("searches an entity and focuses the returned subgraph", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(api.graph).toHaveBeenCalledWith(undefined));
    await user.type(screen.getByLabelText("Search entities"), "Avelios");
    await user.click(screen.getByLabelText("Run search"));

    await waitFor(() => expect(api.search).toHaveBeenCalledWith("Avelios"));
    await waitFor(() => expect(api.graph).toHaveBeenCalledWith("Avelios Medical"));
    expect(screen.getByTestId("selected-node")).toHaveTextContent("Avelios Medical");
  });

  it("reloads a focused graph when a user selects another focusable node", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText("Search entities"), "Avelios");
    await user.click(screen.getByLabelText("Run search"));
    await screen.findByRole("button", { name: "Select Christopher Muhr" });

    await user.click(screen.getByRole("button", { name: "Select Christopher Muhr" }));

    await waitFor(() => expect(api.graph).toHaveBeenCalledWith("Christopher Muhr"));
    expect(screen.getByTestId("selected-node")).toHaveTextContent("Christopher Muhr");
  });
});
