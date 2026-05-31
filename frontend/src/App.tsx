import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, RefreshCcw, Waves } from "lucide-react";
import { FilterBar } from "./components/FilterBar";
import { GraphCanvas } from "./components/GraphCanvas";
import { IngestControl } from "./components/IngestControl";
import { InsightsPanel } from "./components/InsightsPanel";
import { SearchPanel } from "./components/SearchPanel";
import { DetailsPanel } from "./components/DetailsPanel";
import { ApiError, api } from "./lib/api";
import type {
  GraphNode,
  GraphResponse,
  SearchResult,
  TaskStatus
} from "./types/graph";

const ENTITY_TYPES = ["Startup", "Investor", "Company", "Person", "Topic", "Article", "Source"];
const FOCUSABLE_NODE_TYPES = new Set(["Startup", "Investor", "Company", "Person", "Topic"]);
const RELATION_TYPES = [
  "INVESTED_IN",
  "FOUNDED_BY",
  "EMPLOYED_BY",
  "PARTNERED_WITH",
  "MERGED_WITH",
  "ACQUIRED",
  "HAS_TOPIC",
  "MENTIONS",
  "FROM_SOURCE",
];
const ALL_TYPES = new Set(ENTITY_TYPES);
const ALL_RELATIONS = new Set(RELATION_TYPES);

export default function App() {
  const [graph, setGraph] = useState<GraphResponse>({ nodes: [], edges: [] });
  const [selectedNode, setSelectedNode] = useState<GraphNode | undefined>();
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  const [visibleTypes, setVisibleTypes] = useState(new Set(ALL_TYPES));
  const [visibleRelations, setVisibleRelations] = useState(new Set(ALL_RELATIONS));
  const [maxPages, setMaxPages] = useState(2);
  const [task, setTask] = useState<TaskStatus | undefined>();
  const [error, setError] = useState<string | undefined>();
  const [refreshKey, setRefreshKey] = useState(0);

  const loadGraph = useCallback(async (entity?: string) => {
    setGraphLoading(true);
    setError(undefined);
    try {
      const data = await api.graph(entity);
      setGraph(data);
      if (entity) {
        setSelectedNode(findGraphFocusNode(data.nodes, entity));
      } else {
        setSelectedNode(undefined);
      }
      setRefreshKey((value) => value + 1);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Graph request failed");
    } finally {
      setGraphLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph]);

  useEffect(() => {
    if (!task || (task.status !== "queued" && task.status !== "running")) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await api.ingestStatus(task.task_id);
        setTask(next);
        if (next.status === "succeeded") {
          void loadGraph();
        }
      } catch (exc) {
        if (exc instanceof ApiError && exc.status === 404) {
          setTask({
            ...task,
            status: "failed",
            completed_at: new Date().toISOString(),
            error: "Task no longer exists. The backend probably restarted; start a new ingestion run."
          });
          return;
        }
        setError(exc instanceof Error ? exc.message : "Task polling failed");
      }
    }, 2200);
    return () => window.clearInterval(timer);
  }, [task, loadGraph]);

  useEffect(() => {
    if (selectedNode && !visibleTypes.has(selectedNode.type)) {
      setSelectedNode(undefined);
    }
  }, [selectedNode, visibleTypes]);

  const selectNode = useCallback(
    (node: GraphNode | undefined) => {
      setSelectedNode(node);
      if (node && FOCUSABLE_NODE_TYPES.has(node.type)) {
        void loadGraph(node.label);
      }
    },
    [loadGraph],
  );

  async function runSearch(query: string) {
    setSearchLoading(true);
    setError(undefined);
    try {
      const data = await api.search(query);
      setResults(data);
      if (data[0]) {
        await loadGraph(data[0].name);
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Search failed");
    } finally {
      setSearchLoading(false);
    }
  }

  async function runIngest() {
    setError(undefined);
    try {
      const next = await api.startIngest(maxPages);
      setTask(next);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Ingest failed");
    }
  }

  async function runClear() {
    setError(undefined);
    try {
      const result = await api.clearGraph();
      setGraph({ nodes: [], edges: [] });
      setSelectedNode(undefined);
      setResults([]);
      return result;
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Clear failed");
      throw exc;
    }
  }

  const graphSummary = useMemo(() => {
    const startupCount = graph.nodes.filter((node) => node.type === "Startup").length;
    const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
    const investorIds = new Set(
      graph.nodes.filter((node) => node.type === "Investor").map((node) => node.id),
    );
    for (const edge of graph.edges) {
      const source = nodeById.get(edge.source);
      if (
        edge.label === "INVESTED_IN" &&
        source &&
        ["Company", "Person"].includes(source.type)
      ) {
        investorIds.add(source.id);
      }
    }
    const investorCount = investorIds.size;
    return { startupCount, investorCount };
  }, [graph.edges, graph.nodes]);

  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const node of graph.nodes) {
      counts[node.type] = (counts[node.type] ?? 0) + 1;
    }
    return counts;
  }, [graph.nodes]);

  const relationCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const edge of graph.edges) {
      counts[edge.label] = (counts[edge.label] ?? 0) + 1;
    }
    return counts;
  }, [graph.edges]);

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">
            <Waves size={14} />
            Venture Signal Radar
          </p>
          <h1>
            Startup intelligence with real graph context,
            <br />
            not dashboard noise.
          </h1>
          <p>
            Explore capital flows, investor overlap, and entity relationships from live ingestion
            in one surface.
          </p>
        </div>

        <div className="hero-metrics">
          <div className="stat-chip">
            <span className="stat-label">Startups</span>
            <strong className="stat-value">{graphSummary.startupCount}</strong>
          </div>
          <div className="stat-chip">
            <span className="stat-label">Investors</span>
            <strong className="stat-value">{graphSummary.investorCount}</strong>
          </div>
          <button className="icon-button subtle" onClick={() => void loadGraph()} aria-label="Refresh graph">
            <RefreshCcw size={18} className={graphLoading ? "spin" : ""} />
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      <div className="workspace">
        <aside className="left-rail">
          <SearchPanel
            results={results}
            loading={searchLoading}
            onSearch={runSearch}
            onSelect={(result) => {
              void loadGraph(result.name);
            }}
          />

          <IngestControl
            maxPages={maxPages}
            onMaxPagesChange={setMaxPages}
            onRun={runIngest}
            onClear={runClear}
            task={task}
          />

          <FilterBar
            selectedTypes={visibleTypes}
            selectedRelations={visibleRelations}
            typeCounts={typeCounts}
            relationCounts={relationCounts}
            onTypeChange={setVisibleTypes}
            onRelationChange={setVisibleRelations}
          />

          <InsightsPanel
            refreshKey={refreshKey}
            onEntityOpen={(name) => {
              void loadGraph(name);
            }}
          />
        </aside>

        <GraphCanvas
          graph={graph}
          visibleTypes={visibleTypes}
          visibleRelations={visibleRelations}
          selectedNode={selectedNode}
          onNodeSelect={selectNode}
        />

        <DetailsPanel
          node={selectedNode}
          graph={graph}
          visibleTypes={visibleTypes}
          visibleRelations={visibleRelations}
          onOpenGraph={(name) => {
            void loadGraph(name);
          }}
          onNodeSelect={selectNode}
        />
      </div>
    </main>
  );
}

function findGraphFocusNode(nodes: GraphNode[], entity: string): GraphNode | undefined {
  const needle = normalizeEntityKey(entity);
  if (!needle) return undefined;

  let partialMatch: GraphNode | undefined;
  for (const node of nodes) {
    const candidates = [
      node.id,
      node.label,
      valueAsString(node.properties.name),
      valueAsString(node.properties.canonical_name),
      ...valueAsStringArray(node.properties.aliases),
    ];

    for (const candidate of candidates) {
      const normalized = normalizeEntityKey(candidate);
      if (!normalized) continue;
      if (normalized === needle) return node;
      if (!partialMatch && normalized.includes(needle)) {
        partialMatch = node;
      }
    }
  }

  return partialMatch;
}

function normalizeEntityKey(value: unknown): string {
  if (typeof value !== "string") return "";
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
}

function valueAsString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : undefined;
}

function valueAsStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0);
}
