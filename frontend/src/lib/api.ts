import type {
  GraphResponse,
  InsightRow,
  NodeClaimsResponse,
  SearchResult,
  TaskStatus
} from "../types/graph";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new ApiError(response.status, parseErrorDetail(detail) || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  graph(entity?: string, limit = 160) {
    const params = new URLSearchParams({ limit: String(limit) });
    if (entity) params.set("entity", entity);
    return request<GraphResponse>(`/graph?${params.toString()}`);
  },

  nodeClaims(nodeId: string) {
    return request<NodeClaimsResponse>(`/nodes/${encodeURIComponent(nodeId)}/claims`);
  },

  reviewClaim(
    sourceId: string,
    relationship: string,
    targetId: string,
    decision: "accepted" | "rejected" | "unreviewed",
  ) {
    return request<{ status: string; decision: string }>("/claims/review", {
      method: "POST",
      body: JSON.stringify({
        source_id: sourceId,
        relationship,
        target_id: targetId,
        decision,
      }),
    });
  },

  search(query: string) {
    return request<SearchResult[]>(`/search?q=${encodeURIComponent(query)}`);
  },

  startIngest(maxPages: number) {
    return request<TaskStatus>("/ingest", {
      method: "POST",
      body: JSON.stringify({
        max_pages: maxPages
      })
    });
  },

  ingestStatus(taskId: string) {
    return request<TaskStatus>(`/ingest/${taskId}`);
  },

  clearGraph() {
    return request<{ status: string; deleted_nodes: number }>("/graph", { method: "DELETE" });
  },

  insights(kind: "trending-startups" | "top-investors" | "co-investments" | "topic-clusters") {
    return request<InsightRow[]>(`/insights/${kind}`);
  }
};

function parseErrorDetail(value: string): string {
  if (!value) return "";
  try {
    const parsed = JSON.parse(value) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
  } catch {
    // Keep the original response text when the server did not return JSON.
  }
  return value;
}
