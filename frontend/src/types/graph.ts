export type GraphNode = {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  label: string;
  properties: Record<string, unknown>;
};

export type GraphResponse = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type ClaimAssertion = {
  event: "asserted" | "not_reproduced" | "direction_changed" | "processed" | string;
  article_id?: string;
  article_title?: string;
  article_url?: string;
  source_name?: string;
  published_at?: string;
  job_run_id?: string;
  processed_at?: string;
  trace_id?: string;
  mlflow_trace_url?: string;
  mlflow_experiment_id?: string;
  evidence_status?: string;
  evidence?: string;
};

export type ClaimReviewEvent = {
  decision?: "accepted" | "rejected" | "unreviewed" | string;
  comment?: string | null;
  reviewer?: string | null;
  reviewed_at?: string;
};

export type NodeClaim = {
  edge_id: string;
  relationship: string;
  direction: "outgoing" | "incoming" | "undirected";
  counterparty: {
    id: string;
    label: string;
    type: string;
  };
  lifecycle_status: "supported" | "unsupported_by_latest_source_processing" | string;
  review_status: "unreviewed" | "needs_review" | "accepted" | "rejected" | string;
  review_reasons: string[];
  review_comment?: string | null;
  reviewed_by?: string | null;
  reviewed_at?: string;
  review_history?: ClaimReviewEvent[];
  support_changed: boolean;
  active_support_count: number;
  assertions: ClaimAssertion[];
  source_id: string;
  target_id: string;
};

export type NodeClaimsResponse = {
  node_id: string;
  claims: NodeClaim[];
  mentions: NodeClaim[];
};

export type SearchResult = {
  id: string;
  name: string;
  type: string;
  score: number;
  aliases: string[];
  description?: string;
};

export type TaskStatus = {
  task_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  name: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
  result?: Record<string, unknown>;
};

export type InsightRow = Record<string, string | number | string[] | null>;
