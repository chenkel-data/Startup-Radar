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

export type EntityCounts = {
  Startup: number;
  Investor: number;
  Company: number;
  Person: number;
  Topic: number;
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

export type ClaimSourceArticle = {
  article_url?: string;
  article_title?: string;
  published_at?: string;
  latest_processed_at?: string;
  status: "current" | "no_longer_current" | "direction_changed" | "historical" | string;
  evidence?: string;
  trace_url?: string;
  processing_count: number;
  review_source?: boolean;
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
  active_article_urls?: string[];
  source_articles?: ClaimSourceArticle[];
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
  roles: string[];
};

export type EntityAliasResult = {
  node_id: string;
  name: string;
  aliases: string[];
  merged_node_ids: string[];
};

export type EntityDescriptionReviewDecision = "accepted" | "rejected" | "unreviewed";

export type EntityDescriptionReviewResult = {
  node_id: string;
  description: string;
  description_source?: string;
  human_review_status: EntityDescriptionReviewDecision;
  reviewed_at?: string;
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
