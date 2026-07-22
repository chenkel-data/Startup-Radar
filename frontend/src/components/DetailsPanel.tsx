import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowRightLeft,
  ChevronDown,
  ChevronRight,
  Check,
  CheckCircle2,
  ExternalLink,
  FileText,
  Info,
  Network,
  Pencil,
  Plus,
  RotateCcw,
  Tags,
  X,
} from "lucide-react";
import { api } from "../lib/api";
import { stringValue } from "../lib/helpers";
import type {
  ClaimAssertion,
  ClaimSourceArticle,
  EntityAliasResult,
  EntityDescriptionReviewDecision,
  GraphEdge,
  GraphNode,
  GraphResponse,
  NodeClaim,
  NodeClaimsResponse,
} from "../types/graph";

type Props = {
  node?: GraphNode;
  graph: GraphResponse;
  visibleTypes: Set<string>;
  visibleRelations: Set<string>;
  onOpenGraph: (name: string) => void;
  onEntityUpdated: (result: EntityAliasResult) => void | Promise<void>;
  onNodeSelect: (node: GraphNode | undefined) => void;
};

type Connection = {
  edge: GraphEdge;
  counterpart: GraphNode;
};

type RelatedEntity = {
  node: GraphNode;
  weight: number;
  relations: string[];
};

type TraceReference = {
  article_id: string;
  article_title?: string;
  article_url?: string;
  source_name?: string;
  published_at?: string;
  processed_at?: string;
  relationship: string;
  trace_id?: string;
  mlflow_url?: string;
};

type ProfileTraceReference = {
  status?: string;
  decision?: string;
  confidence?: string;
  traceId?: string;
  experimentId?: string;
  traceUrl?: string;
  tracedAt?: string;
  humanReviewStatus?: EntityDescriptionReviewDecision;
  humanReviewedAt?: string;
};

const DISPLAY_KEYS = [
  "canonical_name",
  "concept_id",
  "ontology_version",
  "semantic_boundary",
  "category",
  "stage",
  "amount",
  "currency",
  "published_at",
  "announced_at",
  "evidence_status",
  "evidence",
  "source_name",
  "url",
];

const ARTICLE_FIELD_ORDER = [
  "id",
  "title",
  "url",
  "author",
  "published_at",
  "source_name",
  "source_url",
  "summary",
  "tags",
  "text",
];

const HIDDEN_ARTICLE_FIELDS = new Set(["raw_extracted_entities", "trace_provenance"]);
const RAW_EXTRACTION_GROUPS: Array<{ key: string; label: string }> = [
  { key: "startups", label: "Startups" },
  { key: "investors", label: "Investors" },
  { key: "people", label: "People" },
  { key: "topics", label: "Topics" },
  { key: "companies", label: "Companies" },
  { key: "relationships", label: "Relationships" },
];

type RawExtractionItem = Record<string, unknown>;
type RawExtraction = Record<string, RawExtractionItem[]>;
type ClaimFilter = "review" | "supported" | "reviewed" | "all";

const INITIAL_VISIBLE_CLAIMS = 6;
const CLAIM_VISIBLE_STEP = 6;
const COLLAPSIBLE_EVIDENCE_CHAR_LIMIT = 180;
const ALIAS_EDITABLE_TYPES = new Set(["Startup", "Investor", "Company", "Person"]);

type ClaimHistoryModel = {
  articles: ClaimSourceArticle[];
  currentCount: number;
  totalCount: number;
};

type EntityExtractionReference = {
  edgeId: string;
  relationship: string;
  articleId?: string;
  articleTitle: string;
  articleUrl?: string;
  processedAt?: string;
  publishedAt?: string;
  evidence?: string;
  traceUrl?: string;
  assertion: ClaimAssertion;
};

export function DetailsPanel({
  node,
  graph,
  visibleTypes,
  visibleRelations,
  onOpenGraph,
  onEntityUpdated,
  onNodeSelect,
}: Props) {
  const [claimData, setClaimData] = useState<NodeClaimsResponse | undefined>();
  const [claimLoading, setClaimLoading] = useState(false);
  const [claimError, setClaimError] = useState<string | undefined>();
  const [reviewingEdgeId, setReviewingEdgeId] = useState<string | undefined>();
  const [aliasValue, setAliasValue] = useState("");
  const [aliasEditing, setAliasEditing] = useState(false);
  const [aliasSaving, setAliasSaving] = useState(false);
  const [aliasRemoving, setAliasRemoving] = useState<string | undefined>();
  const [aliasError, setAliasError] = useState<string | undefined>();
  const nodesById = useMemo(() => new Map(graph.nodes.map((entry) => [entry.id, entry])), [graph.nodes]);

  useEffect(() => {
    setAliasValue("");
    setAliasEditing(false);
    setAliasSaving(false);
    setAliasRemoving(undefined);
    setAliasError(undefined);
  }, [node?.id]);

  useEffect(() => {
    if (!node || node.type === "Article") {
      setClaimData(undefined);
      setClaimError(undefined);
      setClaimLoading(false);
      return;
    }

    let cancelled = false;
    setClaimLoading(true);
    setClaimError(undefined);
    void api
      .nodeClaims(node.id)
      .then((response) => {
        if (!cancelled) setClaimData(response);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setClaimData(undefined);
          setClaimError(error instanceof Error ? error.message : "Claims request failed");
        }
      })
      .finally(() => {
        if (!cancelled) setClaimLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [node]);

  const connections = useMemo<Connection[]>(() => {
    if (!node) return [];

    return graph.edges
      .filter(
        (edge) =>
          visibleRelations.has(edge.label) &&
          (edge.source === node.id || edge.target === node.id),
      )
      .map((edge) => {
        const counterpartId = edge.source === node.id ? edge.target : edge.source;
        const counterpart = nodesById.get(counterpartId);
        if (counterpart && !visibleTypes.has(counterpart.type)) return undefined;
        if (!counterpart) return undefined;
        return { edge, counterpart };
      })
      .filter((entry): entry is Connection => Boolean(entry));
  }, [graph.edges, node, nodesById, visibleRelations, visibleTypes]);

  const relationshipMix = useMemo(() => {
    const counts = new Map<string, number>();
    for (const relation of connections) {
      counts.set(relation.edge.label, (counts.get(relation.edge.label) ?? 0) + 1);
    }
    return Array.from(counts.entries()).sort((left, right) => right[1] - left[1]);
  }, [connections]);

  const relatedEntities = useMemo<RelatedEntity[]>(() => {
    const aggregation = new Map<string, { node: GraphNode; weight: number; relations: Set<string> }>();

    for (const relation of connections) {
      const existing = aggregation.get(relation.counterpart.id);
      if (!existing) {
        aggregation.set(relation.counterpart.id, {
          node: relation.counterpart,
          weight: 1,
          relations: new Set([relation.edge.label]),
        });
        continue;
      }
      existing.weight += 1;
      existing.relations.add(relation.edge.label);
    }

    return Array.from(aggregation.values())
      .map((entry) => ({
        node: entry.node,
        weight: entry.weight,
        relations: Array.from(entry.relations).sort(),
      }))
      .sort((left, right) => right.weight - left.weight || left.node.label.localeCompare(right.node.label));
  }, [connections]);

  const aliases = useMemo(() => {
    if (!node) return [];
    const entityNames = new Set(
      [node.label, node.properties.name, node.properties.canonical_name]
        .map(normalizeAliasName)
        .filter(Boolean),
    );
    return Array.isArray(node.properties.aliases)
      ? node.properties.aliases
          .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
          .filter((entry) => entry.length > 0 && !entityNames.has(normalizeAliasName(entry)))
      : [];
  }, [node]);

  const propertyRows = useMemo(() => {
    if (!node) return [];
    return DISPLAY_KEYS.flatMap((key) => {
      const value = node.properties[key];
      if (value === undefined || value === null || value === "") return [];
      const formatted = formatProperty(key, value);
      if (!formatted) return [];
      return [{ key, value: formatted }];
    });
  }, [node]);

  const articleRows = useMemo(() => {
    if (!node || node.type !== "Article") return [];
    return orderedPropertyEntries(node.properties).flatMap(([key, value]) => {
      const formatted = formatArticleProperty(key, value);
      if (!formatted) return [];
      return [{ key, value: formatted }];
    });
  }, [node]);

  const articleTags = useMemo(() => {
    if (!node || node.type !== "Article") return [];
    return stringArrayValue(node.properties.tags);
  }, [node]);

  const rawExtraction = useMemo(() => {
    if (!node || node.type !== "Article") return undefined;
    return parseRawExtraction(node.properties.raw_extracted_entities);
  }, [node]);

  const rawExtractionCount = rawExtraction ? countRawExtraction(rawExtraction) : 0;
  const articleText = node?.type === "Article" ? stringValue(node.properties.text) : undefined;
  const traceRefs = useMemo(() => buildArticleTraceReferences(node), [node]);
  const profileTrace = useMemo(() => buildProfileTraceReference(node), [node]);

  const description = node ? stringValue(node.properties.description) : undefined;
  const summary = node && (description ?? buildSynopsis(node, connections.length, relationshipMix.length));

  const url = node ? stringValue(node.properties.url) : undefined;

  async function reviewClaim(claim: NodeClaim, decision: "accepted" | "rejected" | "unreviewed") {
    if (!node) return;
    setReviewingEdgeId(claim.edge_id);
    setClaimError(undefined);
    try {
      await api.reviewClaim(claim.source_id, claim.relationship, claim.target_id, decision);
      const updated = await api.nodeClaims(node.id);
      setClaimData(updated);
      onOpenGraph(node.label);
    } catch (error) {
      setClaimError(error instanceof Error ? error.message : "Claim review failed");
    } finally {
      setReviewingEdgeId(undefined);
    }
  }

  async function addAlias() {
    const alias = aliasValue.trim();
    if (!node || !alias) return;
    setAliasSaving(true);
    setAliasError(undefined);
    try {
      const result = await api.addEntityAlias(node.id, alias);
      setAliasValue("");
      setAliasEditing(false);
      await onEntityUpdated(result);
    } catch (error) {
      setAliasError(error instanceof Error ? error.message : "Alias could not be added");
    } finally {
      setAliasSaving(false);
    }
  }

  async function removeAlias(alias: string) {
    if (!node) return;
    setAliasRemoving(alias);
    setAliasError(undefined);
    try {
      const result = await api.removeEntityAlias(node.id, alias);
      await onEntityUpdated(result);
    } catch (error) {
      setAliasError(error instanceof Error ? error.message : "Alias could not be removed");
    } finally {
      setAliasRemoving(undefined);
    }
  }

  return (
    <aside className="details-panel" aria-label="Node intelligence">
      <div className="panel-head compact">
        <div className="panel-title">
          <Info size={16} />
          <h2>Node Intelligence</h2>
        </div>
        {node && (
          <button className="ghost-mini" onClick={() => onNodeSelect(undefined)}>
            <X size={14} />
            <span>Clear</span>
          </button>
        )}
      </div>

      {!node && (
        <div className="empty-intel">
          <p>Select a node in the graph to inspect context, relationship mix, and money flow paths.</p>
        </div>
      )}

      {node && (
        <div className="details-stack">
          <section className="node-hero">
            <span className={`entity-badge ${node.type.toLowerCase()}`}>{node.type}</span>
            <h3>{node.label}</h3>
            {node.type === "Article"
              ? summary && <p>{summary}</p>
              : summary && (
                  <EntityDescriptionReview
                    description={description}
                    fallbackDescription={summary}
                    nodeId={node.id}
                    onRefresh={() => onOpenGraph(node.label)}
                    profileTrace={profileTrace}
                  />
                )}

            {ALIAS_EDITABLE_TYPES.has(node.type) && (
              <div aria-label="Aliases" className="alias-list">
                {aliases.map((alias) => (
                  <span
                    aria-label={`${alias}, alias`}
                    className="alias-chip"
                    key={alias}
                    title={`Alias: ${alias}`}
                  >
                    {alias}
                    <button
                      aria-label={`Alias ${alias} entfernen`}
                      className="alias-remove"
                      disabled={aliasRemoving !== undefined}
                      onClick={() => void removeAlias(alias)}
                      title="Alias entfernen"
                      type="button"
                    >
                      <X size={11} />
                    </button>
                  </span>
                ))}
                {aliasEditing ? (
                  <form
                    className="alias-inline-form"
                    onSubmit={(event) => {
                      event.preventDefault();
                      void addAlias();
                    }}
                  >
                    <input
                      aria-label="New alias"
                      autoFocus
                      disabled={aliasSaving}
                      maxLength={200}
                      onChange={(event) => setAliasValue(event.target.value)}
                      placeholder="New alias"
                      value={aliasValue}
                    />
                    <button disabled={aliasSaving || !aliasValue.trim()} type="submit">
                      {aliasSaving ? "Adding…" : "Add"}
                    </button>
                    <button
                      aria-label="Cancel adding alias"
                      className="alias-cancel"
                      disabled={aliasSaving}
                      onClick={() => {
                        setAliasEditing(false);
                        setAliasValue("");
                        setAliasError(undefined);
                      }}
                      type="button"
                    >
                      <X size={12} />
                    </button>
                  </form>
                ) : (
                  <button
                    aria-label="Add alias"
                    className="alias-add-trigger"
                    onClick={() => setAliasEditing(true)}
                    title="Add alias"
                    type="button"
                  >
                    <Plus size={12} />
                    <span>Add alias</span>
                  </button>
                )}
                <span
                  aria-label="About aliases"
                  className="alias-info"
                  role="img"
                  tabIndex={0}
                  title="Ein Alias hilft, alternative Namen künftig dieser Entität zuzuordnen. Gibt es bereits genau eine Entität desselben Typs mit diesem Namen, werden die Knoten automatisch zusammengeführt. Das × entfernt den Alias wieder, macht eine frühere Zusammenführung aber nicht rückgängig."
                >
                  <Info size={13} />
                </span>
                {aliasError && <small className="alias-error" role="alert">{aliasError}</small>}
              </div>
            )}
          </section>

          {node.type === "Article" && (
            <section className="intel-section article-inspector">
              <div className="section-head">
                <FileText size={16} />
                <h4>Article</h4>
                {articleTags.length > 0 && <span>{articleTags.length} tags</span>}
              </div>

              {articleTags.length > 0 ? (
                <div className="tag-cloud" aria-label="Article tags">
                  {articleTags.map((tag) => (
                    <span key={tag}>
                      <Tags size={12} />
                      {tag}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="empty-text">No tags stored for this article yet.</p>
              )}

              <details className="article-expand">
                <summary>Raw LLM extraction{rawExtractionCount > 0 ? ` (${rawExtractionCount})` : ""}</summary>
                {rawExtraction && rawExtractionCount > 0 ? (
                  <div className="raw-extraction">
                    {RAW_EXTRACTION_GROUPS.map(({ key, label }) => {
                      const items = rawExtraction[key] ?? [];
                      if (items.length === 0) return null;
                      return (
                        <section key={key} className="raw-group">
                          <div className="raw-group-head">
                            <strong>{label}</strong>
                            <small>{items.length}</small>
                          </div>
                          <div className="raw-list">
                            {items.map((item, index) => (
                              <article key={`${key}-${index}`} className="raw-row">
                                <strong>{rawItemTitle(key, item)}</strong>
                                {rawItemMeta(key, item) && <small>{rawItemMeta(key, item)}</small>}
                                {rawItemEvidence(item) && <p>{rawItemEvidence(item)}</p>}
                              </article>
                            ))}
                          </div>
                        </section>
                      );
                    })}
                  </div>
                ) : (
                  <p className="empty-text compact">No raw extraction stored for this article yet.</p>
                )}
              </details>

              <details className="article-expand">
                <summary>Fields</summary>
                <dl className="property-grid article-fields">
                  {articleRows.map((entry) => (
                    <div key={entry.key}>
                      <dt>{entry.key.replace(/_/g, " ")}</dt>
                      <dd>{entry.value}</dd>
                    </div>
                  ))}
                </dl>
              </details>

              {articleText && (
                <details className="article-expand">
                  <summary>Text</summary>
                  <p className="article-text">{articleText}</p>
                </details>
              )}
            </section>
          )}

          {node.type === "Article" ? (
            <section className="intel-section trace-section">
              <div className="section-head">
                <Activity size={16} />
                <h4>Processing History</h4>
              </div>

              {traceRefs.length > 0 ? (
                <div className="trace-list">
                  {traceRefs.map((trace) => (
                    <TraceRow key={`${trace.article_id}-${trace.trace_id ?? trace.relationship}`} trace={trace} />
                  ))}
                </div>
              ) : (
                <p className="empty-text">No MLflow processing trace for this article.</p>
              )}
            </section>
          ) : (
            <>
              <EntityExtractionSection mentions={claimData?.mentions ?? []} loading={claimLoading} />
              <ClaimsSection
                claims={claimData?.claims ?? []}
                nodeLabel={node.label}
                loading={claimLoading}
                error={claimError}
                reviewingEdgeId={reviewingEdgeId}
                onReview={(claim, decision) => void reviewClaim(claim, decision)}
              />
            </>
          )}

          <section className="kpi-grid">
            <article className="kpi-card">
              <small>Connections</small>
              <strong>{connections.length}</strong>
            </article>
            <article className="kpi-card">
              <small>Relation types</small>
              <strong>{relationshipMix.length}</strong>
            </article>
            <article className="kpi-card">
              <small>Related entities</small>
              <strong>{relatedEntities.length}</strong>
            </article>
          </section>

          <section className="intel-section">
            <div className="section-head">
              <ArrowRightLeft size={16} />
              <h4>Relationship Mix</h4>
            </div>

            {relationshipMix.length > 0 ? (
              <div className="relation-chips">
                {relationshipMix.map(([label, count]) => (
                  <span key={label} className="relation-chip">
                    <strong>{label}</strong>
                    <small>{count}</small>
                  </span>
                ))}
              </div>
            ) : (
              <p className="empty-text">No direct relationships in this subgraph.</p>
            )}
          </section>

          <section className="intel-section">
            <div className="section-head">
              <Network size={16} />
              <h4>Related Entities</h4>
            </div>

            {relatedEntities.length > 0 ? (
              <div className="related-list">
                {relatedEntities.slice(0, 12).map((entry) => (
                  <button key={entry.node.id} className="related-row" onClick={() => onNodeSelect(entry.node)}>
                    <span className={`type-dot ${entry.node.type.toLowerCase()}`} />
                    <span className="related-meta">
                      <strong>{entry.node.label}</strong>
                      <small>
                        {entry.node.type} | {entry.weight} links | {entry.relations.slice(0, 2).join(", ")}
                      </small>
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <p className="empty-text">No related entities detected.</p>
            )}
          </section>

          {node.type !== "Article" && propertyRows.length > 0 && (
            <section className="intel-section">
              <div className="section-head">
                <Info size={16} />
                <h4>Properties</h4>
              </div>
              <dl className="property-grid">
                {propertyRows.map((entry) => (
                  <div key={entry.key}>
                    <dt>{entry.key.replace(/_/g, " ")}</dt>
                    <dd>{entry.value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          <div className="detail-actions">
            <button className="command-button" onClick={() => onOpenGraph(node.label)}>
              <Info size={15} />
              <span>Open focused subgraph</span>
            </button>

            {url && (
              <a className="command-button ghost" href={url} target="_blank" rel="noreferrer">
                <ExternalLink size={15} />
                <span>Open source</span>
              </a>
            )}
          </div>
        </div>
      )}
    </aside>
  );
}

function EntityDescriptionReview({
  nodeId,
  description,
  fallbackDescription,
  profileTrace,
  onRefresh,
}: {
  nodeId: string;
  description?: string;
  fallbackDescription: string;
  profileTrace?: ProfileTraceReference;
  onRefresh: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(description ?? fallbackDescription);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | undefined>();
  const humanStatus = profileTrace?.humanReviewStatus ?? "unreviewed";
  const hasWarning =
    humanStatus === "rejected" ||
    profileTrace?.status === "needs_human_review" ||
    profileTrace?.decision === "possible_wrong_merge" ||
    profileTrace?.decision === "conflicting_evidence";
  const stateClass =
    humanStatus === "accepted"
      ? "accepted"
      : humanStatus === "rejected"
        ? "rejected"
        : hasWarning
          ? "warning"
          : "";
  const StatusIcon =
    humanStatus === "accepted" ? CheckCircle2 : hasWarning ? AlertTriangle : Info;
  const statusSummary = entityDescriptionStatusSummary(profileTrace, Boolean(description));
  const reviewedAt = profileTrace?.humanReviewedAt
    ? `Human review ${formatDateTime(profileTrace.humanReviewedAt)}`
    : profileTrace?.tracedAt
      ? `Profile checked ${formatDateTime(profileTrace.tracedAt)}`
      : undefined;

  useEffect(() => {
    setEditing(false);
    setValue(description ?? fallbackDescription);
    setSaving(false);
    setError(undefined);
  }, [description, fallbackDescription, nodeId]);

  async function submitReview(
    decision: EntityDescriptionReviewDecision,
    editedDescription?: string,
  ) {
    setSaving(true);
    setError(undefined);
    try {
      await api.reviewEntityDescription(nodeId, decision, editedDescription);
      setEditing(false);
      onRefresh();
    } catch (reviewError) {
      setError(
        reviewError instanceof Error
          ? reviewError.message
          : "Entity description review failed",
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={`profile-status-row ${stateClass}`}>
      <div className="profile-status-head">
        <div className="profile-status-title">
          <StatusIcon size={14} />
          <small>Entity description</small>
        </div>
        <div className="profile-review-actions" aria-label="Entity description review">
          {profileTrace?.traceUrl && (
            <a
              aria-label="View entity description curation trace"
              href={profileTrace.traceUrl}
              rel="noreferrer"
              target="_blank"
              title="View curation trace"
            >
              <ExternalLink size={13} />
            </a>
          )}
          {!editing && description && humanStatus !== "accepted" && (
            <button
              aria-label="Accept entity description"
              disabled={saving}
              onClick={() => void submitReview("accepted")}
              title="Accept description"
              type="button"
            >
              <Check size={13} />
            </button>
          )}
          {!editing && description && humanStatus !== "rejected" && (
            <button
              aria-label="Reject entity description"
              className="danger"
              disabled={saving}
              onClick={() => void submitReview("rejected")}
              title="Reject description"
              type="button"
            >
              <X size={13} />
            </button>
          )}
          {!editing && (
            <button
              aria-label="Edit entity description"
              disabled={saving}
              onClick={() => {
                setEditing(true);
                setError(undefined);
              }}
              title="Edit description"
              type="button"
            >
              <Pencil size={13} />
            </button>
          )}
          {!editing && humanStatus !== "unreviewed" && (
            <button
              aria-label="Reset entity description review"
              disabled={saving}
              onClick={() => void submitReview("unreviewed")}
              title="Reset review"
              type="button"
            >
              <RotateCcw size={13} />
            </button>
          )}
        </div>
      </div>

      {editing ? (
        <form
          className="profile-description-form"
          onSubmit={(event) => {
            event.preventDefault();
            void submitReview("accepted", value.trim());
          }}
        >
          <textarea
            aria-label="Entity description"
            autoFocus
            disabled={saving}
            maxLength={4000}
            onChange={(event) => setValue(event.target.value)}
            rows={5}
            value={value}
          />
          <div>
            <button disabled={saving || !value.trim()} type="submit">
              {saving ? "Saving…" : "Save & accept"}
            </button>
            <button
              className="secondary"
              disabled={saving}
              onClick={() => {
                setEditing(false);
                setValue(description ?? fallbackDescription);
                setError(undefined);
              }}
              type="button"
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <p className="profile-description-text">{description ?? fallbackDescription}</p>
      )}

      <div className="profile-status-copy">
        <strong>{statusSummary}</strong>
        {reviewedAt && <span>{reviewedAt}</span>}
        {!description && (
          <span>The displayed synopsis is generated from graph context until you save a description.</span>
        )}
        {error && (
          <span className="profile-review-error" role="alert">
            {error}
          </span>
        )}
      </div>
    </div>
  );
}

function entityDescriptionStatusSummary(
  profileTrace: ProfileTraceReference | undefined,
  hasDescription: boolean,
): string {
  if (!hasDescription) return "No stored description yet.";
  if (profileTrace?.humanReviewStatus === "accepted") {
    return "Accepted by human review.";
  }
  if (profileTrace?.humanReviewStatus === "rejected") {
    return "Rejected by human review. The text remains visible and can be edited.";
  }
  if (profileTrace) return profileCurationSummary(profileTrace);
  return "Not yet reviewed.";
}

function EntityExtractionSection({
  mentions,
  loading,
}: {
  mentions: NodeClaim[];
  loading: boolean;
}) {
  const [showHistory, setShowHistory] = useState(false);
  const references = useMemo(() => buildEntityExtractionReferences(mentions), [mentions]);
  const latest = references[0];
  const historyReferences = references.slice(1, 4);
  const hasHistory = historyReferences.length > 0;
  const sourceArticleCount = countEntityExtractionArticles(references);

  useEffect(() => {
    setShowHistory(false);
  }, [mentions]);

  return (
    <section className="intel-section extraction-section">
      <div className="section-head extraction-section-head">
        <div className="extraction-icon" aria-hidden="true">
          <FileText size={15} />
        </div>
        <div className="extraction-title-copy">
          <h4>{extractionSectionTitle(sourceArticleCount)}</h4>
        </div>
        {references.length > 0 && (
          hasHistory ? (
            <button
              className="extraction-count-button"
              type="button"
              aria-expanded={showHistory}
              onClick={() => setShowHistory((current) => !current)}
            >
              {sourceArticleCountLabel(sourceArticleCount)}
              {showHistory ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            </button>
          ) : (
            <span className="extraction-count-label">{sourceArticleCountLabel(sourceArticleCount)}</span>
          )
        )}
      </div>

      {loading && references.length === 0 && (
        <p className="empty-text">Loading extraction provenance...</p>
      )}

      {!loading && !latest && (
        <p className="empty-text">No article extraction trace for this node.</p>
      )}

      {latest && <EntityExtractionTraceRow reference={latest} variant="latest" />}
      {showHistory && hasHistory && (
        <div className="extraction-history-list" aria-label="Previous article extraction traces">
          {historyReferences.map((reference, index) => (
            <EntityExtractionTraceRow
              key={entityExtractionReferenceKey(reference, index)}
              reference={reference}
              variant="history"
            />
          ))}
        </div>
      )}
    </section>
  );
}

function EntityExtractionTraceRow({
  reference,
  variant = "history",
}: {
  reference: EntityExtractionReference;
  variant?: "latest" | "history";
}) {
  const meta = [
    reference.processedAt ? `Article extracted ${formatExtractionDateTime(reference.processedAt)}` : undefined,
    reference.publishedAt ? `Published ${formatDate(reference.publishedAt)}` : undefined,
  ].filter(Boolean);
  const content = (
    <>
      <span className="trace-meta">
        <strong>{reference.articleTitle}</strong>
        <small>{meta.join(" | ")}</small>
      </span>
      {reference.traceUrl ? (
        <span className="trace-action">
          <span>Open trace</span>
          <ExternalLink size={13} />
        </span>
      ) : (
        <span className="trace-missing">No MLflow log</span>
      )}
    </>
  );

  if (reference.traceUrl) {
    return (
      <a
        className={`trace-row extraction-trace-row ${variant}`}
        href={reference.traceUrl}
        target="_blank"
        rel="noreferrer"
      >
        {content}
      </a>
    );
  }

  return <div className={`trace-row extraction-trace-row ${variant} muted`}>{content}</div>;
}

function profileCurationSummary(event: Pick<ProfileTraceReference, "status" | "decision" | "confidence">): string {
  const { status, decision } = event;
  if (decision === "exact_duplicate_evidence" || status === "skipped_exact_duplicate_evidence") {
    return "New evidence already matched the current description, so review was skipped.";
  }
  if (decision === "insufficient_evidence") {
    return "Review found the new evidence was insufficient to update the description.";
  }
  if (decision === "possible_wrong_merge") {
    return "Review found the new evidence may describe a different entity.";
  }
  if (decision === "conflicting_evidence") {
    return "Review found the new evidence conflicts with the current description.";
  }
  if (decision === "update_profile" || status === "curated" || status === "updated") {
    return "Review found the description should be updated, and curation updated it.";
  }
  if (decision === "keep_profile" || status === "reviewed_keep" || status === "kept") {
    return "Review found the current description should stay unchanged.";
  }
  if (
    status === "needs_human_review" ||
    decision === "needs_human_review"
  ) {
    return "Review could not safely decide how to update the description.";
  }
  if (status === "embedding_refreshed") return "Description search data was refreshed.";
  return "Description checked.";
}

function ClaimsSection({
  claims,
  nodeLabel,
  loading,
  error,
  reviewingEdgeId,
  onReview,
}: {
  claims: NodeClaim[];
  nodeLabel: string;
  loading: boolean;
  error?: string;
  reviewingEdgeId?: string;
  onReview: (claim: NodeClaim, decision: "accepted" | "rejected" | "unreviewed") => void;
}) {
  const counts = useMemo(() => claimCounts(claims), [claims]);
  const [activeFilter, setActiveFilter] = useState<ClaimFilter>("review");
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_CLAIMS);

  useEffect(() => {
    if (claims.length === 0) return;
    if (counts.review === 0 && activeFilter === "review") {
      setActiveFilter(counts.reviewed > 0 ? "reviewed" : counts.supported > 0 ? "supported" : "all");
    }
  }, [activeFilter, claims.length, counts.review, counts.reviewed, counts.supported]);

  useEffect(() => {
    setVisibleCount(INITIAL_VISIBLE_CLAIMS);
  }, [activeFilter, claims.length]);

  const filteredClaims = useMemo(() => {
    return claims.filter((claim) => {
      if (activeFilter === "review") return isReviewClaim(claim);
      if (activeFilter === "supported") return isSupportedClaim(claim);
      if (activeFilter === "reviewed") return isReviewedClaim(claim);
      return true;
    }).sort(compareClaims);
  }, [activeFilter, claims]);

  const visibleClaims = filteredClaims.slice(0, visibleCount);
  const remainingCount = Math.max(0, filteredClaims.length - visibleClaims.length);
  const nextVisibleCount = Math.min(CLAIM_VISIBLE_STEP, remainingCount);
  const overview = claimOverviewLabel(counts);

  return (
    <section className="intel-section claims-section">
      <div className="section-head">
        <ArrowRightLeft size={16} />
        <h4>Relationships</h4>
        {claims.length > 0 && <span>{claims.length}</span>}
      </div>

      {loading && <p className="empty-text">Loading claim provenance...</p>}
      {error && <p className="claim-error">{error}</p>}
      {!loading && claims.length === 0 && !error && (
        <p className="empty-text">No extracted relationship claims for this node.</p>
      )}

      {claims.length > 0 && (
        <>
          <p className="claim-overview">{overview}</p>

          <div className="claim-filter-list" aria-label="Relationship filter">
            <button
              className={activeFilter === "review" ? "active" : ""}
              onClick={() => setActiveFilter("review")}
              type="button"
            >
              <AlertTriangle size={13} />
              <span>Needs review</span>
              <strong>{counts.review}</strong>
            </button>
            <button
              className={activeFilter === "supported" ? "active" : ""}
              onClick={() => setActiveFilter("supported")}
              type="button"
            >
              <CheckCircle2 size={13} />
              <span>Supported</span>
              <strong>{counts.supported}</strong>
            </button>
            <button
              className={activeFilter === "reviewed" ? "active" : ""}
              onClick={() => setActiveFilter("reviewed")}
              type="button"
            >
              <CheckCircle2 size={13} />
              <span>Reviewed</span>
              <strong>{counts.reviewed}</strong>
            </button>
            <button
              className={activeFilter === "all" ? "active" : ""}
              onClick={() => setActiveFilter("all")}
              type="button"
            >
              <ArrowRightLeft size={13} />
              <span>All relationships</span>
              <strong>{claims.length}</strong>
            </button>
          </div>

          {filteredClaims.length === 0 ? (
            <p className="empty-text compact">{emptyClaimFilterLabel(activeFilter)}</p>
          ) : (
            <div className="claim-list">
              {visibleClaims.map((claim) => (
                <ClaimCard
                  key={claim.edge_id}
                  claim={claim}
                  nodeLabel={nodeLabel}
                  reviewing={reviewingEdgeId === claim.edge_id}
                  onReview={onReview}
                />
              ))}
            </div>
          )}

          {remainingCount > 0 && (
            <button
              className="claim-show-more"
              type="button"
              onClick={() => setVisibleCount((current) => current + CLAIM_VISIBLE_STEP)}
            >
              <ChevronDown size={14} />
              <span>Show {nextVisibleCount} more</span>
            </button>
          )}
        </>
      )}

    </section>
  );
}

function ClaimCard({
  claim,
  nodeLabel,
  reviewing,
  onReview,
}: {
  claim: NodeClaim;
  nodeLabel: string;
  reviewing: boolean;
  onReview: (claim: NodeClaim, decision: "accepted" | "rejected" | "unreviewed") => void;
}) {
  const status = claimStatus(claim);
  const history = useMemo(() => buildClaimHistory(claim), [claim]);
  const primaryArticle = primaryHistoryArticle(history);
  const primary = primaryAssertion(claim);
  const evidence = primaryArticle?.evidence ?? primary?.evidence;
  const articleTitle =
    articleDisplayTitle(primaryArticle?.article_title || primaryArticle?.article_url) ??
    articleDisplayTitle(primary?.article_title || primary?.article_url);
  const traceUrl = primaryArticle?.trace_url ?? (primary ? traceUrlForAssertion(primary) : undefined);
  const direction = claimDirectionView(claim, nodeLabel);
  const DirectionArrow = direction.undirected ? ArrowRightLeft : ArrowRight;
  const canAccept = claim.review_status !== "accepted";
  const canReject = claim.review_status !== "rejected";
  const canReset = claim.review_status === "accepted" || claim.review_status === "rejected";
  const supportLabel = claimSupportLabel(claim, history.currentCount);
  const sourceIntro = claimSourceIntro(claim);
  return (
    <article className={`claim-card ${status.className}`}>
      <div className="claim-card-body">
        <div className="claim-card-top">
          <span className="claim-card-status-line">
            <span className="claim-status">
              {status.icon === "warning" ? <AlertTriangle size={12} /> : <CheckCircle2 size={12} />}
              {status.label}
            </span>
          </span>
          {supportLabel && (
            <span className={`claim-support ${history.currentCount === 0 ? "empty" : "active"}`}>
              {supportLabel}
            </span>
          )}
        </div>

        <div className="claim-graph-line" title={claimDirectionTitle(direction, claim.relationship)}>
          <span className={`claim-node ${direction.source.isSelected ? "selected" : ""}`}>
            {direction.source.label}
          </span>
          <span className="claim-relation-code">{claim.relationship}</span>
          <DirectionArrow className="claim-arrow-icon" size={14} aria-hidden="true" />
          <span className={`claim-node ${direction.target.isSelected ? "selected" : ""}`}>
            {direction.target.label}
          </span>
        </div>

        {articleTitle && (
          <p className="claim-source-sentence">
            <span>{sourceIntro}</span>
            <strong>{articleTitle}</strong>
          </p>
        )}

        {evidence && <ClaimEvidence evidence={evidence} />}

        <div className="claim-footer">
          <div className="claim-actions">
            {traceUrl && (
              <a className="claim-trace-link" href={traceUrl} target="_blank" rel="noreferrer">
                <ExternalLink size={13} />
                <span>Open trace</span>
              </a>
            )}
            {canAccept && (
              <button disabled={reviewing} onClick={() => onReview(claim, "accepted")} type="button">
                <CheckCircle2 size={13} />
                <span>Accept</span>
              </button>
            )}
            {canReject && (
              <button
                className="danger"
                disabled={reviewing}
                onClick={() => onReview(claim, "rejected")}
                type="button"
              >
                <X size={13} />
                <span>Reject</span>
              </button>
            )}
            {canReset && (
              <button disabled={reviewing} onClick={() => onReview(claim, "unreviewed")} type="button">
                <ArrowRightLeft size={13} />
                <span>Reset</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </article>
  );
}

function ClaimEvidence({ evidence }: { evidence: string }) {
  const [expanded, setExpanded] = useState(false);
  const collapsible = evidence.trim().length > COLLAPSIBLE_EVIDENCE_CHAR_LIMIT;

  useEffect(() => {
    setExpanded(false);
  }, [evidence]);

  return (
    <blockquote className="claim-evidence">
      <span>Evidence</span>
      <q className={`claim-evidence-text ${collapsible && !expanded ? "collapsed" : ""}`}>
        {evidence}
      </q>
      {collapsible && (
        <button
          aria-expanded={expanded}
          className="claim-evidence-toggle"
          onClick={() => setExpanded((current) => !current)}
          type="button"
        >
          {expanded ? "Show less" : "Show full evidence"}
        </button>
      )}
    </blockquote>
  );
}

function claimSupportLabel(claim: NodeClaim, currentCount: number): string | undefined {
  if (claim.review_status === "needs_review" && currentCount === 0) return undefined;
  if (currentCount === 0) return "No active source";
  if (currentCount === 1) return "1 source";
  return `${currentCount} sources`;
}

function claimSourceIntro(claim: NodeClaim): string {
  if (claim.review_status === "needs_review") {
    return "Review source";
  }
  if (claim.review_status === "accepted" || claim.review_status === "rejected") return "Reviewed source";
  return "Source";
}

function claimHeadline(claim: NodeClaim): string {
  if (claim.review_status === "needs_review") {
    return reviewReasonHeadline(claim.review_reasons[0]);
  }
  if (claim.review_status === "accepted") return "Accepted";
  if (claim.review_status === "rejected") return "Rejected";
  return "Supported";
}

function reviewReasonHeadline(reason?: string): string {
  if (reason === "not_reproduced_same_article") return "Missing in latest extraction";
  if (reason === "direction_changed_same_article") return "Direction changed";
  if (reason === "inverse_direction") return "Conflicting direction";
  if (reason === "competing_transaction_type") return "Competing claim";
  return "Needs review";
}

function claimStatus(claim: NodeClaim): {
  label: string;
  className: string;
  icon: "warning" | "ok";
} {
  if (claim.review_status === "needs_review") {
    return { label: claimHeadline(claim), className: "needs-review", icon: "warning" };
  }
  if (claim.review_status === "rejected") {
    return { label: "Rejected", className: "rejected", icon: "warning" };
  }
  if (claim.review_status === "accepted") {
    return { label: "Accepted", className: "accepted", icon: "ok" };
  }
  return { label: "Supported", className: "supported", icon: "ok" };
}

function claimCounts(claims: NodeClaim[]): {
  review: number;
  supported: number;
  reviewed: number;
  accepted: number;
  rejected: number;
} {
  return claims.reduce(
    (counts, claim) => {
      if (isReviewClaim(claim)) counts.review += 1;
      if (isSupportedClaim(claim)) counts.supported += 1;
      if (isReviewedClaim(claim)) counts.reviewed += 1;
      if (claim.review_status === "accepted") counts.accepted += 1;
      if (claim.review_status === "rejected") counts.rejected += 1;
      return counts;
    },
    { review: 0, supported: 0, reviewed: 0, accepted: 0, rejected: 0 },
  );
}

function claimOverviewLabel(counts: {
  review: number;
  supported: number;
  reviewed: number;
  accepted: number;
  rejected: number;
}): string {
  const parts = [
    `${counts.review} need review`,
    `${counts.supported} supported`,
    `${counts.accepted} accepted`,
    `${counts.rejected} rejected`,
  ];
  return parts.join(" | ");
}

function isReviewClaim(claim: NodeClaim): boolean {
  return claim.review_status === "needs_review";
}

function isSupportedClaim(claim: NodeClaim): boolean {
  return !isReviewClaim(claim) && claim.review_status !== "rejected";
}

function isReviewedClaim(claim: NodeClaim): boolean {
  return claim.review_status === "accepted" || claim.review_status === "rejected";
}

function compareClaims(left: NodeClaim, right: NodeClaim): number {
  const leftRank = claimRank(left);
  const rightRank = claimRank(right);
  if (leftRank !== rightRank) return leftRank - rightRank;
  return claimSortLabel(left).localeCompare(claimSortLabel(right));
}

function claimRank(claim: NodeClaim): number {
  if (claim.review_status === "needs_review") return 0;
  if (claim.review_status === "unreviewed") return 1;
  if (claim.review_status === "accepted") return 2;
  if (claim.review_status === "rejected") return 3;
  return 5;
}

function emptyClaimFilterLabel(filter: ClaimFilter): string {
  if (filter === "review") return "No relationships currently need review.";
  if (filter === "supported") return "No supported relationships in this set.";
  return "No relationships in this set.";
}

type ClaimDirectionView = {
  source: { label: string; isSelected: boolean };
  target: { label: string; isSelected: boolean };
  undirected: boolean;
};

function claimDirectionView(claim: NodeClaim, nodeLabel: string): ClaimDirectionView {
  const selected = nodeLabel || "This node";
  const other = claim.counterparty.label;
  if (claim.direction === "incoming") {
    return {
      source: { label: other, isSelected: false },
      target: { label: selected, isSelected: true },
      undirected: false,
    };
  }
  return {
    source: { label: selected, isSelected: true },
    target: { label: other, isSelected: false },
    undirected: claim.direction === "undirected",
  };
}

function claimDirectionTitle(direction: ClaimDirectionView, relationship: string): string {
  const connector = direction.undirected ? "<->" : "->";
  return `${direction.source.label} ${connector} ${relationship} ${connector} ${direction.target.label}`;
}

function claimSortLabel(claim: NodeClaim): string {
  return `${claim.relationship}:${claim.direction}:${claim.counterparty.label}`;
}

function primaryHistoryArticle(history: ClaimHistoryModel): ClaimSourceArticle | undefined {
  return (
    history.articles.find((article) => article.review_source) ??
    history.articles.find((article) => article.status === "current" && article.evidence) ??
    history.articles.find((article) => article.status === "current") ??
    history.articles[0]
  );
}

function primaryAssertion(claim: NodeClaim): ClaimAssertion | undefined {
  const sortedAssertions = [...claim.assertions].sort(compareAssertionsByTime);
  return (
    sortedAssertions.find((assertion) => assertion.event === "asserted" && assertion.evidence) ??
    sortedAssertions.find((assertion) => assertion.event === "asserted") ??
    sortedAssertions[0]
  );
}

function buildEntityExtractionReferences(mentions: NodeClaim[]): EntityExtractionReference[] {
  return mentions
    .flatMap((mention) =>
      mention.assertions
        .filter((assertion) => assertion.event === "asserted")
        .map((assertion) => ({
          edgeId: mention.edge_id,
          relationship: mention.relationship,
          articleId: assertion.article_id,
          articleTitle:
            articleDisplayTitle(assertion.article_title) ??
            articleDisplayTitle(mention.counterparty.label) ??
            articleDisplayTitle(assertion.article_url) ??
            "Untitled article",
          articleUrl: assertion.article_url,
          processedAt: assertion.processed_at,
          publishedAt: assertion.published_at,
          evidence: assertion.evidence,
          traceUrl: traceUrlForAssertion(assertion),
          assertion,
        })),
    )
    .sort((left, right) => assertionTimeValue(right.assertion) - assertionTimeValue(left.assertion));
}

function sourceArticleCountLabel(count: number): string {
  return count === 1 ? "1 source article" : `${count} source articles`;
}

function extractionSectionTitle(count: number): string {
  return count === 1 ? "Latest Article Extraction" : "Latest Article Extractions";
}

function countEntityExtractionArticles(references: EntityExtractionReference[]): number {
  const articleKeys = new Set(references.map(entityExtractionArticleKey));
  return articleKeys.size;
}

function entityExtractionArticleKey(reference: EntityExtractionReference): string {
  return (
    reference.articleId ??
    reference.articleUrl ??
    `${reference.articleTitle}:${reference.publishedAt ?? ""}`
  );
}

function entityExtractionReferenceKey(reference: EntityExtractionReference, index: number): string {
  return [
    reference.edgeId,
    reference.articleId,
    reference.traceUrl,
    reference.articleUrl,
    reference.articleTitle,
    reference.processedAt ?? reference.publishedAt,
    index,
  ]
    .filter(Boolean)
    .join(":");
}

function buildClaimHistory(claim: NodeClaim): ClaimHistoryModel {
  const articles = Array.isArray(claim.source_articles)
    ? [...claim.source_articles].sort(compareSourceArticles)
    : [];
  return {
    articles,
    currentCount:
      articles.length > 0
        ? articles.filter((article) => article.status === "current").length
        : claim.active_support_count,
    totalCount: articles.length,
  };
}

function compareSourceArticles(left: ClaimSourceArticle, right: ClaimSourceArticle): number {
  const leftRank = sourceArticleRank(left);
  const rightRank = sourceArticleRank(right);
  if (leftRank !== rightRank) return leftRank - rightRank;
  return sourceArticleTimeValue(right) - sourceArticleTimeValue(left);
}

function sourceArticleRank(article: ClaimSourceArticle): number {
  if (article.review_source) return 0;
  if (article.status === "current") return 1;
  if (article.status === "no_longer_current" || article.status === "direction_changed") return 2;
  return 3;
}

function sourceArticleTimeValue(article: ClaimSourceArticle): number {
  const timestamp = article.latest_processed_at || article.published_at || "";
  const parsed = Date.parse(timestamp);
  return Number.isFinite(parsed) ? parsed : 0;
}

function compareAssertionsByTime(left: ClaimAssertion, right: ClaimAssertion): number {
  return assertionTimeValue(right) - assertionTimeValue(left);
}

function assertionTimeValue(assertion: ClaimAssertion): number {
  const timestamp = assertion.processed_at || assertion.published_at || "";
  const parsed = Date.parse(timestamp);
  return Number.isFinite(parsed) ? parsed : 0;
}

function traceUrlForAssertion(assertion: ClaimAssertion): string | undefined {
  return assertion.mlflow_trace_url ?? fallbackMlflowTraceUrl(assertion.trace_id, assertion.mlflow_experiment_id);
}

function buildProfileTraceReference(node?: GraphNode): ProfileTraceReference | undefined {
  if (!node || node.type === "Article") return undefined;
  const status = stringValue(node.properties.description_curation_status);
  const decision = stringValue(node.properties.description_review_decision);
  const traceId = stringValue(node.properties.description_trace_id);
  const experimentId = stringValue(node.properties.description_mlflow_experiment_id);
  const traceUrl = stringValue(node.properties.description_trace_url) ?? fallbackMlflowTraceUrl(traceId, experimentId);
  const tracedAt = stringValue(node.properties.description_traced_at);
  const humanReviewStatus = stringValue(
    node.properties.description_human_review_status,
  ) as EntityDescriptionReviewDecision | undefined;
  const humanReviewedAt = stringValue(node.properties.description_human_reviewed_at);
  const hasCurationSignal = Boolean(
    status || decision || traceId || traceUrl || tracedAt || humanReviewStatus || humanReviewedAt,
  );
  if (!hasCurationSignal) return undefined;
  const profileTrace: ProfileTraceReference = {
    status,
    decision,
    confidence: stringValue(node.properties.description_confidence),
    traceId,
    experimentId,
    traceUrl,
    tracedAt,
    humanReviewStatus,
    humanReviewedAt,
  };
  return Object.values(profileTrace).some((value) => Boolean(value)) ? profileTrace : undefined;
}

function articleDisplayTitle(value?: string): string | undefined {
  const title = value?.replace(/\s+/g, " ").trim();
  if (!title) return undefined;
  return title;
}

function TraceRow({ trace }: { trace: TraceReference }) {
  const title = trace.article_title || trace.article_url || "Untitled article";
  const meta = [
    trace.source_name,
    trace.processed_at ? `Trace created: ${formatDateTime(trace.processed_at)}` : undefined,
    trace.published_at ? `Published: ${formatDate(trace.published_at)}` : undefined,
    trace.relationship,
  ].filter(Boolean);
  const content = (
    <>
      <span className="trace-meta">
        <strong>{title}</strong>
        <small>{meta.join(" | ")}</small>
      </span>
      {trace.mlflow_url ? (
        <span className="trace-action">
          <span>Open full article trace</span>
          <ExternalLink size={13} />
        </span>
      ) : (
        <span className="trace-missing">No MLflow log</span>
      )}
    </>
  );

  if (trace.mlflow_url) {
    return (
      <a className="trace-row" href={trace.mlflow_url} target="_blank" rel="noreferrer" title={trace.mlflow_url}>
        {content}
      </a>
    );
  }

  return <div className="trace-row muted">{content}</div>;
}

function buildArticleTraceReferences(node: GraphNode | undefined): TraceReference[] {
  if (!node || node.type !== "Article") return [];
  const references = new Map<string, TraceReference>();

  const articleProvenance = provenanceTraceReferences(
    node.properties.trace_provenance,
    "ARTICLE PROCESSING",
    node.id,
  );
  if (articleProvenance.length > 0) {
    articleProvenance.forEach((reference) => addTraceReference(references, reference));
  } else {
    addTraceReference(references, articleTraceReference(node, "LATEST ARTICLE TRACE ONLY"));
  }

  return Array.from(references.values()).sort(compareTraceReferences);
}

function provenanceTraceReferences(
  value: unknown,
  relationship: string,
  ownerId: string,
): TraceReference[] {
  return stringArrayValue(value).flatMap((serialized, index) => {
    const provenance = safeJsonParse(serialized);
    if (!isRecord(provenance)) return [];

    const traceId = stringValue(provenance.trace_id);
    const mlflowUrl =
      stringValue(provenance.mlflow_trace_url) ??
      fallbackMlflowTraceUrl(traceId, stringValue(provenance.mlflow_experiment_id));
    if (!traceId && !mlflowUrl) return [];

    return [
      {
        article_id: `${ownerId}:provenance:${index}`,
        article_title: stringValue(provenance.article_title),
        article_url: stringValue(provenance.article_url),
        source_name: stringValue(provenance.source_name),
        published_at: stringValue(provenance.published_at),
        processed_at: stringValue(provenance.processed_at),
        relationship,
        trace_id: traceId,
        mlflow_url: mlflowUrl,
      },
    ];
  });
}

function addTraceReference(references: Map<string, TraceReference>, reference: TraceReference): void {
  const key = reference.trace_id ?? reference.mlflow_url ?? `${reference.article_id}:${reference.relationship}`;
  const existing = references.get(key);
  if (!existing || traceReferencePriority(reference.relationship) > traceReferencePriority(existing.relationship)) {
    references.set(key, reference);
  }
}

function traceReferencePriority(relationship: string): number {
  if (relationship === "MENTIONS" || relationship === "HAS_TOPIC") return 1;
  if (
    relationship === "FROM_SOURCE" ||
    relationship === "ARTICLE INGESTION" ||
    relationship.includes("latest article trace only")
  ) {
    return 0;
  }
  return 2;
}

function articleTraceReference(article: GraphNode, relationship: string): TraceReference {
  const traceId = stringValue(article.properties.trace_id);
  const experimentId = stringValue(article.properties.mlflow_experiment_id);
  return {
    article_id: article.id,
    article_title: stringValue(article.properties.title) ?? article.label,
    article_url: stringValue(article.properties.url),
    source_name: stringValue(article.properties.source_name),
    published_at: stringValue(article.properties.published_at),
    processed_at: stringValue(article.properties.processed_at),
    relationship,
    trace_id: traceId,
    mlflow_url: stringValue(article.properties.mlflow_trace_url) ?? fallbackMlflowTraceUrl(traceId, experimentId),
  };
}

function fallbackMlflowTraceUrl(traceId?: string, experimentId?: string): string | undefined {
  if (!traceId) return undefined;
  const baseUrl = String(import.meta.env.VITE_MLFLOW_URL ?? "http://localhost:5001").replace(/\/$/, "");
  const experiment = experimentId ?? String(import.meta.env.VITE_MLFLOW_EXPERIMENT_ID ?? "1");
  return `${baseUrl}/#/experiments/${encodeURIComponent(experiment)}/traces/${encodeURIComponent(traceId)}`;
}

function compareTraceReferences(left: TraceReference, right: TraceReference): number {
  const leftTime = Date.parse(left.processed_at ?? left.published_at ?? "");
  const rightTime = Date.parse(right.processed_at ?? right.published_at ?? "");
  if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) {
    return rightTime - leftTime;
  }
  return (left.article_title ?? "").localeCompare(right.article_title ?? "");
}

function buildSynopsis(
  node: GraphNode,
  connectionCount: number,
  relationTypes: number,
): string {
  if (node.type === "Startup") {
    return `${node.label} appears in ${connectionCount} graph links.`;
  }
  if (node.type === "Investor") {
    return `${node.label} has ${connectionCount} links across ${relationTypes} relationship families.`;
  }
  if (node.type === "Company") {
    return `${node.label} appears as a company node with ${connectionCount} nearby links.`;
  }
  return `${node.label} connects to ${connectionCount} nearby graph signals.`;
}

function formatProperty(key: string, value: unknown): string | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    if (key === "amount") return formatMoney(value, "EUR");
    return value.toLocaleString("en-GB");
  }

  if (typeof value === "string") {
    if (key === "evidence_status") {
      return evidenceStatusLabel(value);
    }
    if (key.endsWith("_at")) {
      return formatDate(value);
    }
    return value;
  }

  if (Array.isArray(value)) {
    const items = value
      .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
      .filter((entry) => entry.length > 0);
    if (items.length === 0) return undefined;
    return items.join(", ");
  }

  return undefined;
}

function formatArticleProperty(key: string, value: unknown): string | undefined {
  if (key === "text") {
    const text = stringValue(value);
    if (!text) return undefined;
    return `${text.length.toLocaleString("en-GB")} characters`;
  }
  return formatProperty(key, value);
}

function orderedPropertyEntries(properties: Record<string, unknown>): Array<[string, unknown]> {
  const entries = new Map(
    Object.entries(properties).filter(([key]) => !HIDDEN_ARTICLE_FIELDS.has(key)),
  );
  const ordered: Array<[string, unknown]> = [];

  for (const key of ARTICLE_FIELD_ORDER) {
    if (!entries.has(key)) continue;
    ordered.push([key, entries.get(key)]);
    entries.delete(key);
  }

  return [
    ...ordered,
    ...Array.from(entries.entries()).sort(([left], [right]) => left.localeCompare(right)),
  ];
}

function stringArrayValue(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0);
}

function parseRawExtraction(value: unknown): RawExtraction | undefined {
  const parsed = typeof value === "string" ? safeJsonParse(value) : value;
  if (!isRecord(parsed)) return undefined;

  const extraction: RawExtraction = {};
  for (const { key } of RAW_EXTRACTION_GROUPS) {
    const items = parsed[key];
    extraction[key] = Array.isArray(items) ? items.filter(isRecord) : [];
  }
  return extraction;
}

function countRawExtraction(extraction: RawExtraction): number {
  return RAW_EXTRACTION_GROUPS.reduce((total, group) => total + (extraction[group.key]?.length ?? 0), 0);
}

function safeJsonParse(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
}

function isRecord(value: unknown): value is RawExtractionItem {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function rawItemTitle(group: string, item: RawExtractionItem): string {
  if (group === "relationships") {
    const source = stringValue(item.source_name);
    const target = stringValue(item.target_name);
    if (source && target) return `${source} -> ${target}`;
  }
  return stringValue(item.name) ?? stringValue(item.type) ?? "Unnamed extraction";
}

function rawItemMeta(group: string, item: RawExtractionItem): string | undefined {
  const parts: string[] = [];
  const type = stringValue(item.type);
  const evidenceStatus = stringValue(item.evidence_status);
  const statusDefaulted = item.evidence_status_defaulted === true;
  const aliases = stringArrayValue(item.aliases);

  if (type && group === "relationships") parts.push(type);
  if (evidenceStatus) {
    parts.push(
      statusDefaulted
        ? "Unsure - missing/invalid status; not written to graph"
        : evidenceStatusLabel(evidenceStatus),
    );
  }
  if (aliases.length > 0) parts.push(`aliases: ${aliases.slice(0, 3).join(", ")}`);
  return parts.length > 0 ? parts.join(" | ") : undefined;
}

function evidenceStatusLabel(value: string): string {
  if (value === "stated") return "Stated";
  if (value === "attributed") return "Attributed";
  if (value === "unsure") return "Unsure - not written to graph";
  return value;
}

function rawItemEvidence(item: RawExtractionItem): string | undefined {
  const direct = stringValue(item.evidence);
  if (direct) return direct;
  const source = item.source;
  if (!isRecord(source)) return undefined;
  return stringValue(source.evidence);
}

function formatDate(value: string): string {
  const time = Date.parse(value);
  if (!Number.isFinite(time)) return value;
  return new Intl.DateTimeFormat("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(new Date(time));
}

function formatDateTime(value: string): string {
  const time = Date.parse(value);
  if (!Number.isFinite(time)) return value;
  return new Intl.DateTimeFormat("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
  }).format(new Date(time));
}

function formatExtractionDateTime(value: string): string {
  const time = Date.parse(value);
  if (!Number.isFinite(time)) return value;
  return new Intl.DateTimeFormat("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(time));
}

function formatMoney(amount?: number, currency?: string): string {
  if (amount === undefined || !Number.isFinite(amount)) return "n/a";
  const code = currency ?? "EUR";
  try {
    return new Intl.NumberFormat("en-GB", {
      style: "currency",
      currency: code,
      maximumFractionDigits: 0,
    }).format(amount);
  } catch {
    return `${Math.round(amount).toLocaleString("en-GB")} ${code}`;
  }
}

function normalizeAliasName(value: unknown): string {
  return typeof value === "string" ? value.trim().replace(/\s+/g, " ").toLocaleLowerCase() : "";
}
