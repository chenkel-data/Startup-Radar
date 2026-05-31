import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowRightLeft,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  ExternalLink,
  FileText,
  Info,
  Network,
  Tags,
  X,
} from "lucide-react";
import { api } from "../lib/api";
import { stringValue } from "../lib/helpers";
import type {
  ClaimAssertion,
  ClaimReviewEvent,
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

const DISPLAY_KEYS = [
  "canonical_name",
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

export function DetailsPanel({
  node,
  graph,
  visibleTypes,
  visibleRelations,
  onOpenGraph,
  onNodeSelect,
}: Props) {
  const [claimData, setClaimData] = useState<NodeClaimsResponse | undefined>();
  const [claimLoading, setClaimLoading] = useState(false);
  const [claimError, setClaimError] = useState<string | undefined>();
  const [reviewingEdgeId, setReviewingEdgeId] = useState<string | undefined>();
  const nodesById = useMemo(() => new Map(graph.nodes.map((entry) => [entry.id, entry])), [graph.nodes]);

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
    return Array.isArray(node.properties.aliases)
      ? node.properties.aliases
          .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
          .filter((entry) => entry.length > 0)
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

  const summary =
    node &&
    (stringValue(node.properties.description) ??
      buildSynopsis(node, connections.length, relationshipMix.length));

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
            {summary && <p>{summary}</p>}

            {aliases.length > 0 && (
              <div className="alias-list">
                {aliases.slice(0, 8).map((alias) => (
                  <span key={alias}>{alias}</span>
                ))}
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
            <ClaimsSection
              claims={claimData?.claims ?? []}
              mentions={claimData?.mentions ?? []}
              nodeLabel={node.label}
              loading={claimLoading}
              error={claimError}
              reviewingEdgeId={reviewingEdgeId}
              onReview={(claim, decision) => void reviewClaim(claim, decision)}
            />
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

function ClaimsSection({
  claims,
  mentions,
  nodeLabel,
  loading,
  error,
  reviewingEdgeId,
  onReview,
}: {
  claims: NodeClaim[];
  mentions: NodeClaim[];
  nodeLabel: string;
  loading: boolean;
  error?: string;
  reviewingEdgeId?: string;
  onReview: (claim: NodeClaim, decision: "accepted" | "rejected" | "unreviewed") => void;
}) {
  const counts = useMemo(() => claimCounts(claims), [claims]);
  const [activeFilter, setActiveFilter] = useState<ClaimFilter>("review");
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_CLAIMS);
  const [expandedClaimId, setExpandedClaimId] = useState<string | undefined>();

  useEffect(() => {
    if (claims.length === 0) return;
    if (counts.review === 0 && activeFilter === "review") {
      setActiveFilter(counts.reviewed > 0 ? "reviewed" : counts.supported > 0 ? "supported" : "all");
    }
  }, [activeFilter, claims.length, counts.review, counts.reviewed, counts.supported]);

  useEffect(() => {
    setVisibleCount(INITIAL_VISIBLE_CLAIMS);
    setExpandedClaimId(undefined);
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
                  expanded={expandedClaimId === claim.edge_id}
                  reviewing={reviewingEdgeId === claim.edge_id}
                  onToggle={() =>
                    setExpandedClaimId((current) =>
                      current === claim.edge_id ? undefined : claim.edge_id,
                    )
                  }
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

      {mentions.length > 0 && (
        <details className="article-expand context-claims">
          <summary>Mentioned in articles ({mentions.length})</summary>
          <div className="mention-list">
            {mentions.map((mention) => (
              <div className="mention-row" key={mention.edge_id}>
                <strong>{mention.counterparty.label}</strong>
                <small>{mention.relationship}</small>
              </div>
            ))}
          </div>
        </details>
      )}
    </section>
  );
}

function ClaimCard({
  claim,
  nodeLabel,
  expanded,
  reviewing,
  onToggle,
  onReview,
}: {
  claim: NodeClaim;
  nodeLabel: string;
  expanded: boolean;
  reviewing: boolean;
  onToggle: () => void;
  onReview: (claim: NodeClaim, decision: "accepted" | "rejected" | "unreviewed") => void;
}) {
  const status = claimStatus(claim);
  const supportLabel =
    claim.active_support_count === 1
      ? "1 active supporting article"
      : `${claim.active_support_count} active supporting articles`;
  const primary = primaryAssertion(claim);
  const reason = reviewReasonSummary(claim);
  const evidence = primary?.evidence;
  const articleTitle = articleDisplayTitle(primary?.article_title || primary?.article_url);
  const run = primary ? runLabel(primary) : undefined;
  const direction = claimDirectionView(claim, nodeLabel);
  const DirectionArrow = direction.undirected ? ArrowRightLeft : ArrowRight;
  const latestReview = latestReviewEvent(claim);
  const canAccept = claim.review_status !== "accepted";
  const canReject = claim.review_status !== "rejected";
  const canReset = claim.review_status === "accepted" || claim.review_status === "rejected";
  return (
    <article className={`claim-card ${status.className} ${expanded ? "expanded" : ""}`}>
      <button
        className="claim-row-button"
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        {expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        <span className="claim-row-copy">
          <span className="claim-row-head">
            <span className="claim-row-kicker">Extracted relationship</span>
            <span className="claim-status">
              {status.icon === "warning" ? <AlertTriangle size={12} /> : <CheckCircle2 size={12} />}
              {status.label}
            </span>
          </span>
          <span className="claim-graph-line" title={claimDirectionTitle(direction, claim.relationship)}>
            <span className={`claim-node ${direction.source.isSelected ? "selected" : ""}`}>
              {direction.source.label}
            </span>
            <DirectionArrow className="claim-arrow-icon" size={14} aria-hidden="true" />
            <span className="claim-relation-code">{claim.relationship}</span>
            <DirectionArrow className="claim-arrow-icon" size={14} aria-hidden="true" />
            <span className={`claim-node ${direction.target.isSelected ? "selected" : ""}`}>
              {direction.target.label}
            </span>
          </span>
          {reason && (
            <span className="claim-summary-line claim-row-reason">
              <b>Reason for review:</b> {reason}
            </span>
          )}
          {evidence && (
            <span className="claim-summary-line claim-row-evidence">
              <b>Evidence:</b> {evidence}
            </span>
          )}
          {latestReview && (
            <span className="claim-summary-line claim-row-review">
              <b>Human feedback:</b> {reviewEventSummary(latestReview)}
            </span>
          )}
          <span className="claim-row-meta">
            <span>
              <b>Article:</b> {articleTitle ? `"${articleTitle}"` : "No title stored"}
            </span>
            <span>
              <b>Run:</b> {run ?? "No ingestion date stored"}
            </span>
            <span>
              <b>Support:</b> {supportLabel}
            </span>
          </span>
        </span>
      </button>

      {expanded && (
        <div className="claim-detail">
          <div className="claim-actions">
            {primary && traceUrlForAssertion(primary) && (
              <a className="claim-trace-link" href={traceUrlForAssertion(primary)} target="_blank" rel="noreferrer">
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

          {claim.review_history && claim.review_history.length > 0 && (
            <ReviewHistory events={claim.review_history} />
          )}

          {claim.assertions.length > 0 && (
            <details className="claim-history-panel" open>
              <summary className="claim-history-head">
                <strong>History</strong>
                <span>{claim.assertions.length} events</span>
              </summary>
              <ol className="claim-timeline">
                {claim.assertions.map((assertion, index) => (
                  <ClaimAssertionRow
                    key={`${assertion.trace_id ?? assertion.article_url ?? "assertion"}-${index}`}
                    assertion={assertion}
                  />
                ))}
              </ol>
            </details>
          )}
        </div>
      )}
    </article>
  );
}

function ReviewHistory({ events }: { events: ClaimReviewEvent[] }) {
  const ordered = [...events].reverse();
  return (
    <section className="review-history">
      <strong>Human feedback</strong>
      <div className="review-history-list">
        {ordered.map((event, index) => (
          <div key={`${event.reviewed_at ?? "review"}-${index}`} className="review-history-row">
            <span>{reviewEventSummary(event)}</span>
            {event.comment && <small>{event.comment}</small>}
          </div>
        ))}
      </div>
    </section>
  );
}

function ClaimAssertionRow({ assertion }: { assertion: ClaimAssertion }) {
  const title = articleDisplayTitle(assertion.article_title || assertion.article_url) || "Untitled article";
  const processedAt = assertion.processed_at ? formatDateTime(assertion.processed_at) : undefined;
  const publishedAt = assertion.published_at ? formatDate(assertion.published_at) : undefined;
  const traceUrl = traceUrlForAssertion(assertion);
  return (
    <li className={`assertion-row ${assertionEventClass(assertion)}`}>
      <span className="assertion-marker" aria-hidden="true" />
      <div className="assertion-main">
        <div className="assertion-line">
          <strong>{assertionEventLabel(assertion)}</strong>
          {processedAt && <span>Run: {processedAt}</span>}
        </div>
        <p className="assertion-article" title={assertion.article_title || assertion.article_url}>
          <b>Article:</b> "{title}"
        </p>
        {publishedAt && <p className="assertion-submeta">Published: {publishedAt}</p>}
        {assertion.evidence && (
          <p className="assertion-evidence">
            <b>Evidence:</b> {assertion.evidence}
          </p>
        )}
        <div className="assertion-foot">
          {assertion.job_run_id && <span>Run ID: {assertion.job_run_id.slice(0, 8)}</span>}
          {traceUrl && (
            <a href={traceUrl} target="_blank" rel="noreferrer">
              Open trace
              <ExternalLink size={12} />
            </a>
          )}
        </div>
      </div>
    </li>
  );
}

function assertionEventLabel(assertion: ClaimAssertion): string {
  if (assertion.event === "not_reproduced") return "Not reproduced";
  if (assertion.event === "direction_changed") return "Direction changed";
  return "Asserted";
}

function assertionEventClass(assertion: ClaimAssertion): string {
  if (assertion.event === "not_reproduced") return "omitted";
  if (assertion.event === "direction_changed") return "changed";
  return "asserted";
}

function claimStatus(claim: NodeClaim): {
  label: string;
  className: string;
  icon: "warning" | "ok";
} {
  if (claim.review_status === "needs_review") {
    return { label: "Review required", className: "needs-review", icon: "warning" };
  }
  if (claim.review_status === "rejected") {
    return { label: "Rejected", className: "rejected", icon: "warning" };
  }
  if (claim.review_status === "accepted") {
    return { label: "Accepted", className: "accepted", icon: "ok" };
  }
  if (claim.support_changed) {
    return { label: "Support changed", className: "support-changed", icon: "warning" };
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
  return claim.review_status === "needs_review" || claim.support_changed;
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
  if (claim.support_changed) return 1;
  if (claim.review_status === "unreviewed") return 2;
  if (claim.review_status === "accepted") return 3;
  if (claim.review_status === "rejected") return 4;
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

function primaryAssertion(claim: NodeClaim): ClaimAssertion | undefined {
  return (
    claim.assertions.find((assertion) => assertion.event === "asserted" && assertion.evidence) ??
    claim.assertions.find((assertion) => assertion.event === "asserted") ??
    claim.assertions[0]
  );
}

function latestReviewEvent(claim: NodeClaim): ClaimReviewEvent | undefined {
  const history = claim.review_history ?? [];
  if (history.length > 0) return history[history.length - 1];
  if (claim.review_status === "accepted" || claim.review_status === "rejected") {
    return {
      decision: claim.review_status,
      comment: claim.review_comment,
      reviewer: claim.reviewed_by,
      reviewed_at: claim.reviewed_at,
    };
  }
  return undefined;
}

function reviewEventSummary(event: ClaimReviewEvent): string {
  const decision = reviewDecisionLabel(event.decision);
  const reviewer = event.reviewer ? ` by ${event.reviewer}` : "";
  const reviewedAt = event.reviewed_at ? ` on ${formatDateTime(event.reviewed_at)}` : "";
  return `${decision}${reviewer}${reviewedAt}`;
}

function reviewDecisionLabel(decision?: string): string {
  if (decision === "accepted") return "Accepted";
  if (decision === "rejected") return "Rejected";
  if (decision === "unreviewed") return "Reset to unreviewed";
  return "Reviewed";
}

function traceUrlForAssertion(assertion: ClaimAssertion): string | undefined {
  return assertion.mlflow_trace_url ?? fallbackMlflowTraceUrl(assertion.trace_id, assertion.mlflow_experiment_id);
}

function runLabel(assertion: ClaimAssertion): string | undefined {
  const timestamp = assertion.processed_at || assertion.published_at;
  return timestamp ? formatDateTime(timestamp) : undefined;
}

function articleDisplayTitle(value?: string): string | undefined {
  const title = value?.replace(/\s+/g, " ").trim();
  if (!title) return undefined;
  return title;
}

function reviewReasonSummary(claim: NodeClaim): string | undefined {
  if (claim.review_reasons.length > 0) {
    return claim.review_reasons.map(reviewReasonLabel).join(" | ");
  }
  if (claim.support_changed) {
    return "Supporting evidence changed since this relationship was last processed";
  }
  return undefined;
}

function reviewReasonLabel(reason: string): string {
  if (reason === "not_reproduced_same_article") {
    return "Not reproduced when its source article was processed again";
  }
  if (reason === "direction_changed_same_article") {
    return "Direction changed when its source article was processed again";
  }
  if (reason === "inverse_direction") return "Conflicting direction between the same entities";
  if (reason === "competing_transaction_type") {
    return "Another article describes this pair using a competing transaction type";
  }
  return reason.replace(/_/g, " ");
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
