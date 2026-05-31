import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import ForceGraph2D, {
  type ForceGraphMethods,
  type NodeObject,
} from "react-force-graph-2d";
import { GitFork, Network, Search } from "lucide-react";
import { clamp, stringValue } from "../lib/helpers";
import type { GraphEdge, GraphNode, GraphResponse } from "../types/graph";

type Props = {
  graph: GraphResponse;
  visibleTypes: Set<string>;
  visibleRelations: Set<string>;
  selectedNode?: GraphNode;
  onNodeSelect: (node: GraphNode | undefined) => void;
};

type VizNode = GraphNode & {
  color: string;
  degree: number;
  importance: number;
  radius: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number;
  fy?: number;
};

type VizLink = GraphEdge & {
  color: string;
  emphasis: number;
  marker?: "review" | "changed";
};

type ReframeScope = "all" | "visible" | "active-neighborhood";

const NODE_COLORS: Record<string, string> = {
  Startup: "#0f766e",
  Investor: "#b45309",
  Company: "#6d28d9",
  Person: "#1d4ed8",
  Topic: "#0ea5a5",
  Article: "#475569",
  Source: "#334155",
};

const RELATION_COLORS: Record<string, string> = {
  INVESTED_IN: "#be123c",
  HAS_TOPIC: "#0f766e",
  FOUNDED_BY: "#1d4ed8",
  EMPLOYED_BY: "#2563eb",
  PARTNERED_WITH: "#0891b2",
  MERGED_WITH: "#9333ea",
  ACQUIRED: "#ef4444",
  FROM_SOURCE: "#64748b",
  MENTIONS: "#7c3aed",
};

const TYPE_PRIORITY: Record<string, number> = {
  Startup: 62,
  Investor: 58,
  Company: 54,
  Topic: 45,
  Person: 38,
  Article: 24,
  Source: 20,
};

const FINANCE_RELATIONS = new Set(["INVESTED_IN"]);
const FLOW_ORDER = ["Source", "Article", "Topic", "Person", "Startup", "Investor", "Company"];
const MAX_VISIBLE_NODES = 100;
const ALL_GRAPH_FIT_PADDING = 112;
const FOCUSED_GRAPH_FIT_PADDING = 156;

export function GraphCanvas({
  graph,
  visibleTypes,
  visibleRelations,
  selectedNode,
  onNodeSelect,
}: Props) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const forceRef = useRef<ForceGraphMethods<VizNode, VizLink> | undefined>(undefined);
  const autoFitRef = useRef(false);
  const reframeTimersRef = useRef<number[]>([]);
  const hoverNodeIdRef = useRef<string | undefined>();
  const lastNodeClickAtRef = useRef(0);
  const pendingBackgroundResetRef = useRef(false);

  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const [layoutMode, setLayoutMode] = useState<"force" | "flow">("force");
  const [focusDepth, setFocusDepth] = useState<1 | 2 | 3>(1);
  const [declutter, setDeclutter] = useState(true);
  const [hoverNodeId, setHoverNodeId] = useState<string | undefined>();

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;

    const updateSize = () => {
      setViewport({
        width: Math.max(360, frame.clientWidth),
        height: Math.max(520, frame.clientHeight),
      });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    return () => {
      for (const timer of reframeTimersRef.current) {
        window.clearTimeout(timer);
      }
    };
  }, []);

  const baseGraph = useMemo(() => {
    const nodes = graph.nodes.filter((node) => visibleTypes.has(node.type));
    const allowedNodeIds = new Set(nodes.map((node) => node.id));
    const edges = graph.edges.filter(
      (edge) =>
        visibleRelations.has(edge.label) &&
        allowedNodeIds.has(edge.source) &&
        allowedNodeIds.has(edge.target),
    );

    const degreeById = new Map<string, number>();
    for (const edge of edges) {
      degreeById.set(edge.source, (degreeById.get(edge.source) ?? 0) + 1);
      degreeById.set(edge.target, (degreeById.get(edge.target) ?? 0) + 1);
    }

    return { nodes, edges, degreeById };
  }, [graph, visibleRelations, visibleTypes]);

  const contextNodeId = useMemo(() => {
    if (selectedNode && baseGraph.nodes.some((node) => node.id === selectedNode.id)) {
      return selectedNode.id;
    }
    return highestDegreeNodeId(baseGraph.nodes, baseGraph.degreeById);
  }, [baseGraph.degreeById, baseGraph.nodes, selectedNode]);

  const visibleGraph = useMemo<{ nodes: VizNode[]; links: VizLink[]; hiddenCount: number }>(() => {
    if (baseGraph.nodes.length === 0) {
      return { nodes: [] as VizNode[], links: [] as VizLink[], hiddenCount: 0 };
    }

    let keptIds = new Set(baseGraph.nodes.map((node) => node.id));

    if (selectedNode && contextNodeId) {
      keptIds = bfsNodeIds(baseGraph.edges, contextNodeId, focusDepth);
    } else if (declutter && baseGraph.nodes.length > MAX_VISIBLE_NODES) {
      const ranked = [...baseGraph.nodes].sort((left, right) => {
        const leftScore = importanceScore(left, baseGraph.degreeById.get(left.id) ?? 0);
        const rightScore = importanceScore(right, baseGraph.degreeById.get(right.id) ?? 0);
        return rightScore - leftScore || left.label.localeCompare(right.label);
      });

      keptIds = new Set(ranked.slice(0, MAX_VISIBLE_NODES).map((node) => node.id));
      if (contextNodeId) {
        const contextNeighborhood = bfsNodeIds(baseGraph.edges, contextNodeId, 1);
        for (const id of contextNeighborhood) keptIds.add(id);
      }
    }

    const nodes = baseGraph.nodes.filter((node) => keptIds.has(node.id));
    const links = baseGraph.edges.filter((edge) => keptIds.has(edge.source) && keptIds.has(edge.target));
    const flowPositions =
      layoutMode === "flow"
        ? computeFlowPositions(nodes, viewport.width, viewport.height, baseGraph.degreeById)
        : new Map<string, { x: number; y: number }>();

    return {
      nodes: nodes.map((node) => {
        const degree = baseGraph.degreeById.get(node.id) ?? 0;
        const position = flowPositions.get(node.id);
        const mappedNode: VizNode = {
          ...node,
          color: NODE_COLORS[node.type] ?? "#64748b",
          degree,
          importance: importanceScore(node, degree),
          radius: baseRadius(node.type, degree),
          fx: position?.x,
          fy: position?.y,
        };
        return mappedNode;
      }),
      links: links.map((edge) => {
        const marker = linkMarker(edge);
        const mappedLink: VizLink = {
          ...edge,
          color: marker ? "#d97706" : RELATION_COLORS[edge.label] ?? "#8da2b7",
          emphasis: FINANCE_RELATIONS.has(edge.label) ? 1.4 : 1,
          marker,
        };
        return mappedLink;
      }),
      hiddenCount: baseGraph.nodes.length - nodes.length,
    };
  }, [
    baseGraph.degreeById,
    baseGraph.edges,
    baseGraph.nodes,
    contextNodeId,
    declutter,
    focusDepth,
    layoutMode,
    selectedNode,
    viewport.height,
    viewport.width,
  ]);

  const nodeMap = useMemo(
    () => new Map(visibleGraph.nodes.map((node) => [String(node.id), node])),
    [visibleGraph.nodes],
  );

  const graphData = useMemo(
    () => ({ nodes: visibleGraph.nodes, links: visibleGraph.links }),
    [visibleGraph.nodes, visibleGraph.links],
  );

  const labelsById = useMemo(
    () => new Map(visibleGraph.nodes.map((node) => [String(node.id), node.label])),
    [visibleGraph.nodes],
  );

  const activeNodeId = selectedNode?.id ?? hoverNodeId;

  const activeNeighborhood = useMemo(() => {
    const related = new Set<string>();
    if (!activeNodeId) return related;

    related.add(activeNodeId);
    for (const edge of visibleGraph.links) {
      const sourceId = String(edge.source);
      const targetId = String(edge.target);
      if (sourceId === activeNodeId) related.add(targetId);
      if (targetId === activeNodeId) related.add(sourceId);
    }
    return related;
  }, [activeNodeId, visibleGraph.links]);

  const visibleNodeIds = useMemo(
    () => new Set(visibleGraph.nodes.map((node) => String(node.id))),
    [visibleGraph.nodes],
  );

  const graphDensity = useMemo(() => {
    if (visibleGraph.nodes.length <= 1) return 0;
    return visibleGraph.links.length / visibleGraph.nodes.length;
  }, [visibleGraph.links.length, visibleGraph.nodes.length]);

  useEffect(() => {
    autoFitRef.current = false;
  }, [layoutMode, visibleGraph.links.length, visibleGraph.nodes.length]);

  const fitGraphToView = useCallback(
    (scope: ReframeScope, duration = 520, padding = 105) => {
      const view = forceRef.current;
      if (!view || visibleGraph.nodes.length === 0) return;

      const fitsScope = (node: NodeObject<VizNode>) => {
        if (!hasFinitePosition(node)) return false;
        if (scope === "active-neighborhood" && activeNeighborhood.size > 0) {
          return activeNeighborhood.has(String(node.id));
        }
        if (scope === "visible") return visibleNodeIds.has(String(node.id));
        return true;
      };

      if (!visibleGraph.nodes.some((node) => fitsScope(node))) return;

      view.zoomToFit(duration, padding, fitsScope);
      autoFitRef.current = true;
    },
    [activeNeighborhood, visibleGraph.nodes.length, visibleNodeIds],
  );

  const queueGraphReframe = useCallback(
    (scope: ReframeScope, padding = 105) => {
      for (const timer of reframeTimersRef.current) {
        window.clearTimeout(timer);
      }

      const timings = layoutMode === "force" ? [0, 180, 620] : [0, 120];
      reframeTimersRef.current = timings.map((delay, index) =>
        window.setTimeout(() => {
          window.requestAnimationFrame(() => {
            fitGraphToView(scope, index === 0 ? 260 : 520, padding);
          });
        }, delay),
      );
    },
    [fitGraphToView, layoutMode],
  );

  useEffect(() => {
    const view = forceRef.current;
    if (!view || layoutMode !== "force") return;

    const charge = view.d3Force("charge") as { strength?: (value: number) => void } | undefined;
    if (charge?.strength) charge.strength(-240);

    const link = view.d3Force("link") as {
      distance?: (fn: (edge: VizLink) => number) => void;
      strength?: (fn: (edge: VizLink) => number) => void;
    } | undefined;

    if (link?.distance) link.distance((edge) => linkDistance(edge.label));
    if (link?.strength) link.strength((edge) => linkStrength(edge.label));

    view.d3ReheatSimulation();
  }, [layoutMode, visibleGraph.links]);

  useEffect(() => {
    const view = forceRef.current;
    if (!view || !selectedNode) return;
    const target = nodeMap.get(selectedNode.id);
    if (target && typeof target.x === "number" && typeof target.y === "number") {
      view.centerAt(target.x, target.y, 240);
      view.zoom(Math.max(view.zoom(), 1.12), 240);
    }

    const frame = frameRef.current;
    if (frame) {
      const frameBounds = frame.getBoundingClientRect();
      const isVisible = frameBounds.top >= 0 && frameBounds.bottom <= window.innerHeight;
      if (!isVisible) {
        frame.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
      }
    }

    queueGraphReframe("visible", FOCUSED_GRAPH_FIT_PADDING);
  }, [nodeMap, queueGraphReframe, selectedNode]);

  useEffect(() => {
    if (!pendingBackgroundResetRef.current || selectedNode) return;
    pendingBackgroundResetRef.current = false;
    queueGraphReframe("all", ALL_GRAPH_FIT_PADDING);
  }, [queueGraphReframe, selectedNode, visibleGraph.links.length, visibleGraph.nodes.length]);

  function reframe() {
    if (selectedNode) {
      fitGraphToView("visible", 520, FOCUSED_GRAPH_FIT_PADDING);
      return;
    }

    if (activeNodeId) {
      fitGraphToView("active-neighborhood", 520, FOCUSED_GRAPH_FIT_PADDING);
      return;
    }

    fitGraphToView("all", 620, ALL_GRAPH_FIT_PADDING);
  }

  function handleFrameDoubleClick(event: ReactMouseEvent<HTMLDivElement>) {
    const isRecentNodeClick = performance.now() - lastNodeClickAtRef.current < 360;
    if (isRecentNodeClick || hoverNodeIdRef.current) return;

    event.preventDefault();
    setHoverNodeId(undefined);
    pendingBackgroundResetRef.current = true;
    onNodeSelect(undefined);

    if (!selectedNode) {
      pendingBackgroundResetRef.current = false;
      queueGraphReframe("all", ALL_GRAPH_FIT_PADDING);
    }
  }

  return (
    <section className="graph-shell" aria-label="Graph visualization">
      <div className="graph-overlay">
        <div className="graph-pill">
          <Network size={16} />
          <strong>{visibleGraph.nodes.length}</strong>
          <span>visible nodes</span>
        </div>
        <div className="graph-pill">
          <strong>{visibleGraph.links.length}</strong>
          <span>visible links</span>
        </div>
        {visibleGraph.hiddenCount > 0 && (
          <div className="graph-pill warn">
            <strong>{visibleGraph.hiddenCount}</strong>
            <span>hidden for clarity</span>
          </div>
        )}
      </div>

      <div className="graph-legend" aria-hidden="true">
        <span className="legend-item">
          <span className="legend-dot startup" /> startup
        </span>
        <span className="legend-item">
          <span className="legend-dot investor" /> investor
        </span>
        <span className="legend-item">
          <span className="legend-dot topic" /> topic
        </span>
        <span className="legend-item">
          <span className="legend-dot article" /> article
        </span>
      </div>

      <div className="graph-controls">
        <div className="graph-segmented">
          <button className={layoutMode === "force" ? "active" : ""} onClick={() => setLayoutMode("force")}>
            <Network size={13} />
            <span>Force</span>
          </button>
          <button className={layoutMode === "flow" ? "active" : ""} onClick={() => setLayoutMode("flow")}>
            <GitFork size={13} />
            <span>Flow</span>
          </button>
        </div>

        <div className="graph-segmented">
          {[1, 2, 3].map((depth) => (
            <button
              key={depth}
              className={focusDepth === depth ? "active" : ""}
              onClick={() => setFocusDepth(depth as 1 | 2 | 3)}
            >
              {depth} hop
            </button>
          ))}
        </div>

        <label className="graph-checkbox">
          <input type="checkbox" checked={declutter} onChange={(event) => setDeclutter(event.target.checked)} />
          <span>declutter</span>
        </label>

        <button className="graph-pill graph-pill-button" onClick={reframe}>
          Reframe graph
        </button>
      </div>

      {selectedNode && (
        <div className="graph-focus-note">
          <Search size={14} />
          <span>
            Focusing {compactLabel(selectedNode.label)} with {focusDepth}-hop neighborhood.
          </span>
        </div>
      )}

      <div ref={frameRef} className="graph-frame" onDoubleClick={handleFrameDoubleClick}>
        {viewport.width > 0 && viewport.height > 0 && (
          <ForceGraph2D<VizNode, VizLink>
            ref={forceRef}
            width={viewport.width}
            height={viewport.height}
            graphData={graphData}
            nodeId="id"
            linkSource="source"
            linkTarget="target"
            backgroundColor="rgba(0,0,0,0)"
            minZoom={0.18}
            maxZoom={3.4}
            cooldownTicks={layoutMode === "flow" ? 0 : 220}
            d3VelocityDecay={layoutMode === "flow" ? 0.9 : 0.34}
            autoPauseRedraw={false}
            enableNodeDrag={false}
            linkColor={(rawLink) => {
              const link = rawLink as VizLink;
              if (!activeNodeId) return link.color;
              const sourceId = endpointToId(rawLink.source);
              const targetId = endpointToId(rawLink.target);
              return activeNeighborhood.has(sourceId) && activeNeighborhood.has(targetId)
                ? link.color
                : "rgba(140, 155, 173, 0.14)";
            }}
            linkWidth={(rawLink) => {
              const link = rawLink as VizLink;
              if (!activeNodeId) return 1.2 * link.emphasis;
              const sourceId = endpointToId(rawLink.source);
              const targetId = endpointToId(rawLink.target);
              return activeNeighborhood.has(sourceId) && activeNeighborhood.has(targetId)
                ? 1.5 * link.emphasis
                : 0.52;
            }}
            linkDirectionalArrowLength={(rawLink) =>
              FINANCE_RELATIONS.has((rawLink as VizLink).label) ? 4.8 : 0
            }
            linkDirectionalArrowRelPos={1}
            linkDirectionalParticles={(rawLink) =>
              FINANCE_RELATIONS.has((rawLink as VizLink).label) ? 1 : 0
            }
            linkDirectionalParticleWidth={(rawLink) =>
              FINANCE_RELATIONS.has((rawLink as VizLink).label) ? 2.2 : 0
            }
            linkDirectionalParticleSpeed={(rawLink) =>
              FINANCE_RELATIONS.has((rawLink as VizLink).label) ? 0.0034 : 0
            }
            linkDirectionalParticleColor={(rawLink) =>
              FINANCE_RELATIONS.has((rawLink as VizLink).label) ? "#be123c" : "#8da2b7"
            }
            linkLabel={(rawLink) => {
              const link = rawLink as VizLink;
              const sourceId = endpointToId(rawLink.source);
              const targetId = endpointToId(rawLink.target);
              const sourceLabel = labelsById.get(sourceId) ?? sourceId;
              const targetLabel = labelsById.get(targetId) ?? targetId;
              const status = stringValue(link.properties.evidence_status);
              const statusText = status ? ` | ${status}` : "";
              const reviewText =
                link.marker === "review"
                  ? " | Review required"
                  : link.marker === "changed"
                  ? " | Support changed"
                  : "";
              return `${sourceLabel} -> ${targetLabel} | ${link.label}${statusText}${reviewText}`;
            }}
            linkCanvasObject={(rawLink, context, globalScale) => {
              const link = rawLink as VizLink;
              const source = resolveEndpointCoordinates(rawLink.source, nodeMap);
              const target = resolveEndpointCoordinates(rawLink.target, nodeMap);
              if (!source || !target) return;

              const sourceId = source.id;
              const targetId = target.id;
              const isFocusLink =
                activeNodeId !== undefined &&
                activeNeighborhood.has(sourceId) &&
                activeNeighborhood.has(targetId);

              if (link.marker && (!activeNodeId || isFocusLink)) {
                context.save();
                context.beginPath();
                context.moveTo(source.x, source.y);
                context.lineTo(target.x, target.y);
                context.strokeStyle = "#d97706";
                context.lineWidth = (link.marker === "review" ? 2.4 : 1.8) / globalScale;
                context.setLineDash([6 / globalScale, 4 / globalScale]);
                context.stroke();
                context.restore();
              }

              const zoomThreshold =
                  graphDensity >= 2.6 ? 1.34 : graphDensity >= 1.9 ? 1.22 : graphDensity >= 1.2 ? 1.08 : 0.92;

              if (!isFocusLink && (activeNodeId !== undefined || globalScale < zoomThreshold)) return;

              const dx = target.x - source.x;
              const dy = target.y - source.y;
              const distance = Math.hypot(dx, dy);
              if (distance < 34) return;

                const renderedDistance = distance * globalScale;

              const label = formatRelationLabel(link.label);
                const rawScreenFont =
                  8.4 + Math.log2(Math.max(globalScale, 1) + 1) * 1.45 - Math.min(1.6, graphDensity * 0.3);
                const screenFontSize = clamp(rawScreenFont, 8.1, isFocusLink ? 12.8 : 11.4);
                const fontSize = screenFontSize / globalScale;

              context.save();
                context.font = `${isFocusLink ? 700 : 600} ${fontSize}px "IBM Plex Sans", "Segoe UI", sans-serif`;
                const maxLabelWidthScreen = clamp(
                  renderedDistance * (isFocusLink ? 0.58 : 0.48),
                  52,
                  isFocusLink ? 182 : 148,
                );
                const maxLabelWidth = maxLabelWidthScreen / globalScale;
              const fittedLabel = fitTextToWidth(context, label, maxLabelWidth);
              if (!fittedLabel) {
                context.restore();
                return;
              }

              const labelWidth = context.measureText(fittedLabel).width;
                const badgeHeight = (screenFontSize + 4) / globalScale;
                const badgePadX = 5 / globalScale;
              const midX = source.x + dx * 0.5;
              const midY = source.y + dy * 0.5;
              let angle = Math.atan2(dy, dx);
              if (angle > Math.PI * 0.5 || angle < -Math.PI * 0.5) {
                angle += Math.PI;
              }

              context.translate(midX, midY);
              context.rotate(angle);

              context.fillStyle = isFocusLink ? "rgba(255, 255, 255, 0.95)" : "rgba(255, 255, 255, 0.82)";
              drawRoundedRect(
                context,
                  -labelWidth * 0.5 - badgePadX,
                -badgeHeight * 0.5,
                  labelWidth + badgePadX * 2,
                badgeHeight,
                  5 / globalScale,
              );
              context.fill();

              context.fillStyle = isFocusLink ? "#0f172a" : "#334155";
              context.textAlign = "center";
              context.textBaseline = "middle";
              context.fillText(fittedLabel, 0, 0);
              context.restore();
            }}
            linkCanvasObjectMode={() => "after"}
            nodeLabel={(rawNode) => {
              const node = rawNode as VizNode;
              return `${node.label} | ${node.type} | ${node.degree} connections`;
            }}
            nodePointerAreaPaint={(rawNode, color, context, globalScale) => {
              const node = rawNode as VizNode;
              const x = node.x ?? 0;
              const y = node.y ?? 0;
              const radius = node.radius ?? 8;
              const padding = 5 / globalScale;

              context.fillStyle = color;
              if (node.type === "Article") {
                drawRoundedRect(
                  context,
                  x - radius - 6 - padding,
                  y - radius * 0.6 - padding,
                  radius * 2 + 12 + padding * 2,
                  radius * 1.2 + padding * 2,
                  8,
                );
              } else {
                context.beginPath();
                context.arc(x, y, radius + padding, 0, Math.PI * 2);
              }
              context.fill();
            }}
            nodeCanvasObject={(rawNode, context, globalScale) => {
              const node = rawNode as NodeObject<VizNode>;
              const x = node.x ?? 0;
              const y = node.y ?? 0;
              const radius = node.radius ?? 8;

              const selected = selectedNode?.id === node.id;
              const hovered = hoverNodeId === node.id;
              const nearFocus = activeNodeId ? activeNeighborhood.has(String(node.id)) : true;
              const alpha = activeNodeId && !nearFocus ? 0.14 : 1;

              context.save();
              context.globalAlpha = alpha;

              if (selected || hovered) {
                context.beginPath();
                context.arc(x, y, radius + (selected ? 8 : 5), 0, Math.PI * 2);
                context.fillStyle = selected ? "rgba(249, 115, 22, 0.26)" : "rgba(15, 118, 110, 0.18)";
                context.fill();
              }

              context.fillStyle = node.color ?? "#64748b";
              context.strokeStyle = selected ? "#f97316" : "rgba(255, 255, 255, 0.95)";
              context.lineWidth = selected ? 3 : 1.65;

              if (node.type === "Article") {
                drawRoundedRect(context, x - radius - 6, y - radius * 0.6, radius * 2 + 12, radius * 1.2, 8);
              } else {
                context.beginPath();
                context.arc(x, y, radius, 0, Math.PI * 2);
              }

              context.fill();
              context.stroke();

              const showLabel =
                selected ||
                hovered ||
                radius * globalScale > 14 ||
                (!activeNodeId && (node.degree ?? 0) >= 7 && radius * globalScale > 11.8) ||
                (activeNodeId && nearFocus && radius * globalScale > 10.5);

              if (showLabel) {
                const renderedRadius = radius * globalScale;
                const rawScreenFont = renderedRadius * (selected ? 0.34 : 0.31);
                const screenFontSize = clamp(rawScreenFont, 8.2, selected ? 13.4 : 11.8);
                const fontSize = screenFontSize / globalScale;

                context.font = `${selected ? 700 : 600} ${fontSize}px "IBM Plex Sans", "Segoe UI", sans-serif`;
                const maxTextWidth = nodeLabelMaxWidth(node.type, renderedRadius) / globalScale;
                const fittedLabel = fitTextToWidth(
                  context,
                  String(node.label ?? node.id),
                  maxTextWidth,
                );

                if (fittedLabel) {
                  context.lineWidth = Math.max(0.9 / globalScale, (screenFontSize * 0.15) / globalScale);
                  context.strokeStyle = "rgba(15, 23, 42, 0.42)";
                  context.fillStyle = "#f8fafc";
                  context.textAlign = "center";
                  context.textBaseline = "middle";
                  context.strokeText(fittedLabel, x, y);
                  context.fillText(fittedLabel, x, y);
                }

                context.textAlign = "center";
                context.textBaseline = "middle";
              }

              context.restore();
            }}
            onNodeHover={(node) => {
              const nextNodeId = node ? String((node as VizNode).id) : undefined;
              hoverNodeIdRef.current = nextNodeId;
              setHoverNodeId((currentNodeId) => (currentNodeId === nextNodeId ? currentNodeId : nextNodeId));
            }}
            onNodeClick={(node) => {
              lastNodeClickAtRef.current = performance.now();
              onNodeSelect(nodeMap.get(String((node as VizNode).id)));
            }}
            onBackgroundClick={() => {
              hoverNodeIdRef.current = undefined;
              setHoverNodeId(undefined);
              onNodeSelect(undefined);
            }}
            showPointerCursor={(obj) => Boolean(obj)}
            onEngineStop={() => {
              if (autoFitRef.current || visibleGraph.nodes.length === 0) return;
              fitGraphToView(
                selectedNode ? "visible" : "all",
                620,
                selectedNode ? FOCUSED_GRAPH_FIT_PADDING : ALL_GRAPH_FIT_PADDING,
              );
            }}
          />
        )}

        {visibleGraph.nodes.length === 0 && (
          <div className="graph-empty">
            <p>No nodes match your current filters.</p>
          </div>
        )}
      </div>
    </section>
  );
}

function highestDegreeNodeId(
  nodes: GraphNode[],
  degreeById: Map<string, number>,
): string | undefined {
  if (nodes.length === 0) return undefined;
  return [...nodes].sort((left, right) => {
    const leftDegree = degreeById.get(left.id) ?? 0;
    const rightDegree = degreeById.get(right.id) ?? 0;
    return rightDegree - leftDegree;
  })[0]?.id;
}

function bfsNodeIds(edges: GraphEdge[], startId: string, maxDepth: number): Set<string> {
  const adjacency = new Map<string, Set<string>>();

  for (const edge of edges) {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, new Set<string>());
    if (!adjacency.has(edge.target)) adjacency.set(edge.target, new Set<string>());
    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);
  }

  const visited = new Set<string>([startId]);
  const queue: Array<{ id: string; depth: number }> = [{ id: startId, depth: 0 }];

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) continue;
    if (current.depth >= maxDepth) continue;

    for (const neighbor of adjacency.get(current.id) ?? []) {
      if (visited.has(neighbor)) continue;
      visited.add(neighbor);
      queue.push({ id: neighbor, depth: current.depth + 1 });
    }
  }

  return visited;
}

function computeFlowPositions(
  nodes: GraphNode[],
  width: number,
  height: number,
  degreeById: Map<string, number>,
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const columns = new Map<string, GraphNode[]>();
  const order = [...FLOW_ORDER];

  for (const node of nodes) {
    if (!columns.has(node.type)) columns.set(node.type, []);
    columns.get(node.type)?.push(node);
    if (!order.includes(node.type)) order.push(node.type);
  }

  const activeColumns = order.filter((type) => (columns.get(type)?.length ?? 0) > 0);
  if (activeColumns.length === 0) return positions;

  const marginX = Math.max(70, width * 0.08);
  const marginY = Math.max(52, height * 0.1);
  const usableWidth = Math.max(100, width - marginX * 2);
  const usableHeight = Math.max(100, height - marginY * 2);

  activeColumns.forEach((type, columnIndex) => {
    const columnNodes = columns.get(type) ?? [];
    const x = marginX + (activeColumns.length === 1 ? usableWidth / 2 : (usableWidth * columnIndex) / (activeColumns.length - 1));
    const sorted = [...columnNodes].sort((left, right) => {
      const leftScore = importanceScore(left, degreeById.get(left.id) ?? 0);
      const rightScore = importanceScore(right, degreeById.get(right.id) ?? 0);
      return rightScore - leftScore || left.label.localeCompare(right.label);
    });

    sorted.forEach((node, index) => {
      const y = marginY + (usableHeight * (index + 1)) / (sorted.length + 1);
      positions.set(node.id, { x, y });
    });
  });

  return positions;
}

function importanceScore(node: GraphNode, degree: number): number {
  return degree * 2 + (TYPE_PRIORITY[node.type] ?? 18);
}

function baseRadius(type: string, degree: number): number {
  const size =
    type === "Startup"
      ? 9.2
      : type === "Investor"
      ? 8.5
      : type === "Company"
      ? 8.2
      : type === "Article"
      ? 7
      : 6.8;

  return size + Math.min(8, degree * 0.62);
}

function linkDistance(label: string): number {
  if (label === "INVESTED_IN") return 128;
  if (label === "HAS_TOPIC") return 108;
  if (label === "FROM_SOURCE") return 96;
  if (label === "MENTIONS") return 104;
  return 116;
}

function linkStrength(label: string): number {
  if (label === "INVESTED_IN") return 0.22;
  if (label === "HAS_TOPIC") return 0.1;
  if (label === "MENTIONS") return 0.12;
  return 0.14;
}

function linkMarker(edge: GraphEdge): "review" | "changed" | undefined {
  if (stringValue(edge.properties.review_status) === "needs_review") return "review";
  return edge.properties.support_changed === true ? "changed" : undefined;
}

function endpointToId(value: unknown): string {
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (value && typeof value === "object" && "id" in value) {
    const objectValue = value as { id?: string | number };
    if (objectValue.id !== undefined) return String(objectValue.id);
  }
  return "";
}

function resolveEndpointCoordinates(
  value: unknown,
  nodesById: Map<string, VizNode>,
): { id: string; x: number; y: number } | undefined {
  const endpointId = endpointToId(value);

  if (value && typeof value === "object") {
    const endpoint = value as { id?: string | number; x?: number; y?: number };
    if (typeof endpoint.x === "number" && typeof endpoint.y === "number") {
      return {
        id: endpointId || String(endpoint.id ?? ""),
        x: endpoint.x,
        y: endpoint.y,
      };
    }
  }

  if (!endpointId) return undefined;
  const node = nodesById.get(endpointId);
  if (!node || typeof node.x !== "number" || typeof node.y !== "number") return undefined;
  return { id: endpointId, x: node.x, y: node.y };
}

function formatRelationLabel(label: string): string {
  return label
    .split("_")
    .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
    .join(" ");
}

function fitTextToWidth(
  context: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
): string {
  const normalized = text.trim();
  if (!normalized || maxWidth <= 8) return "";
  if (context.measureText(normalized).width <= maxWidth) return normalized;

  let shortened = normalized;
  while (shortened.length > 1 && context.measureText(`${shortened}...`).width > maxWidth) {
    shortened = shortened.slice(0, -1);
  }
  return shortened.length > 1 ? `${shortened}...` : "";
}

function nodeLabelMaxWidth(type: string, renderedRadius: number): number {
  if (type === "Article") return clamp(renderedRadius * 2.1 + 10, 54, 122);
  return clamp(renderedRadius * 1.72, 48, 108);
}

function hasFinitePosition(node: { x?: number; y?: number }): boolean {
  return Number.isFinite(node.x) && Number.isFinite(node.y);
}

function compactLabel(label: string): string {
  if (label.length <= 38) return label;
  return `${label.slice(0, 35)}...`;
}

function drawRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const r = Math.max(0, Math.min(radius, Math.min(width, height) / 2));
  context.beginPath();
  context.moveTo(x + r, y);
  context.lineTo(x + width - r, y);
  context.quadraticCurveTo(x + width, y, x + width, y + r);
  context.lineTo(x + width, y + height - r);
  context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  context.lineTo(x + r, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - r);
  context.lineTo(x, y + r);
  context.quadraticCurveTo(x, y, x + r, y);
  context.closePath();
}
