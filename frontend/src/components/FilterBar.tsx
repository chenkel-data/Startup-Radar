import { Filter, RotateCcw, SlidersHorizontal } from "lucide-react";

const TYPES = ["Startup", "Investor", "Company", "Person", "Topic", "Article", "Source"];
const CORE_TYPES = ["Startup", "Investor", "Topic"];
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
const CORE_RELATION_TYPES = ["INVESTED_IN", "FOUNDED_BY", "MERGED_WITH", "ACQUIRED", "HAS_TOPIC"];

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

type Props = {
  selectedTypes: Set<string>;
  selectedRelations: Set<string>;
  typeCounts: Record<string, number>;
  relationCounts: Record<string, number>;
  onTypeChange: (types: Set<string>) => void;
  onRelationChange: (relations: Set<string>) => void;
};

export function FilterBar({
  selectedTypes,
  selectedRelations,
  typeCounts,
  relationCounts,
  onTypeChange,
  onRelationChange,
}: Props) {
  function toggleType(type: string) {
    const next = new Set(selectedTypes);
    if (next.has(type)) {
      next.delete(type);
    } else {
      next.add(type);
    }
    onTypeChange(next);
  }

  function toggleRelation(relation: string) {
    const next = new Set(selectedRelations);
    if (next.has(relation)) {
      next.delete(relation);
    } else {
      next.add(relation);
    }
    onRelationChange(next);
  }

  function selectAll() {
    onTypeChange(new Set(TYPES));
  }

  function selectCore() {
    onTypeChange(new Set(CORE_TYPES));
  }

  function selectAllRelations() {
    onRelationChange(new Set(RELATION_TYPES));
  }

  function selectCoreRelations() {
    onRelationChange(new Set(CORE_RELATION_TYPES));
  }

  return (
    <section className="panel filter-panel" aria-label="Graph filters">
      <div className="panel-head compact">
        <div className="panel-title">
          <Filter size={16} />
          <h2>Lens</h2>
        </div>
        <span className="panel-meta">
          {selectedTypes.size}/{TYPES.length} nodes | {selectedRelations.size}/{RELATION_TYPES.length} rels
        </span>
      </div>

      <div className="filter-toolbar">
        <button className="ghost-mini" onClick={selectAll}>
          <RotateCcw size={14} />
          <span>All</span>
        </button>
        <button className="ghost-mini" onClick={selectCore}>
          <SlidersHorizontal size={14} />
          <span>Core signal</span>
        </button>
      </div>

      <div className="filter-section-label">Nodes</div>
      <div className="filter-grid">
        {TYPES.map((type) => (
          <button
            key={type}
            className={`filter-chip ${selectedTypes.has(type) ? "active" : ""}`}
            onClick={() => toggleType(type)}
            aria-pressed={selectedTypes.has(type)}
          >
            <span className={`type-dot ${type.toLowerCase()}`} />
            <span className="filter-name">{type}</span>
            <span className="filter-count">{typeCounts[type] ?? 0}</span>
          </button>
        ))}
      </div>

      <div className="filter-toolbar relation-toolbar">
        <button className="ghost-mini" onClick={selectAllRelations}>
          <RotateCcw size={14} />
          <span>All relations</span>
        </button>
        <button className="ghost-mini" onClick={selectCoreRelations}>
          <SlidersHorizontal size={14} />
          <span>Core relations</span>
        </button>
      </div>

      <div className="filter-section-label">Relations</div>
      <div className="filter-grid relation-filter-grid">
        {RELATION_TYPES.map((relation) => (
          <button
            key={relation}
            className={`filter-chip relation-filter-chip ${selectedRelations.has(relation) ? "active" : ""}`}
            onClick={() => toggleRelation(relation)}
            aria-pressed={selectedRelations.has(relation)}
          >
            <span className="relation-dot" style={{ background: RELATION_COLORS[relation] ?? "#64748b" }} />
            <span className="filter-name">{relation}</span>
            <span className="filter-count">{relationCounts[relation] ?? 0}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
