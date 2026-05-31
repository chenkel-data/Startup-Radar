import { useEffect, useState } from "react";
import { Activity, GitFork, Loader2, TrendingUp, Users } from "lucide-react";
import { api } from "../lib/api";
import { numberValue, stringValue } from "../lib/helpers";
import type { InsightRow } from "../types/graph";

type InsightKind = "trending-startups" | "top-investors" | "co-investments" | "topic-clusters";

const TABS: Array<{ kind: InsightKind; label: string; icon: typeof TrendingUp }> = [
  { kind: "trending-startups", label: "Trending", icon: TrendingUp },
  { kind: "top-investors", label: "Investors", icon: Users },
  { kind: "co-investments", label: "Co-invest", icon: GitFork },
  { kind: "topic-clusters", label: "Topics", icon: Activity }
];

type Props = {
  refreshKey: number;
  onEntityOpen: (name: string) => void;
};

export function InsightsPanel({ refreshKey, onEntityOpen }: Props) {
  const [active, setActive] = useState<InsightKind>("trending-startups");
  const [rows, setRows] = useState<InsightRow[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .insights(active)
      .then((data) => {
        if (!cancelled) setRows(data);
      })
      .catch(() => {
        if (!cancelled) setRows([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [active, refreshKey]);

  return (
    <section className="panel insights-panel" aria-label="Insights">
      <div className="panel-head compact">
        <div className="panel-title">
          <Activity size={16} />
          <h2>Market Pulse</h2>
        </div>
        <span className="panel-meta">{rows.length} rows</span>
      </div>

      <div className="tabs">
        {TABS.map(({ kind, label, icon: Icon }) => (
          <button key={kind} className={active === kind ? "active" : ""} onClick={() => setActive(kind)}>
            <Icon size={15} />
            <span>{label}</span>
          </button>
        ))}
      </div>

      <div className="insight-list">
        {loading && (
          <div className="loading-row">
            <Loader2 size={18} className="spin" />
            <span>Loading insights</span>
          </div>
        )}
        {!loading &&
          rows.map((row, index) => {
            const title = resolveTitle(row, index);
            const subtitle = resolveSubtitle(row);
            const metric = resolveMetric(row);
            const primaryEntity = resolveEntity(row);
            return (
              <button
                key={`${title}-${index}`}
                className="insight-row"
                onClick={() => {
                  if (primaryEntity) onEntityOpen(primaryEntity);
                }}
                disabled={!primaryEntity}
              >
                <span className="rank">{index + 1}</span>
                <span className="insight-main">
                  <strong>{title}</strong>
                  <small>{subtitle}</small>
                </span>
                <span className="metric-block">
                  <strong>{metric.value}</strong>
                  <small>{metric.label}</small>
                </span>
              </button>
            );
          })}
        {!loading && rows.length === 0 && <p className="empty-text">Insights appear once ingestion has seeded data.</p>}
      </div>
    </section>
  );
}

function resolveTitle(row: InsightRow, index: number): string {
  const name = stringValue(row.name);
  if (name) return name;
  const source = stringValue(row.source);
  const target = stringValue(row.target);
  if (source && target) return `${source} + ${target}`;
  return source ?? target ?? `Cluster ${index + 1}`;
}

function resolveEntity(row: InsightRow): string | undefined {
  return stringValue(row.name) ?? stringValue(row.source);
}

function resolveSubtitle(row: InsightRow): string {
  const examples = asStringArray(row.examples) ?? asStringArray(row.articles);
  if (examples && examples.length > 0) {
    return examples.slice(0, 2).join(" | ");
  }
  const source = stringValue(row.source);
  const target = stringValue(row.target);
  if (source && target) {
    return `Co-investment signal between ${source} and ${target}`;
  }
  return "Open to focus graph on this signal";
}

function resolveMetric(row: InsightRow): { value: string; label: string } {
  const mentions = numberValue(row.mentions);
  if (mentions !== undefined) return { value: String(mentions), label: "mentions" };

  const investments = numberValue(row.investments);
  if (investments !== undefined) return { value: String(investments), label: "investments" };

  const sharedStartups = numberValue(row.shared_startups);
  if (sharedStartups !== undefined) return { value: String(sharedStartups), label: "shared startups" };

  const rounds = numberValue(row.rounds);
  if (rounds !== undefined) return { value: String(rounds), label: "rounds" };

  const entities = numberValue(row.entity_count);
  if (entities !== undefined) return { value: String(entities), label: "entities" };

  return { value: "-", label: "metric" };
}

function asStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0);
}
