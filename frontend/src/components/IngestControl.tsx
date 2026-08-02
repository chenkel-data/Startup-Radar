import {
  Activity,
  Database,
  Loader2,
  Play,
  RefreshCcw,
  Sparkles,
  Trash2,
} from "lucide-react";
import { useState } from "react";
import { clamp } from "../lib/helpers";
import type { CurationPending, TaskStatus } from "../types/graph";

type Props = {
  maxPages: number;
  onMaxPagesChange: (value: number) => void;
  onRun: () => void;
  onForceRun: () => void;
  onClear: () => Promise<{ deleted_nodes: number }>;
  task?: TaskStatus;
  curation?: CurationPending;
  curationTask?: TaskStatus;
  onCurate: () => void;
};

export function IngestControl({
  maxPages,
  onMaxPagesChange,
  onRun,
  onForceRun,
  onClear,
  task,
  curation,
  curationTask,
  onCurate,
}: Props) {
  const ingestRunning = task?.status === "queued" || task?.status === "running";
  const curationRunning =
    curationTask?.status === "queued" || curationTask?.status === "running";
  const workflowRunning = ingestRunning || curationRunning;
  const processed = task?.result?.articles_processed as number | undefined;
  const found = task?.result?.articles_found as number | undefined;
  const cached = task?.result?.articles_cached as number | undefined;
  const scraped = task?.result?.articles_scraped as number | undefined;
  const skipped = task?.result?.articles_skipped as number | undefined;
  const cacheMessage =
    typeof task?.result?.cache_message === "string" ? task.result.cache_message : undefined;
  const showCachePrompt =
    task?.status === "succeeded" && Boolean(cacheMessage) && (skipped ?? 0) > 0;
  const safePages = clamp(maxPages, 1, 50);

  const [clearState, setClearState] = useState<"idle" | "confirm" | "clearing">("idle");
  const [lastCleared, setLastCleared] = useState<number | null>(null);

  function updatePages(value: number) {
    onMaxPagesChange(clamp(Math.round(value), 1, 50));
  }

  async function handleClear() {
    if (clearState === "idle") {
      setClearState("confirm");
      return;
    }
    setClearState("clearing");
    try {
      const result = await onClear();
      setLastCleared(result.deleted_nodes);
    } catch {
      // error display is handled by the parent via onClear
    } finally {
      setClearState("idle");
    }
  }

  const statusLabel = task?.status ?? "idle";
  const statusCopy =
    statusLabel === "succeeded"
      ? `${processed ?? 0} processed, ${cached ?? 0} cached, `
        + `${scraped ?? 0} scraped from ${found ?? 0} discovered`
      : statusLabel === "failed"
      ? task?.error ?? "Pipeline failed"
      : statusLabel === "running" || statusLabel === "queued"
      ? "Pipeline is collecting and extracting entities"
      : "Run ingestion to refresh graph evidence";

  return (
    <section className="panel ingest-panel" aria-label="Ingestion">
      <div className="panel-head">
        <div className="panel-title">
          <Database size={17} />
          <h2>Ingestion Pipeline</h2>
        </div>
        <button className="command-button" onClick={onRun} disabled={workflowRunning}>
          {ingestRunning ? <Loader2 size={17} className="spin" /> : <Play size={16} />}
          <span>{ingestRunning ? "Running" : "Start"}</span>
        </button>
      </div>

      <div className="slider-row">
        <label className="range-meta" htmlFor="max-pages-slider">
          <span>Max pages per run</span>
          <strong>{safePages}</strong>
        </label>
        <input
          id="max-pages-slider"
          className="range-slider"
          type="range"
          min={1}
          max={50}
          value={safePages}
          onChange={(event) => updatePages(Number(event.target.value))}
        />
      </div>

      <div className="ingest-row">
        <label className="ingest-number">
          <span>Precise value</span>
          <input
            type="number"
            min={1}
            max={50}
            value={safePages}
            onChange={(event) => updatePages(Number(event.target.value))}
          />
        </label>
      </div>

      <div className="task-status-line">
        <span className={`status-pill ${statusLabel}`}>{statusLabel}</span>
        <span className="task-copy">{statusCopy}</span>
      </div>

      {showCachePrompt && (
        <div className="cache-prompt">
          <span>{cacheMessage} Re-scrape anyway?</span>
          <button className="ghost-mini" onClick={onForceRun} disabled={workflowRunning}>
            <RefreshCcw size={14} />
            <span>Re-scrape</span>
          </button>
        </div>
      )}

      {task?.task_id && (
        <div className="task-status-line subtle">
          <Activity size={14} />
          <span className="task-copy">Task {task.task_id.slice(0, 8)}</span>
        </div>
      )}

      {task?.status === "failed" && task.error && (
        <div className="task-status-line subtle">
          <span className="error-text">{task.error}</span>
        </div>
      )}

      <div className="curation-row">
        <div>
          <strong>Entity descriptions</strong>
          <span className="task-copy">{curationStatusCopy(curation)}</span>
        </div>
        <button
          className="ghost-mini"
          onClick={onCurate}
          disabled={
            workflowRunning
            || !curation?.enabled
            || curation.ready_entities < 1
          }
        >
          {curationRunning
            ? <Loader2 size={14} className="spin" />
            : <Sparkles size={14} />}
          <span>{curationRunning ? "Curating" : "Curate descriptions"}</span>
        </button>
      </div>

      {curationTask?.status === "succeeded" && (
        <div className="task-status-line subtle">
          <Activity size={14} />
          <span className="task-copy">{curationResultCopy(curationTask)}</span>
        </div>
      )}

      {curationTask?.status === "failed" && curationTask.error && (
        <div className="task-status-line subtle">
          <span className="error-text">{curationTask.error}</span>
        </div>
      )}

      <div className="clear-graph-row">
        {clearState === "confirm" && (
          <span className="task-copy" style={{ color: "#b91c1c" }}>
            This deletes all nodes and relationships.
          </span>
        )}
        {lastCleared !== null && clearState === "idle" && (
          <span className="task-copy">{lastCleared} nodes deleted</span>
        )}
        <button
          className={`command-button danger${clearState === "confirm" ? " confirm" : ""}`}
          onClick={handleClear}
          disabled={clearState === "clearing" || workflowRunning}
          onBlur={() => { if (clearState === "confirm") setClearState("idle"); }}
        >
          {clearState === "clearing"
            ? <Loader2 size={15} className="spin" />
            : <Trash2 size={15} />}
          <span>
            {clearState === "confirm" ? "Confirm clear" : "Clear graph"}
          </span>
        </button>
      </div>
    </section>
  );
}

function curationStatusCopy(curation?: CurationPending): string {
  if (!curation) return "Checking descriptions…";
  if (!curation.enabled) return "Description updates are disabled.";
  if (curation.ready_entities > 0) {
    return `${curation.ready_entities} ready to update; `
      + `${curation.waiting_entities} waiting for more evidence.`;
  }
  if (curation.waiting_entities > 0) {
    return `${curation.waiting_entities} waiting for more evidence.`;
  }
  return "No descriptions need updating.";
}

function curationResultCopy(task: TaskStatus): string {
  const result = task.result;
  const candidates = numberResult(result, "candidate_entities");
  const updated = numberResult(result, "updated");
  const kept = numberResult(result, "kept");
  const cost = numberResult(result, "cost_usd");
  return `${candidates} curated: ${updated} updated, ${kept} kept; $${cost.toFixed(4)} LLM cost.`;
}

function numberResult(result: Record<string, unknown> | undefined, key: string): number {
  const value = result?.[key];
  return typeof value === "number" ? value : 0;
}
