import { Activity, Database, Loader2, Play, Trash2 } from "lucide-react";
import { useState } from "react";
import { clamp } from "../lib/helpers";
import type { TaskStatus } from "../types/graph";

type Props = {
  maxPages: number;
  onMaxPagesChange: (value: number) => void;
  onRun: () => void;
  onClear: () => Promise<{ deleted_nodes: number }>;
  task?: TaskStatus;
};

export function IngestControl({ maxPages, onMaxPagesChange, onRun, onClear, task }: Props) {
  const running = task?.status === "queued" || task?.status === "running";
  const processed = task?.result?.articles_processed as number | undefined;
  const found = task?.result?.articles_found as number | undefined;
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
      ? `${processed ?? 0} processed from ${found ?? 0} discovered`
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
        <button className="command-button" onClick={onRun} disabled={running}>
          {running ? <Loader2 size={17} className="spin" /> : <Play size={16} />}
          <span>{running ? "Running" : "Start"}</span>
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
          disabled={clearState === "clearing" || running}
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


