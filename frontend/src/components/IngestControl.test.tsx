import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { IngestControl } from "./IngestControl";
import type { TaskStatus } from "../types/graph";

describe("IngestControl", () => {
  it("offers a force re-scrape action when every discovered article is cached", async () => {
    const user = userEvent.setup();
    const onForceRun = vi.fn();
    const task: TaskStatus = {
      task_id: "task-123",
      name: "ingest",
      status: "succeeded",
      created_at: "2026-07-04T10:00:00Z",
      completed_at: "2026-07-04T10:00:05Z",
      result: {
        articles_found: 2,
        articles_cached: 2,
        articles_scraped: 0,
        articles_skipped: 2,
        articles_processed: 0,
        cache_message:
          "All 2 discovered articles are already cached for the current extraction prompts/settings.",
      },
    };

    render(
      <IngestControl
        maxPages={2}
        onMaxPagesChange={vi.fn()}
        onRun={vi.fn()}
        onForceRun={onForceRun}
        onClear={vi.fn().mockResolvedValue({ deleted_nodes: 0 })}
        onCurate={vi.fn()}
        task={task}
      />,
    );

    await user.click(screen.getByRole("button", { name: /re-scrape/i }));

    expect(onForceRun).toHaveBeenCalledTimes(1);
  });

  it("enables manual curation only when three distinct evidence texts are ready", async () => {
    const user = userEvent.setup();
    const onCurate = vi.fn();
    const props = {
      maxPages: 2,
      onMaxPagesChange: vi.fn(),
      onRun: vi.fn(),
      onForceRun: vi.fn(),
      onClear: vi.fn().mockResolvedValue({ deleted_nodes: 0 }),
      onCurate,
    };
    const { rerender } = render(
      <IngestControl
        {...props}
        curation={{
          enabled: true,
          threshold: 3,
          ready_entities: 0,
          waiting_entities: 1,
          ready_evidence: 0,
          waiting_evidence: 2,
        }}
      />,
    );

    expect(screen.getByRole("button", { name: /curate descriptions/i })).toBeDisabled();

    rerender(
      <IngestControl
        {...props}
        curation={{
          enabled: true,
          threshold: 3,
          ready_entities: 1,
          waiting_entities: 0,
          ready_evidence: 3,
          waiting_evidence: 0,
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: /curate descriptions/i }));
    expect(onCurate).toHaveBeenCalledTimes(1);
  });
});
