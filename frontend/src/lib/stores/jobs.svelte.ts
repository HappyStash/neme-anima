import type { JobStages, PipelineStage, ServerEvent } from "$lib/types";

const EXTRACT_STAGE_KEYS: Array<[string, string]> = [
  ["setup", "Setup"],
  ["scenes", "Scene detection"],
  ["detect", "Person detection"],
  ["track", "Tracking"],
  ["identify", "Identify · select · save"],
  ["tag", "Tagging"],
];

const RERUN_STAGE_KEYS: Array<[string, string]> = [
  ["setup", "Setup"],
  ["identify", "Identify · select · save"],
  ["tag", "Tagging"],
];

function pendingStages(kind: "extract" | "rerun"): PipelineStage[] {
  const defs = kind === "rerun" ? RERUN_STAGE_KEYS : EXTRACT_STAGE_KEYS;
  return defs.map(([key, label]) => ({
    key, label, status: "pending", current: 0, total: 0, pct: 0, message: "",
  }));
}

class JobsStore {
  byJobId = $state<Record<string, JobStages>>({});

  /** Seed an entry the moment Run is clicked, before any server event arrives. */
  seedPending(args: {
    job_id: string;
    project: string;
    source_idx: number;
    kind: "extract" | "rerun";
  }) {
    if (this.byJobId[args.job_id]) return;
    this.byJobId[args.job_id] = {
      job_id: args.job_id,
      project: args.project,
      source_idx: args.source_idx,
      kind: args.kind,
      stages: pendingStages(args.kind),
      summary: null,
      updated_at: Date.now(),
    };
  }

  ingest(event: ServerEvent) {
    if (event.type === "job.stages") {
      const p = event.payload as Omit<JobStages, "updated_at">;
      this.byJobId[p.job_id] = { ...p, updated_at: Date.now() };
    } else if (event.type === "job.done") {
      // Mark any still-running stage as done so the UI snaps to a green
      // terminal state if the runner returned before the final progress
      // event was published.
      const jobId = event.payload.job_id as string | undefined;
      if (!jobId) return;
      const existing = this.byJobId[jobId];
      if (!existing) return;
      const allDone = existing.stages.every((s) => s.status === "done" || s.status === "failed");
      if (allDone) return;
      this.byJobId[jobId] = {
        ...existing,
        stages: existing.stages.map((s) =>
          s.status === "running" ? { ...s, status: "done", pct: 1.0 } : s,
        ),
        updated_at: Date.now(),
      };
    }
  }

  /** Find the most recent job for a given (project slug, source index). */
  forSource(slug: string, sourceIdx: number): JobStages | null {
    let best: JobStages | null = null;
    for (const job of Object.values(this.byJobId)) {
      if (job.project !== slug || job.source_idx !== sourceIdx) continue;
      if (!best || job.updated_at > best.updated_at) best = job;
    }
    return best;
  }
}

export const jobsStore = new JobsStore();
