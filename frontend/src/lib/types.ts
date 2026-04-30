// Mirrors the Pydantic models in src/neme_extractor/server/api/*.py.

export interface Source {
  path: string;
  added_at: string;
  excluded_refs: string[];
  extraction_runs: unknown[];
}

export interface RefImage {
  path: string;
  added_at: string;
}

export interface ProjectView {
  slug: string;
  name: string;
  folder: string;
  created_at: string;
  sources: Source[];
  refs: RefImage[];
  thresholds_overrides: Record<string, Record<string, unknown>>;
  source_root: string | null;
}

export interface ProjectListEntry {
  slug: string;
  name: string;
  folder: string;
  missing: boolean;
  source_count: number;
  ref_count: number;
  last_opened_at: string;
}

export interface FrameRecord {
  filename: string;
  kept: boolean;
  video_stem: string;
  scene_idx: number;
  tracklet_id: number;
  frame_idx: number;
  timestamp_seconds: number;
  ccip_distance: number;
  score: number;
}

export interface FramesPage {
  count: number;
  items: FrameRecord[];
}

export interface QueueItem {
  job_id: string;
  status: "pending" | "running" | "done" | "cancelled" | "failed";
  payload: Record<string, unknown>;
  error: string | null;
}

export type EventType =
  | "queue.update"
  | "job.progress"
  | "job.frame"
  | "job.log"
  | "job.done";

export interface ServerEvent {
  type: EventType;
  payload: Record<string, unknown>;
}
