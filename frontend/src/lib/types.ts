// Mirrors the Pydantic models in src/neme_extractor/server/api/*.py.

export interface Source {
  path: string;
  added_at: string;
  excluded_refs: string[];
  /** True when the project has at least one kept frame on disk for this
   *  video stem — survives server restarts. */
  extracted: boolean;
}

export interface RefImage {
  path: string;
  added_at: string;
}

export interface LLMConfig {
  enabled: boolean;
  endpoint: string;
  model: string;
  prompt: string;
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
  pause_before_tag: boolean;
  llm: LLMConfig;
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
  /** True when the .txt sidecar has a non-empty second line (an LLM
   *  description). Drives the at-a-glance "described" badge in the grid. */
  has_description: boolean;
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
  | "job.stages"
  | "job.frame"
  | "job.log"
  | "job.done";

export type StageStatus = "pending" | "running" | "done" | "failed";

export interface PipelineStage {
  key: string;
  label: string;
  status: StageStatus;
  current: number;
  total: number;
  pct: number;
  message: string;
}

export interface JobStages {
  job_id: string;
  project: string;
  source_idx: number | null;
  kind: "extract" | "rerun" | string;
  stages: PipelineStage[];
  summary: { kept?: number; rejected?: number } | null;
  updated_at: number;
  /** True when the runner is parked at wait_for_resume() and a click on the
   *  yellow pause indicator will release it. */
  paused?: boolean;
  pause_message?: string;
}

export interface ServerEvent {
  type: EventType;
  payload: Record<string, unknown>;
}
