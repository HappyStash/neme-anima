import type {
  FramesPage, ProjectListEntry, ProjectView, QueueItem,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(`API ${status}: ${JSON.stringify(detail)}`);
  }
}

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  const headers = { "Content-Type": "application/json", ...(init.headers ?? {}) };
  const resp = await fetch(url, { ...init, method: init.method ?? "GET", headers });
  if (!resp.ok) {
    let detail: unknown = null;
    try { detail = await resp.json(); } catch { /* body not JSON */ }
    throw new ApiError(resp.status, detail);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json() as Promise<T>;
}

// ---- projects ----

export const listProjects = () => request<ProjectListEntry[]>("/api/projects");

export const createProject = (body: { name: string; folder: string }) =>
  request<ProjectView>("/api/projects", { method: "POST", body: JSON.stringify(body) });

export const registerProject = (folder: string) =>
  request<ProjectView>("/api/projects/register", {
    method: "POST", body: JSON.stringify({ folder }),
  });

export const getProject = (slug: string) =>
  request<ProjectView>(`/api/projects/${encodeURIComponent(slug)}`);

export const patchProject = (
  slug: string,
  body: { name?: string; thresholds_overrides?: Record<string, Record<string, unknown>> },
) => request<ProjectView>(`/api/projects/${encodeURIComponent(slug)}`, {
  method: "PATCH", body: JSON.stringify(body),
});

export const deleteProject = (slug: string, deleteFiles: boolean) =>
  request<void>(`/api/projects/${encodeURIComponent(slug)}`, {
    method: "DELETE", body: JSON.stringify({ delete_files: deleteFiles }),
  });

// ---- sources ----

export const addSources = (slug: string, paths: string[]) =>
  request<{ added: string[]; skipped: string[] }>(
    `/api/projects/${encodeURIComponent(slug)}/sources`,
    { method: "POST", body: JSON.stringify({ paths }) },
  );

export const importSourcesFolder = (slug: string, folder: string) =>
  request<{ added: string[]; skipped: string[]; source_root: string }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/import-folder`,
    { method: "POST", body: JSON.stringify({ folder }) },
  );

export const reimportSources = (slug: string) =>
  request<{ added: string[]; skipped: string[]; source_root: string }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/reimport`,
    { method: "POST" },
  );

export const sourceThumbnailUrl = (slug: string, idx: number) =>
  `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/thumbnail`;

export const removeSource = (slug: string, idx: number) =>
  request<void>(`/api/projects/${encodeURIComponent(slug)}/sources/${idx}`, { method: "DELETE" });

export const setExcludedRefs = (slug: string, idx: number, excluded: string[]) =>
  request<{ excluded_refs: string[] }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}`,
    { method: "PATCH", body: JSON.stringify({ excluded_refs: excluded }) },
  );

export const extractSource = (slug: string, idx: number) =>
  request<{ job_id: string }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/extract`,
    { method: "POST" },
  );

export const rerunSource = (slug: string, idx: number) =>
  request<{ job_id: string }>(
    `/api/projects/${encodeURIComponent(slug)}/sources/${idx}/rerun`,
    { method: "POST" },
  );

// ---- refs ----

export const addRefs = (slug: string, paths: string[]) =>
  request<{ added: string[]; skipped: string[] }>(
    `/api/projects/${encodeURIComponent(slug)}/refs`,
    { method: "POST", body: JSON.stringify({ paths }) },
  );

export const refImageUrl = (slug: string, refPath: string): string => {
  const name = refPath.split("/").pop() ?? refPath;
  return `/api/projects/${encodeURIComponent(slug)}/refs/${encodeURIComponent(name)}/image`;
};

export const uploadRefs = async (slug: string, files: File[]) => {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  const resp = await fetch(
    `/api/projects/${encodeURIComponent(slug)}/refs/upload`,
    { method: "POST", body: fd },
  );
  if (!resp.ok) {
    let detail: unknown = null;
    try { detail = await resp.json(); } catch { /* body not JSON */ }
    throw new ApiError(resp.status, detail);
  }
  return resp.json() as Promise<{ added: string[]; skipped: string[] }>;
};

export const removeRef = (slug: string, path: string) =>
  request<void>(`/api/projects/${encodeURIComponent(slug)}/refs`, {
    method: "DELETE", body: JSON.stringify({ path }),
  });

// ---- frames ----

export const listFrames = (
  slug: string,
  opts: { source?: string; offset?: number; limit?: number } = {},
) => {
  const q = new URLSearchParams();
  if (opts.source) q.set("source", opts.source);
  if (opts.offset !== undefined) q.set("offset", String(opts.offset));
  if (opts.limit !== undefined) q.set("limit", String(opts.limit));
  const qs = q.toString();
  return request<FramesPage>(
    `/api/projects/${encodeURIComponent(slug)}/frames${qs ? `?${qs}` : ""}`,
  );
};

export const getTags = (slug: string, filename: string) =>
  request<{ text: string }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/tags`,
  );

export const putTags = (slug: string, filename: string, text: string) =>
  request<{ text: string }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/tags`,
    { method: "PUT", body: JSON.stringify({ text }) },
  );

export const deleteFrame = (slug: string, filename: string) =>
  request<void>(
    `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
  );

export const bulkDeleteFrames = (slug: string, filenames: string[]) =>
  request<{ deleted: number }>(
    `/api/projects/${encodeURIComponent(slug)}/frames/bulk-delete`,
    { method: "POST", body: JSON.stringify({ filenames }) },
  );

export const bulkTagsReplace = (
  slug: string,
  body: { filenames: string[]; pattern: string; replacement: string; case_insensitive?: boolean },
) => request<{ changed: number }>(
  `/api/projects/${encodeURIComponent(slug)}/frames/bulk-tags-replace`,
  { method: "POST", body: JSON.stringify(body) },
);

export const frameImageUrl = (slug: string, filename: string) =>
  `/api/projects/${encodeURIComponent(slug)}/frames/${encodeURIComponent(filename)}/image`;

// ---- queue ----

export const listQueue = () => request<QueueItem[]>("/api/queue");
export const cancelJob = (jobId: string) =>
  request<void>(`/api/queue/${encodeURIComponent(jobId)}`, { method: "DELETE" });
