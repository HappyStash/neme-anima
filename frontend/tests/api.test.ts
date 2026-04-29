import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../src/lib/api";

describe("api client", () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("listProjects GETs /api/projects", async () => {
    const mock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([]), { status: 200 })
    );
    globalThis.fetch = mock as any;
    const result = await api.listProjects();
    expect(result).toEqual([]);
    expect(mock).toHaveBeenCalledWith("/api/projects", expect.objectContaining({ method: "GET" }));
  });

  it("createProject POSTs name + folder", async () => {
    const mock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ slug: "x" }), { status: 201 })
    );
    globalThis.fetch = mock as any;
    await api.createProject({ name: "x", folder: "/tmp/x" });
    expect(mock).toHaveBeenCalledWith(
      "/api/projects",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
        body: JSON.stringify({ name: "x", folder: "/tmp/x" }),
      }),
    );
  });

  it("addSources hits the slug-scoped path", async () => {
    const mock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ added: [], skipped: [] }), { status: 200 })
    );
    globalThis.fetch = mock as any;
    await api.addSources("megumin", ["/v.mkv"]);
    expect(mock).toHaveBeenCalledWith(
      "/api/projects/megumin/sources",
      expect.objectContaining({
        body: JSON.stringify({ paths: ["/v.mkv"] }),
      }),
    );
  });

  it("setExcludedRefs sends PATCH with body", async () => {
    const mock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ excluded_refs: ["/r.png"] }), { status: 200 })
    );
    globalThis.fetch = mock as any;
    await api.setExcludedRefs("p", 0, ["/r.png"]);
    expect(mock).toHaveBeenCalledWith(
      "/api/projects/p/sources/0",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ excluded_refs: ["/r.png"] }),
      }),
    );
  });

  it("listFrames passes the source query parameter", async () => {
    const mock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ count: 0, items: [] }), { status: 200 })
    );
    globalThis.fetch = mock as any;
    await api.listFrames("p", { source: "ep01" });
    const call = mock.mock.calls[0][0] as string;
    expect(call).toContain("source=ep01");
  });

  it("non-2xx throws an ApiError with status + body", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "missing" }), { status: 404 })
    ) as any;
    await expect(api.getProject("nope")).rejects.toThrow(/404/);
  });
});
