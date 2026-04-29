<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import type { RefImage, Source } from "$lib/types";
  import RefStrip from "./RefStrip.svelte";

  type Props = {
    source: Source;
    sourceIdx: number;
    projectRefs: readonly RefImage[];
  };
  const { source, sourceIdx, projectRefs }: Props = $props();

  let busy = $state(false);

  async function run() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    busy = true;
    try {
      await api.extractSource(slug, sourceIdx);
    } finally {
      busy = false;
    }
  }

  async function rerun() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    busy = true;
    try {
      await api.rerunSource(slug, sourceIdx);
    } finally {
      busy = false;
    }
  }

  async function remove() {
    if (!confirm(`Remove ${source.path.split("/").pop()} from project?`)) return;
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    await api.removeSource(slug, sourceIdx);
    if (projectsStore.active) await projectsStore.load(projectsStore.active.slug);
  }

  let activeRefs = $derived(projectRefs.length - source.excluded_refs.length);
  let basename = $derived(source.path.split("/").pop() ?? source.path);
</script>

<div class="bg-ink-900 border border-ink-700 rounded-xl px-4 py-3.5 mb-2.5 grid grid-cols-[1fr_auto_auto] gap-3.5 items-center hover:border-ink-600">
  <div class="flex flex-col gap-1.5">
    <div class="flex items-center gap-2">
      <span class="text-sm text-slate-200 font-medium">{basename}</span>
      <span class="text-[10px] uppercase tracking-wide text-slate-500">{source.extraction_runs.length} run{source.extraction_runs.length === 1 ? "" : "s"}</span>
    </div>
    <div class="text-xs text-slate-500">
      {activeRefs} of {projectRefs.length} ref{projectRefs.length === 1 ? "" : "s"} active
    </div>
    <RefStrip
      sourceIdx={sourceIdx}
      refPaths={projectRefs.map((r) => r.path)}
      excluded={source.excluded_refs}
    />
  </div>
  <div class="flex gap-1.5">
    <button
      type="button"
      onclick={run}
      disabled={busy || activeRefs === 0}
      class="px-3 py-1.5 text-xs rounded gradient-accent text-white disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_2px_8px_rgba(99,102,241,0.3)]"
    >Run</button>
    <button
      type="button"
      onclick={rerun}
      disabled={busy}
      class="px-3 py-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-300 border border-ink-700"
    >Rerun</button>
  </div>
  <button
    type="button"
    onclick={remove}
    class="text-slate-600 hover:text-red-400 text-xs px-2 py-1"
  >✕</button>
</div>
