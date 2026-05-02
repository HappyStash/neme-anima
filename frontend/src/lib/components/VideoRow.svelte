<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { jobsStore } from "$lib/stores/jobs.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import type { RefImage, Source } from "$lib/types";
  import PipelineRunner from "./PipelineRunner.svelte";
  import RefStrip from "./RefStrip.svelte";

  type Props = {
    source: Source;
    sourceIdx: number;
    /** Refs of the *active* character — they're what the per-video strip
     *  toggles on/off via per-character opt-outs. */
    projectRefs: readonly RefImage[];
  };
  const { source, sourceIdx, projectRefs }: Props = $props();

  // Per-video opt-outs are now keyed by character slug — surface only the
  // active character's list so the existing "active refs" math still
  // works on a flat array of paths.
  let activeExcluded = $derived(
    source.excluded_refs[viewStore.activeCharacterSlug] ?? [],
  );

  let busy = $state(false);
  let thumbBroken = $state(false);

  async function run() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    busy = true;
    try {
      const { job_id } = await api.extractSource(slug, sourceIdx);
      jobsStore.seedPending({ job_id, project: slug, source_idx: sourceIdx, kind: "extract" });
    } finally {
      busy = false;
    }
  }

  async function rerun() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    busy = true;
    try {
      const { job_id } = await api.rerunSource(slug, sourceIdx);
      jobsStore.seedPending({ job_id, project: slug, source_idx: sourceIdx, kind: "rerun" });
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

  let activeRefs = $derived(projectRefs.length - activeExcluded.length);
  let basename = $derived(source.path.split("/").pop() ?? source.path);
  let thumbUrl = $derived.by(() => {
    const slug = projectsStore.active?.slug;
    return slug ? api.sourceThumbnailUrl(slug, sourceIdx) : "";
  });
  let job = $derived.by(() => {
    const slug = projectsStore.active?.slug;
    return slug ? jobsStore.forSource(slug, sourceIdx) : null;
  });
  let pipelineActive = $derived.by(() => {
    if (!job) return false;
    return !job.stages.every((s) => s.status === "done")
      && !job.stages.some((s) => s.status === "failed");
  });
  let actionsDisabled = $derived(busy || pipelineActive);
</script>

<div
  class="bg-ink-900 border rounded-xl px-3 py-3 mb-2.5 grid grid-cols-[auto_1fr_auto_auto] gap-3 items-center
    {source.extracted ? 'border-emerald-500/70 hover:border-emerald-400' : 'border-ink-700 hover:border-ink-600'}"
  title={source.extracted ? 'Already extracted (frames on disk)' : ''}
>
  <!-- Thumbnail (left). Falls back to a play glyph if extraction fails. -->
  <div class="w-24 h-14 rounded overflow-hidden bg-ink-950 border border-ink-800 flex-shrink-0 flex items-center justify-center">
    {#if thumbUrl && !thumbBroken}
      <img
        src={thumbUrl}
        alt={basename}
        loading="lazy"
        onerror={() => (thumbBroken = true)}
        class="w-full h-full object-cover"
      />
    {:else}
      <span class="text-slate-600 text-lg">▶</span>
    {/if}
  </div>

  <!-- Center: when a job exists, split 50/50 between the source-info block and the pipeline. -->
  <div class={job ? "grid grid-cols-2 gap-3 min-w-0 items-center" : "min-w-0"}>
    <div class="flex flex-col gap-1.5 min-w-0">
      <div class="flex items-center gap-2 min-w-0">
        <span class="text-sm text-slate-200 font-medium truncate" title={source.path}>{basename}</span>
      </div>
      <div class="text-xs text-slate-500">
        {activeRefs} of {projectRefs.length} ref{projectRefs.length === 1 ? "" : "s"} active
      </div>
      <RefStrip
        sourceIdx={sourceIdx}
        refPaths={projectRefs.map((r) => r.path)}
        excluded={activeExcluded}
      />
    </div>
    {#if job}
      <PipelineRunner {job} />
    {/if}
  </div>

  <div class="flex gap-1.5">
    <button
      type="button"
      onclick={run}
      disabled={actionsDisabled || activeRefs === 0}
      title={pipelineActive ? "Pipeline already running" : ""}
      class="px-3 py-1.5 text-xs rounded gradient-accent text-white disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_2px_8px_rgba(99,102,241,0.3)]"
    >Run</button>
    <button
      type="button"
      onclick={rerun}
      disabled={actionsDisabled}
      title={pipelineActive ? "Pipeline already running" : ""}
      class="px-3 py-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-300 border border-ink-700 disabled:opacity-40 disabled:cursor-not-allowed"
    >Rerun</button>
  </div>
  <button
    type="button"
    onclick={remove}
    disabled={actionsDisabled}
    title={pipelineActive ? "Pipeline running — wait for it to finish" : "Remove from project"}
    class="text-slate-600 hover:text-red-400 text-xs px-2 py-1 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:text-slate-600"
  >✕</button>
</div>
