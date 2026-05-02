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

  // ---- smart Extract / Re-process state ----
  // The buttons mean different things depending on the detection cache.
  // We render both at all times (so the user can always force the path
  // they want) but mute the redundant one and fence off the dangerous one
  // with a tooltip + disabled state.
  let cacheState = $derived(source.extraction_cache);

  /** Extract is the heavy "scan from scratch" pipeline. Disabled when
   *  there's a fresh cache and no scan-affecting threshold has changed
   *  — the user would just be paying the YOLO cost again for no reason. */
  let extractDisabled = $derived(
    actionsDisabled || activeRefs === 0 || cacheState === "current",
  );
  let extractTooltip = $derived(
    pipelineActive
      ? "Pipeline already running"
      : activeRefs === 0
        ? "Add at least one active reference for the current character"
        : cacheState === "current"
          ? "Already extracted with these scan settings — use Re-process to re-evaluate identification, frames, dedup, or tags"
          : cacheState === "stale"
            ? "Scene / detection / tracking settings changed since last extract — Extract will rebuild the detection cache"
            : "Run the full pipeline: detect every character in every scene, track them, identify, crop, and tag",
  );
  /** Visual emphasis: primary when there's no cache OR cache went stale,
   *  muted when we'd rather the user click Re-process. */
  let extractPrimary = $derived(
    cacheState === "none" || cacheState === "stale",
  );

  /** Re-process replays identification/selection/crop/dedup/tag with
   *  the cached scenes + tracklets. Disabled when there's no cache. */
  let rerunDisabled = $derived(
    actionsDisabled || activeRefs === 0 || cacheState === "none",
  );
  let rerunTooltip = $derived(
    pipelineActive
      ? "Pipeline already running"
      : activeRefs === 0
        ? "Add at least one active reference for the current character"
        : cacheState === "none"
          ? "No detection cache yet — run Extract first to build it"
          : cacheState === "stale"
            ? "Detection cache is stale (scan settings changed) — Re-process will use the OLD detections; consider Extract instead"
            : "Quickly re-evaluate identification, frame selection, dedup, and tagging using the cached detections — typically under a minute",
  );
  let rerunPrimary = $derived(cacheState === "current");
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
      disabled={extractDisabled}
      title={extractTooltip}
      class="px-3 py-1.5 text-xs rounded inline-flex items-center gap-1
        disabled:opacity-40 disabled:cursor-not-allowed
        {extractPrimary
          ? 'gradient-accent text-white shadow-[0_2px_8px_rgba(99,102,241,0.3)]'
          : 'bg-ink-800 hover:bg-ink-700 text-slate-300 border border-ink-700'}"
    >
      Extract{#if cacheState === "stale"}
        <!-- Subtle warning glyph: the existing detection cache no longer
             reflects the current scene/detect/track settings. Click to
             rebuild from scratch. -->
        <span aria-hidden="true" class="text-amber-300" title="Scan settings changed since last extract">!</span>
      {/if}
    </button>
    <button
      type="button"
      onclick={rerun}
      disabled={rerunDisabled}
      title={rerunTooltip}
      class="px-3 py-1.5 text-xs rounded inline-flex items-center disabled:opacity-40 disabled:cursor-not-allowed
        {rerunPrimary
          ? 'gradient-accent text-white shadow-[0_2px_8px_rgba(99,102,241,0.3)]'
          : 'bg-ink-800 hover:bg-ink-700 text-slate-300 border border-ink-700'}"
    >Re-process</button>
  </div>
  <button
    type="button"
    onclick={remove}
    disabled={actionsDisabled}
    title={pipelineActive ? "Pipeline running — wait for it to finish" : "Remove from project"}
    class="text-slate-600 hover:text-red-400 text-xs px-2 py-1 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:text-slate-600"
  >✕</button>
</div>
