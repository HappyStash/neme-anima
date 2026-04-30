<script lang="ts">
  import type { JobStages, PipelineStage } from "$lib/types";

  type Props = { job: JobStages };
  const { job }: Props = $props();

  let allDone = $derived(job.stages.every((s) => s.status === "done"));
  let anyFailed = $derived(job.stages.some((s) => s.status === "failed"));
  let runningStage = $derived(job.stages.find((s) => s.status === "running") ?? null);

  let overallPct = $derived.by(() => {
    if (job.stages.length === 0) return 0;
    let sum = 0;
    for (const s of job.stages) {
      if (s.status === "done") sum += 1;
      else if (s.status === "running") sum += s.pct || 0;
    }
    return Math.round((sum / job.stages.length) * 100);
  });

  function statusGlyph(s: PipelineStage): string {
    switch (s.status) {
      case "done": return "✓";
      case "failed": return "✕";
      case "running": return "▶";
      default: return "·";
    }
  }
</script>

<div
  class="rounded-lg border bg-ink-950/70 backdrop-blur px-3 py-2.5 transition-colors w-full
    {anyFailed
      ? 'border-red-500/40 shadow-[0_0_20px_rgba(239,68,68,0.15)]'
      : allDone
        ? 'border-emerald-500/40 shadow-[0_0_20px_rgba(16,185,129,0.15)]'
        : 'border-accent-500/30 shadow-[0_0_24px_rgba(99,102,241,0.15)]'}"
>
  <!-- Header: overall state + percentage -->
  <div class="flex items-center justify-between mb-2 min-w-0">
    <div class="flex items-center gap-2 min-w-0">
      {#if allDone}
        <span class="text-emerald-400 text-sm flex-shrink-0">✓</span>
        <span class="text-[11px] uppercase tracking-wide text-emerald-400 font-medium flex-shrink-0">Pipeline complete</span>
        {#if job.summary}
          <span class="text-[11px] text-slate-400 truncate">
            · kept <span class="text-emerald-300 font-mono">{job.summary.kept ?? 0}</span>
            · rejected <span class="text-slate-300 font-mono">{job.summary.rejected ?? 0}</span>
          </span>
        {/if}
      {:else if anyFailed}
        <span class="text-red-400 text-sm flex-shrink-0">✕</span>
        <span class="text-[11px] uppercase tracking-wide text-red-400 font-medium flex-shrink-0">Pipeline failed</span>
      {:else}
        <span class="relative inline-flex w-2 h-2 flex-shrink-0">
          <span class="absolute inset-0 rounded-full bg-accent-400 animate-ping opacity-75"></span>
          <span class="relative inline-flex rounded-full w-2 h-2 bg-accent-500"></span>
        </span>
        <span class="text-[11px] uppercase tracking-wide text-accent-300 font-medium truncate">
          {runningStage ? runningStage.label : "Starting"}
        </span>
      {/if}
    </div>
    <span class="text-[11px] font-mono text-slate-400 flex-shrink-0 ml-2 tabular-nums">{overallPct}%</span>
  </div>

  <!-- Stage chips with connectors -->
  <div class="flex items-stretch gap-0 min-w-0">
    {#each job.stages as s, i (s.key)}
      <div class="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10.5px] font-medium leading-none transition-all flex-shrink-0
        {s.status === 'done'
          ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
          : s.status === 'running'
            ? 'bg-accent-500/20 text-accent-200 border border-accent-400/50 shadow-[0_0_12px_rgba(99,102,241,0.4)]'
            : s.status === 'failed'
              ? 'bg-red-500/15 text-red-300 border border-red-500/30'
              : 'bg-ink-900 text-slate-500 border border-ink-800'}"
        title="{s.label}{s.message ? ' — ' + s.message : ''}"
      >
        <span class="font-mono w-3 text-center">{statusGlyph(s)}</span>
        <span class="whitespace-nowrap">{s.label}</span>
      </div>
      {#if i < job.stages.length - 1}
        <div class="flex-1 self-center mx-1 min-w-[8px] h-px
          {s.status === 'done' ? 'bg-emerald-500/40' : 'bg-ink-700'}"></div>
      {/if}
    {/each}
  </div>

  <!-- Progress bar + message for the running stage (or last failed) -->
  {#if runningStage || anyFailed}
    {@const focus = runningStage ?? job.stages.find((s) => s.status === "failed")!}
    <div class="mt-2.5">
      <div class="flex items-center justify-between mb-1 min-w-0">
        <span class="text-[10.5px] font-mono text-slate-400 truncate">{focus.message || "…"}</span>
        {#if focus.status === "running" && focus.total > 0}
          <span class="text-[10.5px] font-mono text-slate-500 ml-2 flex-shrink-0 tabular-nums">
            {focus.current.toLocaleString()} / {focus.total.toLocaleString()}
          </span>
        {/if}
      </div>
      <div class="h-1.5 rounded-full overflow-hidden bg-ink-900 border border-ink-800">
        <div
          class="h-full transition-all duration-300 ease-out
            {focus.status === 'failed'
              ? 'bg-gradient-to-r from-red-500 to-red-400'
              : 'bg-gradient-to-r from-accent-500 to-accent-400 animate-pulse'}"
          style="width: {focus.status === 'failed' ? 100 : Math.max(2, Math.round((focus.pct || 0) * 100))}%"
        ></div>
      </div>
    </div>
  {/if}
</div>
