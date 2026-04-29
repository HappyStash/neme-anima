<script lang="ts">
  import { queueStore } from "$lib/stores/queue.svelte";

  let active = $derived(queueStore.totalActive());
  let running = $derived(queueStore.running());
</script>

<div class="flex items-center gap-1.5 px-2.5 py-1 bg-ink-900 border border-ink-700 rounded-full text-xs">
  {#if active > 0}
    <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)] animate-pulse"></span>
    <span class="text-slate-200">
      {active} job{active === 1 ? "" : "s"}
      {#if running}· {String((running.payload as Record<string, unknown>).kind ?? "?")}{/if}
    </span>
  {:else}
    <span class="w-1.5 h-1.5 rounded-full bg-slate-600"></span>
    <span class="text-slate-500">idle</span>
  {/if}
</div>
