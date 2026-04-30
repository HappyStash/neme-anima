<script lang="ts">
  import * as api from "$lib/api";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";

  type Props = { onopenRegex: () => void };
  const { onopenRegex }: Props = $props();

  let count = $derived.by(() => {
    framesStore.selectionVersion; // reactive dependency
    return framesStore.selection.count();
  });

  let onFramesTab = $derived(viewStore.tab === "frames");
  let total = $derived(framesStore.items.length);
  let allSelected = $derived(count > 0 && count === total);

  async function deleteSelected() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    if (!confirm(`Delete ${filenames.length} frame${filenames.length === 1 ? "" : "s"}?`)) return;
    await api.bulkDeleteFrames(slug, filenames);
    framesStore.removeLocal(filenames);
  }

  function toggleSelectAll() {
    if (allSelected) framesStore.clear();
    else framesStore.selectAll();
  }
</script>

<!-- Pill is sized to match other top-bar items (h-7) so toggling the
     "X selected" actions never resizes the bar. -->
{#if onFramesTab}
  <div class="flex items-center gap-2 h-7">
    <button
      type="button"
      onclick={toggleSelectAll}
      disabled={total === 0}
      class="h-7 px-3 rounded-full text-xs bg-ink-900 border border-ink-700 text-slate-300 hover:bg-ink-800 hover:text-slate-100 disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center"
    >{allSelected ? "Deselect all" : "Select all"}</button>

    <span
      class="h-7 px-3 rounded-full text-xs bg-ink-900 border border-ink-700 text-slate-400 inline-flex items-center"
      title="Total frames in the current view"
    >{total} frame{total === 1 ? "" : "s"}</span>

    {#if count > 0}
      <div class="gradient-accent text-white h-7 pl-3 pr-1 rounded-full inline-flex items-center gap-1.5 text-xs font-medium border border-white/10 shadow-[0_2px_12px_rgba(99,102,241,0.4)]">
        <span class="bg-white/20 px-2 py-0.5 rounded-full leading-none">{count} selected</span>
        <button
          type="button"
          onclick={deleteSelected}
          class="bg-white/15 hover:bg-red-500/60 rounded-full px-2.5 h-5 transition-colors inline-flex items-center"
        >Delete</button>
        <button
          type="button"
          onclick={onopenRegex}
          class="bg-white/15 hover:bg-white/25 rounded-full px-2.5 h-5 transition-colors inline-flex items-center"
        >Regex…</button>
        <button
          type="button"
          onclick={() => framesStore.clear()}
          class="opacity-70 hover:opacity-100 h-5 w-5 inline-flex items-center justify-center"
          title="Clear selection"
        >✕</button>
      </div>
    {/if}
  </div>
{/if}
