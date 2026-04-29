<script lang="ts">
  import * as api from "$lib/api";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = { onopenRegex: () => void };
  const { onopenRegex }: Props = $props();

  let count = $derived.by(() => {
    framesStore.selectionVersion; // reactive dependency
    return framesStore.selection.count();
  });

  async function deleteSelected() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    if (!confirm(`Delete ${filenames.length} frame${filenames.length === 1 ? "" : "s"}?`)) return;
    await api.bulkDeleteFrames(slug, filenames);
    framesStore.removeLocal(filenames);
  }
</script>

{#if count > 0}
  <div class="gradient-accent text-white px-1 pl-3 py-1 rounded-full inline-flex items-center gap-1.5 text-xs font-medium border border-white/10 shadow-[0_2px_12px_rgba(99,102,241,0.4)]">
    <span class="bg-white/20 px-2 py-0.5 rounded-full">{count} selected</span>
    <button
      type="button"
      onclick={deleteSelected}
      class="bg-white/15 hover:bg-red-500/60 rounded-full px-2.5 py-1 transition-colors"
    >Delete</button>
    <button
      type="button"
      onclick={onopenRegex}
      class="bg-white/15 hover:bg-white/25 rounded-full px-2.5 py-1 transition-colors"
    >Regex…</button>
    <button
      type="button"
      onclick={() => framesStore.clear()}
      class="opacity-70 hover:opacity-100 px-1.5 py-1"
      title="Clear selection"
    >✕</button>
  </div>
{/if}
