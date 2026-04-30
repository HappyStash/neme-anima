<script lang="ts">
  import { viewStore } from "$lib/stores/view.svelte";

  const tabs: { key: typeof viewStore.tab; label: string }[] = [
    { key: "sources", label: "Sources" },
    { key: "frames", label: "Frames" },
    { key: "training", label: "Training" },
    { key: "settings", label: "Settings" },
  ];
</script>

<div class="flex bg-ink-900 rounded-md p-0.5">
  {#each tabs as t (t.key)}
    <button
      class="px-3 py-1 text-xs rounded transition-all
        {viewStore.tab === t.key
          ? 'bg-ink-700 text-white'
          : 'text-slate-400 hover:text-slate-200'}"
      onclick={() => (viewStore.tab = t.key)}
    >
      {t.label}
    </button>
  {/each}
</div>
