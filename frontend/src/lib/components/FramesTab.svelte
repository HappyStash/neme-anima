<script lang="ts">
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import FrameThumb from "./FrameThumb.svelte";

  $effect(() => {
    const slug = projectsStore.active?.slug;
    if (slug) {
      framesStore.refresh(slug, viewStore.sourceFilter ? { source: viewStore.sourceFilter } : {});
    }
  });

  function handleClick(index: number, ev: MouseEvent) {
    framesStore.click(index, { shift: ev.shiftKey, ctrl: ev.ctrlKey || ev.metaKey });
  }

  let cols = $derived(viewStore.density);
</script>

<div class="mt-4">
  {#if framesStore.loading}
    <p class="text-slate-500 py-12 text-center">Loading frames…</p>
  {:else if framesStore.items.length === 0}
    <div class="py-24 text-center text-slate-500">
      <p class="text-lg mb-1">No frames yet.</p>
      <p class="text-sm">Add a video in Sources and run extract.</p>
    </div>
  {:else}
    <div
      class="grid gap-2"
      style="grid-template-columns: repeat({cols}, minmax(0, 1fr));"
    >
      {#each framesStore.items as f, i (f.filename)}
        {@const _v = framesStore.selectionVersion /* force re-render on selection change */}
        <FrameThumb
          frame={f}
          selected={framesStore.selection.has(f.filename)}
          onclick={(ev) => handleClick(i, ev)}
        />
      {/each}
    </div>
  {/if}
</div>
