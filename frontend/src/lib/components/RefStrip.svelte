<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = {
    sourceIdx: number;
    refPaths: readonly string[];
    excluded: readonly string[];
  };
  const { sourceIdx, refPaths, excluded }: Props = $props();

  function isActive(path: string): boolean {
    return !excluded.includes(path);
  }

  async function toggle(path: string) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const next = isActive(path)
      ? [...excluded, path]
      : excluded.filter((p) => p !== path);
    await api.setExcludedRefs(slug, sourceIdx, next);
    if (projectsStore.active) await projectsStore.load(projectsStore.active.slug);
  }

  let slug = $derived(projectsStore.active?.slug ?? "");
</script>

<div class="flex flex-wrap items-center gap-1">
  <span class="text-[9px] uppercase tracking-wide text-slate-600 mr-1">refs</span>
  {#each refPaths as path (path)}
    {@const active = isActive(path)}
    <button
      type="button"
      onclick={() => toggle(path)}
      class="w-6 h-6 rounded overflow-hidden transition-all relative bg-ink-800 border border-ink-700
        {active ? 'shadow-[0_0_0_1.5px_rgba(99,102,241,1)]' : 'grayscale brightness-50 opacity-50 hover:opacity-90 hover:brightness-75'}"
      title={path}
    >
      {#if slug}
        <img
          src={api.refImageUrl(slug, path)}
          alt={path.split('/').pop() ?? ''}
          loading="lazy"
          class="w-full h-full object-cover"
        />
      {/if}
      {#if !active}
        <span class="absolute inset-x-0 top-1/2 -translate-y-1/2 h-px bg-red-500 -rotate-[22deg] shadow-[0_0_4px_rgba(239,68,68,0.6)]"></span>
      {/if}
    </button>
  {/each}
</div>
