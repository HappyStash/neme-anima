<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = {
    sourceIdx: number;
    /** Character whose opt-outs this strip toggles. The video row now
     *  shows one strip per character, so the slug must come in from the
     *  parent rather than being read off the globally-active chip. */
    characterSlug: string;
    refPaths: readonly string[];
    /** This character's per-video opt-outs — flat list of ref paths. */
    excluded: readonly string[];
    /** Full-opacity rgba for the active-ref outline. Driven by the
     *  parent's per-character color so each strip's "this ref is active"
     *  ring matches the chip color, not a fixed accent. */
    activeRingRgba?: string;
  };
  const {
    sourceIdx,
    characterSlug,
    refPaths,
    excluded,
    activeRingRgba = "rgba(99,102,241,1)",
  }: Props = $props();

  function isActive(path: string): boolean {
    return !excluded.includes(path);
  }

  async function toggle(path: string) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const next = isActive(path)
      ? [...excluded, path]
      : excluded.filter((p) => p !== path);
    await api.setExcludedRefs(slug, sourceIdx, next, characterSlug);
    if (projectsStore.active) await projectsStore.load(projectsStore.active.slug);
  }

  let slug = $derived(projectsStore.active?.slug ?? "");
</script>

<div class="flex flex-wrap items-center gap-1">
  {#each refPaths as path (path)}
    {@const active = isActive(path)}
    <button
      type="button"
      onclick={() => toggle(path)}
      style={active ? `box-shadow: 0 0 0 1.5px ${activeRingRgba}` : undefined}
      class="w-6 h-6 rounded overflow-hidden transition-all relative bg-ink-800 border border-ink-700
        {active ? '' : 'grayscale brightness-50 opacity-50 hover:opacity-90 hover:brightness-75'}"
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
