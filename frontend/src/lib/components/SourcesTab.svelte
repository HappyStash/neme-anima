<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import DropZone from "./DropZone.svelte";
  import VideoRow from "./VideoRow.svelte";

  async function addVideos(paths: string[]) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    await api.addSources(slug, paths);
    await projectsStore.load(slug);
  }

  async function addRefs(paths: string[]) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    await api.addRefs(slug, paths);
    await projectsStore.load(slug);
  }

  async function removeRef(path: string) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const name = path.split("/").pop() ?? path;
    if (!confirm(`Remove reference image “${name}” from this project?`)) return;
    await api.removeRef(slug, path);
    await projectsStore.load(slug);
  }

  function gradientFor(path: string): string {
    let hash = 0;
    for (let i = 0; i < path.length; i++) hash = (hash * 31 + path.charCodeAt(i)) >>> 0;
    const hue = hash % 360;
    return `linear-gradient(135deg, hsl(${hue}, 60%, 65%), hsl(${(hue + 40) % 360}, 70%, 35%))`;
  }
</script>

<div class="mt-4">
  <div class="grid grid-cols-2 gap-3 mb-4">
    <DropZone
      title="Drop video files here"
      subtitle=".mkv · .mp4 · .webm — paths saved, files not copied"
      icon="▶"
      onpaths={addVideos}
    />
    <DropZone
      title="Drop reference images here"
      subtitle="One portrait is usually enough; more refs improve recall"
      icon="◇"
      onpaths={addRefs}
    />
  </div>

  {#if projectsStore.active}
    {#if projectsStore.active.refs.length > 0}
      <p class="text-[10px] uppercase text-slate-500 tracking-wide mb-2">project references — apply to every video by default</p>
      <div class="bg-ink-950 border border-ink-700 rounded-xl px-3 py-2.5 flex items-center gap-2 mb-4">
        <span class="text-[10px] uppercase text-slate-600 tracking-wide w-24">Refs ({projectsStore.active.refs.length})</span>
        <div class="flex gap-1.5 flex-1 flex-wrap">
          {#each projectsStore.active.refs as r (r.path)}
            <div class="relative group w-9 h-9">
              <div class="w-9 h-9 rounded" title={r.path} style="background: {gradientFor(r.path)}"></div>
              <button
                type="button"
                onclick={() => removeRef(r.path)}
                title="Remove reference"
                aria-label="Remove reference {r.path.split('/').pop()}"
                class="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-ink-900 border border-ink-700 text-slate-500 hover:text-red-400 hover:border-red-400/60 text-[9px] leading-none flex items-center justify-center opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
              >✕</button>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <p class="text-[10px] uppercase text-slate-500 tracking-wide mb-2">videos in this project</p>
    {#if projectsStore.active.sources.length === 0}
      <p class="text-slate-500 text-sm py-8 text-center">No videos yet. Drop some above.</p>
    {:else}
      {#each projectsStore.active.sources as s, i (s.path)}
        <VideoRow source={s} sourceIdx={i} projectRefs={projectsStore.active.refs} />
      {/each}
    {/if}
  {/if}
</div>
