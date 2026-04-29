<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import type { FrameRecord } from "$lib/types";
  import TagPill from "./TagPill.svelte";

  type Props = {
    frame: FrameRecord;
    selected: boolean;
    onclick: (ev: MouseEvent) => void;
  };
  const { frame, selected, onclick }: Props = $props();

  let tagText = $state<string | null>(null);
  let hovered = $state(false);

  async function loadTags() {
    if (tagText !== null) return;
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    try {
      const r = await api.getTags(slug, frame.filename);
      tagText = r.text;
    } catch {
      tagText = "";
    }
  }

  async function saveTagText(next: string) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    await api.putTags(slug, frame.filename, next);
    tagText = next;
  }

  let imageUrl = $derived(
    projectsStore.active ? api.frameImageUrl(projectsStore.active.slug, frame.filename) : "",
  );

  let pills = $derived(
    (tagText ?? "")
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean),
  );

  function replaceTag(oldTag: string, newTag: string) {
    const next = pills.map((t) => (t === oldTag ? newTag : t)).join(", ");
    void saveTagText(next);
  }
</script>

<button
  type="button"
  class="relative aspect-[3/4] rounded-lg overflow-hidden cursor-pointer transition-transform duration-150 hover:scale-[1.02] hover:z-10
    {selected ? 'ring-2 ring-accent-500 shadow-[0_0_20px_rgba(99,102,241,0.4)]' : 'shadow-md'}"
  onclick={onclick}
  onmouseenter={() => { hovered = true; void loadTags(); }}
  onmouseleave={() => (hovered = false)}
>
  <img src={imageUrl} alt="" class="w-full h-full object-cover" loading="lazy" />

  <span class="absolute bottom-1.5 left-1.5 px-1.5 py-0.5 text-[9px] bg-black/60 backdrop-blur-sm rounded text-white opacity-0 transition-opacity {hovered ? 'opacity-100' : ''}">
    {frame.video_stem}
  </span>

  {#if selected}
    <span class="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-accent-500 text-white text-xs flex items-center justify-center font-bold">✓</span>
  {/if}

  {#if hovered}
    <div class="absolute inset-x-0 bottom-0 max-h-[60%] overflow-hidden p-1.5 pt-6 flex flex-wrap gap-1
      bg-gradient-to-t from-black/85 via-black/70 to-transparent">
      {#each pills as p (p)}
        <TagPill text={p} onreplace={(next) => replaceTag(p, next)} />
      {/each}
    </div>
  {/if}
</button>
