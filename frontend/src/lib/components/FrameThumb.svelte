<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import type { FrameRecord } from "$lib/types";
  import TagPill from "./TagPill.svelte";

  type Props = {
    frame: FrameRecord;
    selected: boolean;
    onclick: (ev: MouseEvent) => void;
    onexpand: () => void;
  };
  const { frame, selected, onclick, onexpand }: Props = $props();

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

<!-- Wrapper div lets the "expand" action be a real <button> sibling instead
     of nesting a button inside a button (invalid HTML). -->
<div
  class="relative aspect-[3/4] group"
  onmouseenter={() => { hovered = true; void loadTags(); }}
  onmouseleave={() => (hovered = false)}
  role="presentation"
>
  <button
    type="button"
    class="absolute inset-0 rounded-lg overflow-hidden cursor-pointer transition-transform duration-150 hover:scale-[1.02] hover:z-10
      {selected ? 'shadow-[0_0_20px_rgba(99,102,241,0.4)]' : 'shadow-md'}"
    onclick={onclick}
  >
    <img src={imageUrl} alt="" class="w-full h-full object-cover" loading="lazy" />

    <span class="absolute top-1.5 left-1.5 px-1.5 py-0.5 text-[9px] bg-black/60 backdrop-blur-sm rounded text-white opacity-0 transition-opacity {hovered ? 'opacity-100' : ''}">
      {frame.video_stem}
    </span>

    {#if selected}
      <!-- Inset border overlay drawn on top of the image, with a faint tint
           so the selection state is unmissable on any image. -->
      <span class="absolute inset-0 rounded-lg border-[3px] border-accent-500 bg-accent-500/20 pointer-events-none"></span>
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

  <!-- Top-right expand button. Sibling of the main button, on top via z-20.
       stopPropagation keeps the click off the selection toggle. -->
  <button
    type="button"
    onclick={(e) => { e.stopPropagation(); onexpand(); }}
    title="View full size"
    aria-label="View frame full size"
    class="absolute top-1.5 right-1.5 w-6 h-6 rounded-full bg-black/60 hover:bg-black/85 backdrop-blur-sm text-white text-xs leading-none flex items-center justify-center z-20 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
  >⤢</button>
</div>
