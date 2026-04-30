<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import type { FrameRecord } from "$lib/types";
  import TagPill from "./TagPill.svelte";

  type Props = {
    frame: FrameRecord;
    selected: boolean;
    /** Left-click on the image area opens the preview modal. */
    onpreview: () => void;
    /** Middle-click on the image area, OR clicking the toggle pill, toggles
     *  selection. Mods are forwarded so shift-range still works on middle-click. */
    onselect: (mods: { shift: boolean; ctrl: boolean }) => void;
  };
  const { frame, selected, onpreview, onselect }: Props = $props();

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

  // Middle-click (auxclick with button === 1) on the image toggles selection.
  // We handle it via mousedown to suppress the browser's auto-scroll cursor.
  function onMouseDown(ev: MouseEvent) {
    if (ev.button === 1) {
      ev.preventDefault();
      onselect({ shift: ev.shiftKey, ctrl: ev.ctrlKey || ev.metaKey });
    }
  }

  // Left click → preview. Shift-click left button still calls onselect so
  // the shift-range bulk-select shortcut keeps working without forcing the
  // user onto the middle button.
  function onMainClick(ev: MouseEvent) {
    if (ev.button !== 0) return;
    if (ev.shiftKey || ev.ctrlKey || ev.metaKey) {
      onselect({ shift: ev.shiftKey, ctrl: ev.ctrlKey || ev.metaKey });
      return;
    }
    onpreview();
  }
</script>

<!-- Wrapper div lets the toggle pill be a real <button> sibling instead
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
    onclick={onMainClick}
    onmousedown={onMouseDown}
    onauxclick={(e) => { if (e.button === 1) e.preventDefault(); }}
    aria-label="Open frame {frame.filename} preview"
  >
    <img src={imageUrl} alt="" class="w-full h-full object-cover" loading="lazy" />

    <span class="absolute top-1.5 left-1.5 px-1.5 py-0.5 text-[9px] bg-black/60 backdrop-blur-sm rounded text-white opacity-0 transition-opacity {hovered ? 'opacity-100' : ''}" style="margin-left: 32px;">
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

  <!-- Top-left toggle pill: emerald + checkmark when selected, neutral otherwise.
       Always visible (the user needs an unmissable affordance even without
       hovering) and on top of the main button via z-20. -->
  <button
    type="button"
    onclick={(e) => {
      e.stopPropagation();
      onselect({ shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey });
    }}
    title={selected ? "Deselect" : "Select"}
    aria-label={selected ? "Deselect frame" : "Select frame"}
    aria-pressed={selected}
    class="absolute top-1.5 left-1.5 w-6 h-6 rounded-full text-xs leading-none flex items-center justify-center z-20 transition-all border
      {selected
        ? 'bg-emerald-500 border-emerald-300 text-white shadow-[0_0_10px_rgba(16,185,129,0.6)] opacity-100'
        : 'bg-black/60 border-white/30 text-white/80 hover:bg-black/80 opacity-0 group-hover:opacity-100 focus:opacity-100'}"
  >{selected ? "✓" : "○"}</button>
</div>
