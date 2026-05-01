<script lang="ts">
  import * as api from "$lib/api";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import type { FrameRecord } from "$lib/types";
  import DescriptionModal from "./DescriptionModal.svelte";
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
  // Local override so a fresh save through the description modal flips the
  // badge without us having to refetch the whole frames list.
  let hasDescriptionLocal = $state<boolean | null>(null);
  let hasDescription = $derived(hasDescriptionLocal ?? frame.has_description);

  // Reset the local override whenever the underlying prop changes — that
  // catches both filename changes (parent swapped the row) and the
  // server-side flag flipping after a list refresh post-bulk-retag.
  // Also re-runs when either per-filename version counter ticks so the
  // cached sidecar gets dropped after a WD14 retag *or* an LLM describe
  // finishes — the FrameRecord doesn't change for in-place rewrites, only
  // the .txt does, and the badge tooltip reads from this cache.
  $effect(() => {
    void frame.filename;
    void frame.has_description;
    void framesStore.retaggedVersion.get(frame.filename);
    void framesStore.describedVersion.get(frame.filename);
    hasDescriptionLocal = null;
    tagText = null; // force a fresh tag fetch on next hover
  });

  // Frames currently in flight for a bulk re-tag / re-describe — render a
  // spinner overlay and swallow clicks so the user can't kick off a preview
  // or toggle selection on a tile that's actively being written to.
  let processing = $derived(framesStore.processing.has(frame.filename));

  // A sentinel value that, when present in `pills`, renders as the empty
  // editable placeholder created by the "+" button. Picked to be impossible
  // to confuse with a real tag.
  const PLACEHOLDER = " __new__";
  let addingTag = $state(false);

  let descriptionModalOpen = $state(false);

  // One-shot pop animation on the description badge. Driven by a per-filename
  // version counter in framesStore that ActionBar bumps when a single-frame
  // LLM describe call completes. We capture the counter the first time we
  // see a given filename so already-described frames don't pop on initial
  // render or when this component instance is reused for a different tile.
  let popping = $state(false);
  let popTimer: ReturnType<typeof setTimeout> | null = null;
  let lastSeenFilename: string | null = null;
  let popBaseline = 0;

  $effect(() => {
    const fn = frame.filename;
    const v = framesStore.describedVersion.get(fn) ?? 0;
    if (lastSeenFilename !== fn) {
      // First run for this filename — capture baseline silently.
      lastSeenFilename = fn;
      popBaseline = v;
      return;
    }
    if (v > popBaseline) {
      popBaseline = v;
      if (popTimer) clearTimeout(popTimer);
      popping = false;
      // Force a tick so the class toggle restarts the animation when the
      // same frame is popped twice in quick succession.
      requestAnimationFrame(() => { popping = true; });
      popTimer = setTimeout(() => { popping = false; }, 560);
    }
  });

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

  // The .txt sidecar is two-line: comma-separated WD14 tags on row 1, optional
  // LLM description on row 2. Pills only edit row 1; row 2 stays intact.
  function splitSidecar(text: string): { danbooru: string; description: string } {
    if (!text) return { danbooru: "", description: "" };
    const lines = text.split("\n");
    const danbooru = (lines[0] ?? "").trim();
    let rest = lines.slice(1);
    while (rest.length > 0 && rest[0].trim() === "") rest = rest.slice(1);
    const description = rest.join("\n").replace(/\s+$/, "");
    return { danbooru, description };
  }

  async function saveTagsLine(nextDanbooru: string) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const { description } = splitSidecar(tagText ?? "");
    const next = description
      ? `${nextDanbooru}\n${description}`
      : nextDanbooru;
    // Use the server's response so any normalization the route applied
    // (dedupe, whitespace collapse) flows back into the local cache —
    // otherwise the next render would still show the raw text we sent.
    const r = await api.putTags(slug, frame.filename, next);
    tagText = r.text;
  }

  let imageUrl = $derived(
    projectsStore.active ? api.frameImageUrl(projectsStore.active.slug, frame.filename) : "",
  );

  // Dedupe the pill list defensively — the server now normalizes on write,
  // but pre-existing sidecars may still have duplicates and the keyed
  // `{#each}` below would otherwise collapse the row (Svelte requires
  // unique keys; identical tag text breaks the keyed iteration).
  let realPills = $derived.by(() => {
    const raw = splitSidecar(tagText ?? "")
      .danbooru.split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    const seen = new Set<string>();
    const out: string[] = [];
    for (const t of raw) {
      if (seen.has(t)) continue;
      seen.add(t);
      out.push(t);
    }
    return out;
  });

  // Tooltip text for the top-left "described" badge. Tags get loaded the
  // first time the thumb is hovered (which always happens before/while the
  // user reaches the small badge), so by the time the native tooltip kicks
  // in we usually have the real description ready. The fallback strings
  // cover the brief window before the fetch resolves and the rare case of
  // an empty 2nd line slipping past has_description.
  let badgeTitle = $derived.by(() => {
    if (!hasDescription) return "";
    if (tagText === null) return "Loading description…";
    const desc = splitSidecar(tagText).description;
    return desc || "Has description";
  });

  // realPills + (placeholder when adding). Drives the rendered list; the
  // placeholder isn't part of saved state until the user commits.
  let pills = $derived(addingTag ? [...realPills, PLACEHOLDER] : realPills);

  function replaceTag(oldTag: string, newTag: string) {
    const next = realPills.map((t) => (t === oldTag ? newTag : t)).join(", ");
    void saveTagsLine(next);
  }

  function deleteTag(oldTag: string) {
    const next = realPills.filter((t) => t !== oldTag).join(", ");
    void saveTagsLine(next);
  }

  function addTag(newTag: string) {
    addingTag = false;
    const trimmed = newTag.trim();
    if (!trimmed) return;
    // Always go through the save round-trip — even for an exact duplicate.
    // The server dedupes in join_sidecar and the response is what we write
    // back into tagText, so the pill list ends up identical to what was
    // already there. Skipping the save short-circuited too aggressively in
    // practice: when the user committed a duplicate, the placeholder was
    // unmounted without any write firing, and the next render cycle could
    // leave the keyed `each` out of sync with realPills. The HTTP roundtrip
    // is cheap; correctness wins.
    const next = [...realPills, trimmed].join(", ");
    void saveTagsLine(next);
  }

  function startAddingTag(ev: MouseEvent) {
    ev.stopPropagation();
    void loadTags();
    addingTag = true;
  }

  function openDescriptionModal(ev: MouseEvent) {
    ev.stopPropagation();
    descriptionModalOpen = true;
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

<!-- Wrapper div lets the toggle pill and the action pills live as real
     <button> siblings of the image button instead of nesting inside it
     (invalid HTML). -->
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

    {#if selected}
      <!-- Inset border overlay drawn on top of the image, with a faint tint
           so the selection state is unmissable on any image. -->
      <span class="absolute inset-0 rounded-lg border-[3px] border-accent-500 bg-accent-500/20 pointer-events-none"></span>
    {/if}
  </button>

  <!-- Top-LEFT "described" indicator: a small chat-bubble shown only when the
       .txt sidecar already has a non-empty 2nd line. Semitransparent so it
       reads as a soft glance-cue rather than competing with the image. We
       render it as a <button> so the native browser tooltip works on hover
       (it can't on a pointer-events-none span). Clicking it opens the
       description editor — same action as the bottom action-pill, but
       reachable directly from the badge for users who already see it. -->
  {#if hasDescription}
    <button
      type="button"
      onclick={openDescriptionModal}
      title={badgeTitle}
      aria-label="Edit description"
      class="absolute top-1.5 left-1.5 w-5 h-5 rounded-full bg-accent-500/40 hover:bg-accent-500/60 text-white/90 flex items-center justify-center z-20 cursor-help transition-colors {popping ? 'badge-pop' : ''}"
    >
      <svg viewBox="0 0 16 16" class="w-3 h-3" fill="currentColor" aria-hidden="true">
        <path d="M3 3h10a1.5 1.5 0 0 1 1.5 1.5v6A1.5 1.5 0 0 1 13 12H7.4l-3 2.4a.5.5 0 0 1-.8-.4V12H3a1.5 1.5 0 0 1-1.5-1.5v-6A1.5 1.5 0 0 1 3 3z"/>
      </svg>
    </button>
  {/if}

  <!-- Video stem badge, hover-only. Sits at the top-left, offset right of
       the description badge if present so the two don't collide. -->
  <span
    class="absolute top-1.5 px-1.5 py-0.5 text-[9px] bg-black/60 backdrop-blur-sm rounded text-white opacity-0 transition-opacity z-20 pointer-events-none {hovered ? 'opacity-100' : ''}"
    style="left: {hasDescription ? 28 : 6}px;"
  >
    {frame.video_stem}
  </span>

  <!-- Spinner overlay: covers the whole tile during a bulk re-tag /
       re-describe. Sits above every other badge (z-30) and absorbs pointer
       events so the image button underneath can't fire while busy. The dim
       backdrop is just enough to make the spinner read against any image. -->
  {#if processing}
    <div
      class="absolute inset-0 rounded-lg bg-ink-950/55 backdrop-blur-[1px] flex items-center justify-center z-30"
      role="status"
      aria-live="polite"
      aria-label="Processing frame"
    >
      <span
        class="block w-7 h-7 rounded-full border-2 border-white/25 border-t-accent-500 animate-spin"
        aria-hidden="true"
      ></span>
    </div>
  {/if}

  <!-- Top-RIGHT toggle pill: emerald + checkmark when selected, neutral
       otherwise. Always visible; on top of the image button via z-20. -->
  <button
    type="button"
    onclick={(e) => {
      e.stopPropagation();
      onselect({ shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey });
    }}
    title={selected ? "Deselect" : "Select"}
    aria-label={selected ? "Deselect frame" : "Select frame"}
    aria-pressed={selected}
    class="absolute top-1.5 right-1.5 w-6 h-6 rounded-full text-xs leading-none flex items-center justify-center z-20 transition-all border
      {selected
        ? 'bg-emerald-500 border-emerald-300 text-white shadow-[0_0_10px_rgba(16,185,129,0.6)] opacity-100'
        : 'bg-black/60 border-white/30 text-white/80 hover:bg-black/80 opacity-0 group-hover:opacity-100 focus:opacity-100'}"
  >{selected ? "✓" : "○"}</button>

  <!-- Bottom hover panel: tag pills + (+) pill + description pill, all
       inline. Sits above the image button (z-20) so clicks don't bubble
       into the preview-open handler. The TagPills' own e.stopPropagation
       handles the rest. -->
  {#if hovered}
    <div
      class="absolute inset-x-0 bottom-0 max-h-[60%] overflow-hidden p-1.5 pt-6 flex flex-wrap gap-1 items-center z-20
        bg-gradient-to-t from-black/85 via-black/70 to-transparent pointer-events-none"
    >
      <!-- pointer-events-none on the gradient + auto on each pill keeps the
           dim area between pills from intercepting clicks meant for the
           image preview. -->
      <div class="contents pointer-events-auto">
        {#each pills as p, i (p === PLACEHOLDER ? `__new__${i}` : p)}
          {#if p === PLACEHOLDER}
            <TagPill
              text=""
              startEditing
              onreplace={(next) => addTag(next)}
              oncancel={() => { addingTag = false; }}
            />
          {:else}
            <TagPill
              text={p}
              onreplace={(next) => replaceTag(p, next)}
              ondelete={() => deleteTag(p)}
            />
          {/if}
        {/each}

        <!-- "+" pill: same shape/size family as the tag pills so it reads
             as part of the row, not a floating action button. -->
        <button
          type="button"
          onclick={startAddingTag}
          title="Add tag"
          aria-label="Add tag"
          class="px-2 py-0.5 text-[9.5px] leading-none rounded-full bg-white/15 hover:bg-white/25 text-white border border-white/10 transition-colors"
        >+</button>

        <!-- "describe" pill: only shown when the frame has no description yet.
             Once a description exists, the top-left badge takes over editing
             so we don't render two buttons for the same action. -->
        {#if !hasDescription}
          <button
            type="button"
            onclick={openDescriptionModal}
            title="Add description"
            aria-label="Add description"
            class="px-2 py-0.5 text-[9.5px] leading-none rounded-full bg-white/15 hover:bg-white/25 text-white border border-white/10 transition-colors inline-flex items-center gap-1"
          >
            <svg viewBox="0 0 16 16" class="w-2.5 h-2.5" fill="currentColor" aria-hidden="true">
              <path d="M3 3h10a1.5 1.5 0 0 1 1.5 1.5v6A1.5 1.5 0 0 1 13 12H7.4l-3 2.4a.5.5 0 0 1-.8-.4V12H3a1.5 1.5 0 0 1-1.5-1.5v-6A1.5 1.5 0 0 1 3 3z"/>
            </svg>
            describe
          </button>
        {/if}
      </div>
    </div>
  {/if}
</div>

{#if descriptionModalOpen}
  <DescriptionModal
    filename={frame.filename}
    onclose={() => (descriptionModalOpen = false)}
    onsaved={async (text) => {
      hasDescriptionLocal = text.trim().length > 0;
      // Refetch the full sidecar from the server so the badge tooltip is
      // server-truth, regardless of whether the cache had been populated
      // before the save. Falls back to a local patch if the fetch fails so
      // we still surface the just-saved text rather than a stale value.
      const slug = projectsStore.active?.slug;
      if (slug) {
        try {
          const r = await api.getTags(slug, frame.filename);
          tagText = r.text;
          return;
        } catch {
          // fall through to the local patch
        }
      }
      if (tagText !== null) {
        const { danbooru } = splitSidecar(tagText);
        const trimmed = text.trim();
        tagText = trimmed ? `${danbooru}\n${trimmed}` : danbooru;
      }
    }}
  />
{/if}
