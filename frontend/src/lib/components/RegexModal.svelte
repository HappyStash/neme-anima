<script lang="ts">
  import * as api from "$lib/api";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = { onclose: () => void };
  const { onclose }: Props = $props();

  let pattern = $state("");
  let replacement = $state("");
  let caseInsensitive = $state(false);
  // Each preview row carries the segment-by-segment split for both the
  // before and after lines. `match: true` segments are highlighted; the
  // rest read as plain context. The wrapper renders these inline with
  // CSS wrap so long tag lines are fully visible without truncation.
  type Segment = { text: string; match: boolean };
  let preview = $state<{ before: Segment[]; after: Segment[] }[]>([]);
  let previewError = $state<string | null>(null);
  let applying = $state(false);

  let filenames = $derived(framesStore.selectedFilenames());

  /** Walk a single line, splitting it into plain/match segments and
   *  computing the corresponding after-segments by applying the regex's
   *  replacement (with $1 etc. resolved) to each matched substring.
   *  Returns null when the regex didn't match anywhere — that row is
   *  skipped entirely so the preview only shows lines that change. */
  function diffLine(text: string, re: RegExp, repl: string):
    { before: Segment[]; after: Segment[] } | null
  {
    // A non-global twin so `replace` resolves `$1`/`$&` etc. on a single
    // match without iterating again. Cheap to construct per call; the
    // sample size is capped at 20.
    const single = new RegExp(re.source, re.flags.replace("g", ""));
    const before: Segment[] = [];
    const after: Segment[] = [];
    let lastIndex = 0;
    re.lastIndex = 0;
    let matched = false;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      matched = true;
      if (m.index > lastIndex) {
        const plain = text.slice(lastIndex, m.index);
        before.push({ text: plain, match: false });
        after.push({ text: plain, match: false });
      }
      const replaced = m[0].replace(single, repl);
      before.push({ text: m[0], match: true });
      after.push({ text: replaced, match: true });
      lastIndex = m.index + m[0].length;
      // Zero-width matches advance lastIndex manually so we don't loop
      // forever (e.g. pattern `^` or `(?=)`).
      if (m[0].length === 0) re.lastIndex++;
    }
    if (!matched) return null;
    if (lastIndex < text.length) {
      const tail = text.slice(lastIndex);
      before.push({ text: tail, match: false });
      after.push({ text: tail, match: false });
    }
    return { before, after };
  }

  // Debounce + race protection. Clearing `preview = []` on every keystroke
  // would have collapsed the preview pane and re-grown it as results came
  // back, making the modal flash and shift as the user typed. Instead we
  // wait a short window after the last keystroke, then swap the new
  // preview in atomically. `previewGen` discards results from any
  // in-flight refresh that's been superseded by a newer one.
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  let previewGen = 0;

  async function refreshPreview() {
    const myGen = ++previewGen;
    if (!pattern) {
      preview = [];
      previewError = null;
      return;
    }
    try {
      const flags = caseInsensitive ? "ig" : "g";
      // Use JS regex for the preview; the server uses Python's re, which is
      // very close but not identical. The server is authoritative on apply.
      const re = new RegExp(pattern, flags);
      const slug = projectsStore.active?.slug;
      if (!slug) return;
      const sample = filenames.slice(0, 20);
      const out: { before: Segment[]; after: Segment[] }[] = [];
      for (const fn of sample) {
        const t = await api.getTags(slug, fn);
        if (myGen !== previewGen) return;
        const firstLine = t.text.split("\n", 1)[0];
        const diff = diffLine(firstLine, re, replacement);
        if (diff) out.push(diff);
      }
      if (myGen !== previewGen) return;
      preview = out;
      previewError = null;
    } catch (e) {
      if (myGen !== previewGen) return;
      previewError = String(e);
      preview = [];
    }
  }

  $effect(() => {
    // depends on: pattern, replacement, caseInsensitive, filenames
    void pattern; void replacement; void caseInsensitive; void filenames;
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { void refreshPreview(); }, 200);
    return () => {
      if (debounceTimer) {
        clearTimeout(debounceTimer);
        debounceTimer = null;
      }
    };
  });

  async function apply() {
    const slug = projectsStore.active?.slug;
    if (!slug || !pattern) return;
    applying = true;
    // Snapshot the selection before the call: we deselect on success and
    // we want to bust the thumb tag-cache for the same exact set even if
    // `filenames` (a derived) shifts mid-apply.
    const targets = [...filenames];
    try {
      await api.bulkTagsReplace(slug, {
        filenames: targets,
        pattern,
        replacement,
        case_insensitive: caseInsensitive,
      });
      // Bump each frame's retag counter so FrameThumb drops its cached
      // tagText — the FrameRecord doesn't change for a regex replace,
      // only the on-disk .txt does, so without this the old tags keep
      // showing on hover. Cheap to bump for unchanged frames too: the
      // next hover just refetches the same line.
      for (const fn of targets) framesStore.markRetagged(fn);
      framesStore.deselect(targets);
      onclose();
    } catch (e) {
      previewError = String(e);
    } finally {
      applying = false;
    }
  }
</script>

<div
  class="fixed inset-0 bg-black/60 z-40 flex items-center justify-center"
  role="dialog"
  tabindex="-1"
  onmousedown={(e) => { if (e.target === e.currentTarget) onclose(); }}
  onkeydown={(e) => { if (e.key === 'Escape') onclose(); }}
>
  <div
    class="bg-ink-900 border border-ink-700 rounded-xl p-5 max-w-xl w-full mx-4 shadow-2xl"
    role="document"
    onkeydown={(e) => e.stopPropagation()}
  >
    <h2 class="text-lg font-semibold mb-1">Bulk regex tag replace</h2>
    <p class="text-xs text-slate-500 mb-2">{filenames.length} frame{filenames.length === 1 ? "" : "s"} selected · operates on the danbooru tag line only (LLM description preserved)</p>
    <p class="text-[11px] text-slate-600 mb-4 leading-snug">
      Tip: pattern <code class="text-slate-400">^</code> with replacement <code class="text-slate-400">new_tag,&nbsp;</code> prepends a tag.
      Pattern <code class="text-slate-400">$</code> with replacement <code class="text-slate-400">,&nbsp;new_tag</code> appends one.
    </p>

    <div class="space-y-3">
      <label class="block">
        <span class="text-[10px] uppercase text-slate-500 tracking-wide">Pattern</span>
        <input
          bind:value={pattern}
          class="w-full mt-1 px-3 py-2 bg-ink-950 border border-ink-700 rounded text-sm font-mono focus:outline-none focus:border-accent-500"
          placeholder="red[ _]eyes"
        />
      </label>

      <label class="block">
        <span class="text-[10px] uppercase text-slate-500 tracking-wide">Replacement</span>
        <input
          bind:value={replacement}
          class="w-full mt-1 px-3 py-2 bg-ink-950 border border-ink-700 rounded text-sm font-mono focus:outline-none focus:border-accent-500"
          placeholder="red eyes"
        />
      </label>

      <label class="flex items-center gap-2 text-xs text-slate-300">
        <input type="checkbox" bind:checked={caseInsensitive} class="accent-accent-500" />
        Case-insensitive
      </label>
    </div>

    <!-- Fixed height (h-48) instead of max-h: the pane never grows or
         shrinks as the preview content changes, so the modal stops
         shifting and flashing while the user is typing. -->
    <div class="mt-4 h-48 overflow-y-auto bg-ink-950 border border-ink-700 rounded p-2 text-xs font-mono">
      {#if previewError}
        <p class="text-red-400">{previewError}</p>
      {:else if preview.length === 0}
        <p class="text-slate-500">{pattern ? "No matches in selection." : "Type a pattern to preview."}</p>
      {:else}
        <p class="text-slate-500 mb-2">{preview.length} of first 20 will change:</p>
        {#each preview as p}
          <!-- whitespace-pre-wrap + break-words: long tag lines wrap onto
               multiple lines instead of getting truncated, so the changed
               segment is always visible. The surrounding plain context
               renders dim while the matched/replacement segments get a
               solid background — quick eye-catch for "what changed". -->
          <div class="mb-2 leading-snug">
            <p class="text-slate-500 whitespace-pre-wrap break-words">
              <span class="text-red-400 select-none">−&nbsp;</span>
              {#each p.before as seg}
                {#if seg.match}
                  <span class="bg-red-500/30 text-red-200 rounded px-0.5">{seg.text}</span>
                {:else}{seg.text}{/if}
              {/each}
            </p>
            <p class="text-slate-500 whitespace-pre-wrap break-words">
              <span class="text-emerald-400 select-none">+&nbsp;</span>
              {#each p.after as seg}
                {#if seg.match}
                  <span class="bg-emerald-500/25 text-emerald-200 rounded px-0.5">{seg.text}</span>
                {:else}{seg.text}{/if}
              {/each}
            </p>
          </div>
        {/each}
      {/if}
    </div>

    <div class="flex justify-end gap-2 mt-5">
      <button
        type="button"
        onclick={onclose}
        class="px-4 py-2 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-300"
      >Cancel</button>
      <button
        type="button"
        onclick={apply}
        disabled={!pattern || preview.length === 0 || applying}
        class="px-4 py-2 text-xs rounded gradient-accent text-white disabled:opacity-40 disabled:cursor-not-allowed"
      >{applying ? "Applying…" : `Apply to ${filenames.length}`}</button>
    </div>
  </div>
</div>
