<script lang="ts">
  import * as api from "$lib/api";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = { onclose: () => void };
  const { onclose }: Props = $props();

  let pattern = $state("");
  let replacement = $state("");
  let caseInsensitive = $state(false);
  let preview = $state<{ before: string; after: string }[]>([]);
  let previewError = $state<string | null>(null);
  let applying = $state(false);

  let filenames = $derived(framesStore.selectedFilenames());

  async function refreshPreview() {
    previewError = null;
    preview = [];
    if (!pattern) return;
    try {
      const flags = caseInsensitive ? "ig" : "g";
      // Use JS regex for the preview; the server uses Python's re, which is
      // very close but not identical. The server is authoritative on apply.
      const re = new RegExp(pattern, flags);
      const slug = projectsStore.active?.slug;
      if (!slug) return;
      const sample = filenames.slice(0, 20);
      const out: { before: string; after: string }[] = [];
      for (const fn of sample) {
        const t = await api.getTags(slug, fn);
        // The server applies the regex only to the danbooru (first) line.
        // Preview the same way so the user sees the actual diff.
        const firstLine = t.text.split("\n", 1)[0];
        const after = firstLine.replace(re, replacement);
        if (after !== firstLine) out.push({ before: firstLine, after });
      }
      preview = out;
    } catch (e) {
      previewError = String(e);
    }
  }

  $effect(() => {
    void refreshPreview();
    // depends on: pattern, replacement, caseInsensitive, filenames
    void pattern; void replacement; void caseInsensitive; void filenames;
  });

  async function apply() {
    const slug = projectsStore.active?.slug;
    if (!slug || !pattern) return;
    applying = true;
    try {
      await api.bulkTagsReplace(slug, {
        filenames, pattern, replacement, case_insensitive: caseInsensitive,
      });
      onclose();
    } catch (e) {
      previewError = String(e);
    } finally {
      applying = false;
    }
  }
</script>

<div class="fixed inset-0 bg-black/60 z-40 flex items-center justify-center" role="dialog" tabindex="-1" onclick={onclose} onkeydown={(e) => { if (e.key === 'Escape') onclose(); }}>
  <div
    class="bg-ink-900 border border-ink-700 rounded-xl p-5 max-w-xl w-full mx-4 shadow-2xl"
    role="document"
    onclick={(e) => e.stopPropagation()}
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

    <div class="mt-4 max-h-48 overflow-y-auto bg-ink-950 border border-ink-700 rounded p-2 text-xs font-mono">
      {#if previewError}
        <p class="text-red-400">{previewError}</p>
      {:else if preview.length === 0}
        <p class="text-slate-500">{pattern ? "No matches in selection." : "Type a pattern to preview."}</p>
      {:else}
        <p class="text-slate-500 mb-2">{preview.length} of first 20 will change:</p>
        {#each preview as p}
          <div class="mb-1.5">
            <p class="text-slate-500 truncate"><span class="text-red-400">−</span> {p.before}</p>
            <p class="text-emerald-400 truncate">+ {p.after}</p>
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
