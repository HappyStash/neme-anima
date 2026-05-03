<script lang="ts">
  import * as api from "$lib/api";
  import type { CharacterView } from "$lib/types";
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = {
    sourceSlug: string;
    character: CharacterView;
    onclose: () => void;
  };
  let { sourceSlug, character, onclose }: Props = $props();

  // Other registered projects, excluding the source.
  let candidates = $derived(
    (projectsStore.list ?? []).filter(p => p.slug !== sourceSlug && !p.missing),
  );
  let destinationSlug = $state<string>("");
  $effect(() => {
    if (!destinationSlug && candidates.length > 0) {
      destinationSlug = candidates[0].slug;
    }
  });

  let preview = $state<api.CharacterCopyReport | null>(null);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let confirmed = $state(false);

  async function runPreview() {
    if (!destinationSlug) return;
    busy = true; error = null;
    try {
      preview = await api.copyCharacterToProject(sourceSlug, character.slug, {
        destination_slug: destinationSlug,
        dry_run: true,
      });
    } catch (e) {
      error = (e as Error).message;
      preview = null;
    } finally {
      busy = false;
    }
  }

  async function runCopy() {
    if (!destinationSlug) return;
    busy = true; error = null;
    try {
      const result = await api.copyCharacterToProject(
        sourceSlug, character.slug,
        { destination_slug: destinationSlug, dry_run: false },
      );
      preview = result;
      confirmed = true;
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Escape") onclose();
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div
  role="dialog"
  aria-modal="true"
  aria-labelledby="copy-char-title"
  class="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
>
  <div class="bg-ink-900 border border-ink-700 rounded-xl p-6 w-[40rem] max-w-[92vw]">
    <h2 id="copy-char-title" class="text-lg font-medium text-slate-100 mb-1">
      Copy character "{character.name}" to another project
    </h2>
    <p class="text-xs text-slate-500 mb-4">
      Drops conflicts (videos, refs, frames already in the destination) on a
      per-object basis. Refusing the whole copy if a character with this slug
      already exists in the destination.
    </p>

    <label class="block mb-3 text-xs">
      <span class="text-[10px] uppercase tracking-wide text-slate-500">
        Destination project
      </span>
      <select
        bind:value={destinationSlug}
        disabled={candidates.length === 0 || busy}
        class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono text-slate-200"
      >
        {#each candidates as p (p.slug)}
          <option value={p.slug}>{p.name} ({p.slug})</option>
        {/each}
      </select>
      {#if candidates.length === 0}
        <span class="block text-[10px] text-amber-400 mt-1">
          No other registered projects available.
        </span>
      {/if}
    </label>

    {#if error}
      <div class="bg-red-900/30 border border-red-700 text-red-200 text-xs rounded p-2 mb-3">
        {error}
      </div>
    {/if}

    {#if preview}
      <div class="bg-ink-950 border border-ink-700 rounded p-3 mb-3 text-xs space-y-1 text-slate-300">
        <div>{confirmed ? "Copied" : "Will copy"}:
          {preview.sources_added.length} source(s),
          {preview.refs_added.length} ref(s),
          {preview.frames_added.length} frame(s),
          {preview.crops_copied} crop(s),
          {preview.custom_uploads_added} custom upload(s).
        </div>
        {#if preview.sources_skipped.length || preview.frames_skipped.length || Object.keys(preview.refs_renamed).length}
          <div class="text-amber-300">
            Skipped/renamed:
            {preview.sources_skipped.length} source(s) already in dst,
            {preview.frames_skipped.length} frame(s) collided,
            {Object.keys(preview.refs_renamed).length} ref(s) renamed.
          </div>
        {/if}
      </div>
    {/if}

    <div class="flex justify-end gap-2">
      <button
        type="button"
        onclick={onclose}
        class="px-3 py-1.5 text-xs rounded bg-ink-800 border border-ink-700 text-slate-300 hover:bg-ink-700"
      >{confirmed ? "Close" : "Cancel"}</button>
      {#if !confirmed}
        <button
          type="button"
          onclick={runPreview}
          disabled={busy || !destinationSlug}
          class="px-3 py-1.5 text-xs rounded bg-ink-800 border border-ink-700 text-slate-200 hover:bg-ink-700 disabled:opacity-40"
        >Preview</button>
        <button
          type="button"
          onclick={runCopy}
          disabled={busy || !destinationSlug || !preview}
          class="px-3 py-1.5 text-xs rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/30 disabled:opacity-40"
        >Confirm copy</button>
      {/if}
    </div>
  </div>
</div>
