<script lang="ts">
  import * as api from "$lib/api";
  import { framesStore } from "$lib/stores/frames.svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";

  type Props = { onopenRegex: () => void };
  const { onopenRegex }: Props = $props();

  let count = $derived.by(() => {
    framesStore.selectionVersion; // reactive dependency
    return framesStore.selection.count();
  });

  let onFramesTab = $derived(viewStore.tab === "frames");
  let total = $derived(framesStore.items.length);
  let allSelected = $derived(count > 0 && count === total);

  // The LLM-retag button only renders when the project has a model picked —
  // matches Settings logic so a user can't fire a doomed retag against an
  // unconfigured endpoint.
  let llmModelSelected = $derived(!!projectsStore.active?.llm?.model);

  let retagBusy = $state(false);

  async function deleteSelected() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    if (!confirm(`Delete ${filenames.length} frame${filenames.length === 1 ? "" : "s"}?`)) return;
    await api.bulkDeleteFrames(slug, filenames);
    framesStore.removeLocal(filenames);
  }

  async function retagDanbooru() {
    const slug = projectsStore.active?.slug;
    if (!slug || retagBusy) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    retagBusy = true;
    try {
      const res = await api.bulkRetagDanbooru(slug, filenames);
      // Refresh so the on-hover tag preview reads the new WD14 output rather
      // than serving the now-stale cached text from each FrameThumb.
      await framesStore.refresh(
        slug,
        viewStore.sourceFilter ? { source: viewStore.sourceFilter } : {},
      );
      alert(`Re-tagged ${res.retagged} of ${res.total} frame${res.total === 1 ? "" : "s"}.`);
    } catch (e) {
      alert(`Re-tag failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      retagBusy = false;
    }
  }

  async function retagLLM() {
    const slug = projectsStore.active?.slug;
    if (!slug || retagBusy) return;
    const filenames = framesStore.selectedFilenames();
    if (filenames.length === 0) return;
    retagBusy = true;
    // Process one frame at a time so the user gets per-frame feedback (badge
    // pop) as descriptions are written, and so each LLM call uses a fresh
    // chat-completions context (no carry-over between images). Errors
    // accumulate but don't abort the rest of the batch — a stuck endpoint
    // shouldn't black-hole an N-frame queue silently, but we surface a
    // single summary alert at the end if anything actually failed.
    let described = 0;
    let lastError: string | null = null;
    try {
      for (const filename of filenames) {
        try {
          const res = await api.bulkRetagLLM(slug, [filename]);
          if (res.described > 0) {
            // The backend silently retargets to a `_crop` derivative when
            // one exists, so pop the badge on the row that actually got
            // written rather than the one the user clicked.
            const eff = res.effective_filenames?.[0] ?? filename;
            framesStore.markDescribed(eff);
            described += 1;
          } else if (res.error) {
            lastError = res.error;
          }
        } catch (e) {
          lastError = e instanceof Error ? e.message : String(e);
        }
      }
    } finally {
      retagBusy = false;
    }
    // Only alert when something went wrong — the success path is
    // communicated visually by the per-frame badge pop animation.
    if (described < filenames.length) {
      const failed = filenames.length - described;
      const detail = lastError ? ` Last error: ${lastError}` : "";
      alert(`Described ${described} of ${filenames.length}; ${failed} failed.${detail}`);
    }
  }

  function toggleSelectAll() {
    if (allSelected) framesStore.clear();
    else framesStore.selectAll();
  }
</script>

<!-- Order, left → right: selected-pill (visible when count > 0), Select all,
     N frames. The purple pill leads so the destructive cluster sits at the
     visual edge of the row, with the static select/count on its right. -->
{#if onFramesTab}
  <div class="flex items-center gap-2 h-7">
    {#if count > 0}
      <div class="gradient-accent text-white h-7 pl-3 pr-1 rounded-full inline-flex items-center gap-1.5 text-xs font-medium border border-white/10 shadow-[0_2px_12px_rgba(99,102,241,0.4)]">
        <span class="bg-white/20 px-2 py-0.5 rounded-full leading-none">{count} selected</span>
        <button
          type="button"
          onclick={deleteSelected}
          class="bg-white/15 hover:bg-red-500/60 rounded-full px-2.5 h-5 transition-colors inline-flex items-center"
        >Delete</button>
        <button
          type="button"
          onclick={onopenRegex}
          class="bg-white/15 hover:bg-white/25 rounded-full px-2.5 h-5 transition-colors inline-flex items-center"
        >Regex…</button>
        <button
          type="button"
          onclick={retagDanbooru}
          disabled={retagBusy}
          title="Re-run WD14 tagger on selected frames (preserves the LLM description line)"
          class="bg-white/15 hover:bg-white/25 rounded-full px-2.5 h-5 transition-colors inline-flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
        >Re-tag</button>
        {#if llmModelSelected}
          <button
            type="button"
            onclick={retagLLM}
            disabled={retagBusy}
            title="Re-run LLM description on selected frames (preserves WD14 tags)"
            class="bg-white/15 hover:bg-white/25 rounded-full px-2.5 h-5 transition-colors inline-flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
          >Describe</button>
        {/if}
        <button
          type="button"
          onclick={() => framesStore.clear()}
          class="opacity-70 hover:opacity-100 h-5 w-5 inline-flex items-center justify-center"
          title="Clear selection"
        >✕</button>
      </div>
    {/if}

    <button
      type="button"
      onclick={toggleSelectAll}
      disabled={total === 0}
      class="h-7 px-3 rounded-full text-xs bg-ink-900 border border-ink-700 text-slate-300 hover:bg-ink-800 hover:text-slate-100 disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center"
    >{allSelected ? "Deselect all" : "Select all"}</button>

    <span
      class="h-7 px-3 rounded-full text-xs bg-ink-900 border border-ink-700 text-slate-400 inline-flex items-center"
      title="Total frames in the current view"
    >{total} frame{total === 1 ? "" : "s"}</span>
  </div>
{/if}
