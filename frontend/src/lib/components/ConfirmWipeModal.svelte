<script lang="ts">
  import type { WipePreview } from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = {
    /** The "what would be wiped" snapshot fetched just before the user
     *  triggers the action. */
    preview: WipePreview;
    /** "Extract" or "Re-process" — purely a label for the dialog so the
     *  user can read what they're about to fire. */
    action: "Extract" | "Re-process";
    onconfirm: () => void;
    oncancel: () => void;
  };
  const { preview, action, onconfirm, oncancel }: Props = $props();

  /** Prefer the character's display name over the raw slug — the slug is
   *  filesystem-safe but reads poorly in user-visible text. Falls back to
   *  the slug when no project is loaded or the slug isn't in the
   *  project's character list (e.g. orphan rows from a renamed/deleted
   *  character — surfaces in the "Unsorted" filter elsewhere in the UI). */
  function nameFor(slug: string): string {
    if (slug === "__untracked__") return "untracked";
    const c = projectsStore.active?.characters.find((c) => c.slug === slug);
    return c?.name ?? slug;
  }

  let wipeRows = $derived(
    Object.entries(preview.to_wipe.by_character)
      .filter(([, n]) => n > 0)
      .sort(([a], [b]) => a.localeCompare(b)),
  );
  let preserveRows = $derived(
    Object.entries(preview.to_preserve.by_character)
      .filter(([, n]) => n > 0)
      .sort(([a], [b]) => a.localeCompare(b)),
  );
</script>

<!-- Backdrop click → cancel; Escape too. The modal floats above the
     pipeline strip via z-50. -->
<div
  class="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/70 backdrop-blur-sm"
  role="presentation"
  onclick={oncancel}
  onkeydown={(e) => { if (e.key === "Escape") oncancel(); }}
>
  <div
    role="dialog"
    aria-modal="true"
    aria-labelledby="confirm-wipe-title"
    class="bg-ink-900 border border-ink-700 rounded-xl shadow-2xl p-5 max-w-md w-full mx-4"
    onclick={(e) => e.stopPropagation()}
    onkeydown={(e) => e.stopPropagation()}
  >
    <h2 id="confirm-wipe-title" class="text-base font-semibold text-slate-100 mb-1">
      {action} will replace existing frames
    </h2>
    <p class="text-xs text-slate-500 mb-4">
      Frames belonging to characters with no active references will be preserved.
    </p>

    <div class="space-y-3 text-sm">
      <!-- "About to wipe" section. Per-character counts plus the
           rejected-samples total (always wiped because they're
           diagnostic, not curated training data). -->
      <div>
        <p class="text-xs uppercase tracking-wide text-amber-400 mb-1">
          Will be replaced ({preview.to_wipe.total})
        </p>
        {#if wipeRows.length === 0 && preview.to_wipe.rejected_samples === 0}
          <p class="text-slate-400 text-sm">Nothing — fresh dataset.</p>
        {:else}
          <ul class="text-slate-300 text-sm pl-4">
            {#each wipeRows as [slug, n] (slug)}
              <li class="list-disc">
                <span class="text-slate-100">{n}</span>
                {n === 1 ? "frame" : "frames"} from
                <span class="text-slate-200">{nameFor(slug)}</span>
              </li>
            {/each}
            {#if preview.to_wipe.rejected_samples > 0}
              <li class="list-disc text-slate-500">
                <span class="text-slate-400">{preview.to_wipe.rejected_samples}</span>
                rejected sample{preview.to_wipe.rejected_samples === 1 ? "" : "s"}
                <span class="text-slate-600">(diagnostic)</span>
              </li>
            {/if}
          </ul>
        {/if}
      </div>

      {#if preserveRows.length > 0}
        <div>
          <p class="text-xs uppercase tracking-wide text-emerald-400 mb-1">
            Will be preserved ({preview.to_preserve.total})
          </p>
          <ul class="text-slate-300 text-sm pl-4">
            {#each preserveRows as [slug, n] (slug)}
              <li class="list-disc">
                <span class="text-slate-100">{n}</span>
                {n === 1 ? "frame" : "frames"} from
                <span class="text-slate-200">{nameFor(slug)}</span>
              </li>
            {/each}
          </ul>
          <p class="text-[11px] text-slate-500 mt-1.5">
            These characters have no active refs for this video. The
            new run will not produce frames at filenames that already
            belong to them.
          </p>
        </div>
      {/if}
    </div>

    <div class="flex justify-end gap-2 mt-5">
      <button
        type="button"
        onclick={oncancel}
        class="px-3 py-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-300 border border-ink-700"
      >Cancel</button>
      <button
        type="button"
        onclick={onconfirm}
        class="px-3 py-1.5 text-xs rounded gradient-accent text-white shadow-[0_2px_8px_rgba(99,102,241,0.3)]"
      >{action} anyway</button>
    </div>
  </div>
</div>
