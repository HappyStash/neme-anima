<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = {
    filename: string;
    onclose: () => void;
    /** Called after a successful save with the new description text so the
     *  parent (FrameThumb) can update its has_description flag without
     *  refetching the whole frames list. */
    onsaved?: (text: string) => void;
  };
  const { filename, onclose, onsaved }: Props = $props();

  let text = $state("");
  let loading = $state(true);
  let saving = $state(false);
  let error = $state<string | null>(null);

  $effect(() => {
    void loadDescription();
  });

  async function loadDescription() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    loading = true;
    error = null;
    try {
      const r = await api.getDescription(slug, filename);
      text = r.text;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  async function save() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    saving = true;
    error = null;
    try {
      const r = await api.putDescription(slug, filename, text);
      onsaved?.(r.text);
      onclose();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      saving = false;
    }
  }
</script>

<div
  class="fixed inset-0 bg-black/60 z-40 flex items-center justify-center"
  role="dialog"
  tabindex="-1"
  onclick={onclose}
  onkeydown={(e) => { if (e.key === 'Escape') onclose(); }}
>
  <div
    class="bg-ink-900 border border-ink-700 rounded-xl p-5 max-w-xl w-full mx-4 shadow-2xl"
    role="document"
    onclick={(e) => e.stopPropagation()}
    onkeydown={(e) => e.stopPropagation()}
  >
    <h2 class="text-lg font-semibold mb-1">Edit LLM description</h2>
    <p class="text-xs text-slate-500 mb-4 truncate" title={filename}>
      {filename}
    </p>

    {#if loading}
      <p class="text-slate-500 text-sm py-8 text-center">Loading…</p>
    {:else}
      <textarea
        bind:value={text}
        rows="6"
        placeholder="Describe the image — this becomes the second line of the .txt sidecar. Leave blank to remove the description."
        class="w-full px-3 py-2 bg-ink-950 border border-ink-700 rounded text-sm focus:outline-none focus:border-accent-500 resize-y"
      ></textarea>
      <p class="text-[11px] text-slate-600 mt-2 leading-snug">
        The danbooru tag line above this caption stays untouched. Saving an
        empty value reverts the file to a single-line sidecar.
      </p>
    {/if}

    {#if error}
      <p class="text-xs text-red-400 mt-2 break-all">{error}</p>
    {/if}

    <div class="flex justify-end gap-2 mt-5">
      <button
        type="button"
        onclick={onclose}
        class="px-4 py-2 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-300"
      >Cancel</button>
      <button
        type="button"
        onclick={save}
        disabled={loading || saving}
        class="px-4 py-2 text-xs rounded gradient-accent text-white disabled:opacity-40 disabled:cursor-not-allowed"
      >{saving ? "Saving…" : "Save"}</button>
    </div>
  </div>
</div>
