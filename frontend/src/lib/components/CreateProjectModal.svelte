<script lang="ts">
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = { onclose: () => void };
  const { onclose }: Props = $props();

  let name = $state("");
  let folder = $state("");
  let creating = $state(false);
  let error = $state<string | null>(null);
  let folderTouched = $state(false);

  $effect(() => {
    // Default the folder to ~/neme-projects/<slug> until the user types one.
    if (name && !folderTouched) {
      const slug = name.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "");
      folder = `~/neme-projects/${slug}`;
    }
  });

  async function create() {
    if (!name || !folder) return;
    creating = true;
    error = null;
    try {
      await projectsStore.create(name, folder);
      onclose();
    } catch (e) {
      error = String(e);
    } finally {
      creating = false;
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
    class="bg-ink-900 border border-ink-700 rounded-xl p-5 max-w-md w-full mx-4 shadow-2xl"
    role="document"
    onkeydown={(e) => e.stopPropagation()}
  >
    <h2 class="text-lg font-semibold mb-4">New project</h2>
    <div class="space-y-3">
      <label class="block">
        <span class="text-[10px] uppercase text-slate-500 tracking-wide">Name</span>
        <input
          bind:value={name}
          autofocus
          class="w-full mt-1 px-3 py-2 bg-ink-950 border border-ink-700 rounded text-sm focus:outline-none focus:border-accent-500"
          placeholder="megumin"
        />
      </label>
      <label class="block">
        <span class="text-[10px] uppercase text-slate-500 tracking-wide">Folder</span>
        <input
          bind:value={folder}
          oninput={() => (folderTouched = true)}
          class="w-full mt-1 px-3 py-2 bg-ink-950 border border-ink-700 rounded text-sm font-mono focus:outline-none focus:border-accent-500"
          placeholder="~/neme-projects/megumin"
        />
      </label>
    </div>
    {#if error}<p class="text-red-400 text-xs mt-3">{error}</p>{/if}
    <div class="flex justify-end gap-2 mt-5">
      <button type="button" onclick={onclose} class="px-4 py-2 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-300">Cancel</button>
      <button
        type="button"
        onclick={create}
        disabled={!name || !folder || creating}
        class="px-4 py-2 text-xs rounded gradient-accent text-white disabled:opacity-40 disabled:cursor-not-allowed"
      >{creating ? "Creating…" : "Create"}</button>
    </div>
  </div>
</div>
