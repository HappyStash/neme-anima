<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";

  type Props = {
    /** When true, renders the "+ Add" button + per-chip × delete affordance.
     *  False = read-only chip strip (used in the Frames tab where adding a
     *  character belongs on the Sources tab). */
    editable?: boolean;
    /** Optional extra leading chips. The Frames tab feeds this with "All"
     *  and "unsorted"; the Sources tab leaves it empty. Each item is
     *  ``{key, label, selected}`` — ``key`` becomes the active value. */
    leadingChips?: { key: string; label: string }[];
    /** Currently-selected key. For the Sources tab this is the active
     *  character slug; for the Frames tab it's "all" / "unsorted" / a slug. */
    activeKey: string;
    onselect: (key: string) => void;
  };
  const {
    editable = false,
    leadingChips = [],
    activeKey,
    onselect,
  }: Props = $props();

  let creating = $state(false);
  let renaming = $state<string | null>(null);
  let nameDraft = $state("");

  let characters = $derived(projectsStore.active?.characters ?? []);

  async function startCreate() {
    const name = prompt("Character name (e.g. Yui, Mio, Ritsu)?");
    if (!name || !name.trim()) return;
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    creating = true;
    try {
      const created = await api.createCharacter(slug, { name: name.trim() });
      // Refresh project view so the new character appears in the strip and
      // any other surface that reads project.characters (Settings, etc).
      await projectsStore.load(slug);
      onselect(created.slug);
    } catch (e) {
      alert(`Failed to create character: ${e}`);
    } finally {
      creating = false;
    }
  }

  async function deleteCharacter(slug: string, name: string) {
    if (!confirm(
      `Delete character "${name}"?\n\n` +
      "This drops their reference images and their per-video opt-outs. " +
      "Frames already routed to them stay on disk but become 'unsorted'."
    )) return;
    const projectSlug = projectsStore.active?.slug;
    if (!projectSlug) return;
    try {
      await api.deleteCharacter(projectSlug, slug);
      await projectsStore.load(projectSlug);
      // If the user just nuked the active character, fall back to whichever
      // character the project ended up with after the delete.
      if (activeKey === slug) {
        const next = projectsStore.active?.characters[0]?.slug ?? "default";
        onselect(next);
      }
    } catch (e) {
      // Server returns 409 when this would be the last character.
      const msg = e instanceof Error ? e.message : String(e);
      alert(msg.includes("409")
        ? "Can't delete the last character — every project needs at least one."
        : `Failed to delete: ${msg}`);
    }
  }

  async function commitRename(slug: string) {
    const projectSlug = projectsStore.active?.slug;
    if (!projectSlug || !nameDraft.trim()) {
      renaming = null;
      return;
    }
    try {
      await api.updateCharacter(projectSlug, slug, { name: nameDraft.trim() });
      await projectsStore.load(projectSlug);
    } catch (e) {
      alert(`Rename failed: ${e}`);
    } finally {
      renaming = null;
      nameDraft = "";
    }
  }

  function startRename(slug: string, currentName: string) {
    renaming = slug;
    nameDraft = currentName;
  }
</script>

<div class="flex flex-wrap items-center gap-1.5">
  {#each leadingChips as chip (chip.key)}
    <button
      type="button"
      onclick={() => onselect(chip.key)}
      class="h-7 px-3 rounded-full text-xs inline-flex items-center transition-colors
        {activeKey === chip.key
          ? 'gradient-accent text-white shadow-[0_2px_8px_rgba(99,102,241,0.3)]'
          : 'bg-ink-900 border border-ink-700 text-slate-300 hover:bg-ink-800 hover:text-slate-100'}"
    >{chip.label}</button>
  {/each}

  {#each characters as c (c.slug)}
    {@const active = activeKey === c.slug}
    <div
      class="h-7 rounded-full text-xs inline-flex items-center gap-1.5 transition-colors
        {active
          ? 'gradient-accent text-white shadow-[0_2px_8px_rgba(99,102,241,0.3)] pr-1'
          : 'bg-ink-900 border border-ink-700 text-slate-300 hover:bg-ink-800 hover:text-slate-100 pr-1'}"
    >
      {#if renaming === c.slug && editable}
        <input
          type="text"
          bind:value={nameDraft}
          onkeydown={(e) => {
            if (e.key === "Enter") commitRename(c.slug);
            else if (e.key === "Escape") { renaming = null; nameDraft = ""; }
          }}
          onblur={() => commitRename(c.slug)}
          class="bg-transparent border-0 outline-none text-xs px-3 w-32"
          aria-label="Rename character"
        />
      {:else}
        <button
          type="button"
          onclick={() => onselect(c.slug)}
          ondblclick={() => editable && startRename(c.slug, c.name)}
          class="pl-3 inline-flex items-center gap-1.5 h-full"
          title={editable ? "Click to select · double-click to rename" : "Click to select"}
        >
          <span>{c.name}</span>
          <span class="opacity-70 tabular-nums">({c.ref_count})</span>
        </button>
      {/if}
      {#if editable}
        <button
          type="button"
          onclick={() => deleteCharacter(c.slug, c.name)}
          aria-label="Delete {c.name}"
          title="Delete this character"
          class="h-5 w-5 inline-flex items-center justify-center rounded-full
            {active
              ? 'opacity-70 hover:opacity-100 text-white'
              : 'text-slate-500 hover:text-red-400'}"
        >×</button>
      {/if}
    </div>
  {/each}

  {#if editable}
    <button
      type="button"
      onclick={startCreate}
      disabled={creating}
      class="h-7 px-3 rounded-full text-xs inline-flex items-center gap-1
        bg-ink-900 border border-dashed border-ink-700 text-slate-400
        hover:border-accent-500 hover:text-slate-100 transition-colors
        disabled:opacity-40"
    >+ Add character</button>
  {/if}
</div>
