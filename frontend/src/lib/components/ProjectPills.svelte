<script lang="ts">
  import { projectsStore } from "$lib/stores/projects.svelte";

  function isActive(slug: string): boolean {
    return projectsStore.active?.slug === slug;
  }

  async function selectProject(slug: string) {
    await projectsStore.load(slug);
  }

  // CreateProjectModal opens via this flag — Task 13 wires the modal itself.
  let showCreate = $state(false);
</script>

<div class="flex items-center gap-1">
  {#each projectsStore.list as p (p.slug)}
    <button
      class="px-3 py-1 text-xs rounded-full transition-all border border-transparent
        {isActive(p.slug)
          ? 'gradient-accent text-white border-white/10 shadow-[0_2px_8px_rgba(99,102,241,0.35)]'
          : 'bg-ink-800 text-slate-400 hover:bg-ink-700 hover:text-slate-200'}"
      onclick={() => selectProject(p.slug)}
      title={p.folder}
    >
      {p.name}
    </button>
  {/each}
  <button
    class="w-6 py-1 text-xs rounded-full border border-dashed border-ink-600 text-slate-500 hover:border-accent-500 hover:text-accent-400"
    onclick={() => (showCreate = true)}
    title="New project"
  >
    +
  </button>
</div>
