<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import { queueStore } from "$lib/stores/queue.svelte";
  import { connectEvents, type Connection } from "$lib/ws";
  import TopStrip from "$lib/components/TopStrip.svelte";
  import FramesTab from "$lib/components/FramesTab.svelte";

  let conn: Connection | null = null;

  onMount(async () => {
    await projectsStore.refresh();
    if (projectsStore.list.length > 0) {
      await projectsStore.load(projectsStore.list[0].slug);
    }
    await queueStore.refresh();
    conn = connectEvents({
      url: `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/ws`,
      onEvent: (ev) => queueStore.ingest(ev),
      onStatus: (s) => queueStore.setStatus(s),
    });
  });

  onDestroy(() => {
    conn?.close();
  });
</script>

<div class="min-h-screen bg-ink-950 text-slate-100">
  <TopStrip />
  <main class="px-4 pb-12">
    {#if projectsStore.active}
      {#if viewStore.tab === "sources"}
        <p class="text-slate-500 mt-8">Sources tab — Task 11.</p>
      {:else if viewStore.tab === "frames"}
        <FramesTab />
      {:else if viewStore.tab === "settings"}
        <p class="text-slate-500 mt-8">Settings tab — Task 12.</p>
      {/if}
    {:else}
      <div class="flex flex-col items-center justify-center py-32 text-slate-400">
        <p class="text-lg mb-2">No project selected.</p>
        <p class="text-sm">Click "+" in the top bar to create one.</p>
      </div>
    {/if}
  </main>
</div>
