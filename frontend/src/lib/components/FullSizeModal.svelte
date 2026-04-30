<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";

  type Props = {
    filename: string;
    onclose: () => void;
  };
  const { filename, onclose }: Props = $props();

  let imageUrl = $derived(
    projectsStore.active
      ? api.frameImageUrl(projectsStore.active.slug, filename)
      : "",
  );

  function onKey(ev: KeyboardEvent) {
    if (ev.key === "Escape") onclose();
  }
</script>

<svelte:window onkeydown={onKey} />

<!-- Click anywhere — backdrop or image — closes the modal, per spec. -->
<button
  type="button"
  class="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-6 cursor-zoom-out"
  onclick={onclose}
  aria-label="Close fullsize preview"
>
  <img
    src={imageUrl}
    alt={filename}
    class="max-w-full max-h-full object-contain rounded shadow-2xl"
  />
</button>
