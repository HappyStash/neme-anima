<script lang="ts">
  import { viewStore } from "$lib/stores/view.svelte";
</script>

<label class="flex items-center gap-2 text-[10px] uppercase tracking-wide text-slate-500" title="Grid density (3-12 columns)">
  <span>thumbs</span>
  <input
    type="range" min="3" max="12" step="1"
    value={viewStore.density}
    oninput={(e) => viewStore.setDensity(Number((e.target as HTMLInputElement).value))}
    class="w-16 accent-accent-500"
  />
</label>
