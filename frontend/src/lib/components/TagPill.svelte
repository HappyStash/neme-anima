<script lang="ts">
  type Props = {
    text: string;
    onreplace: (next: string) => void;
  };
  const { text, onreplace }: Props = $props();

  let editing = $state(false);
  let value = $state(text);

  $effect(() => {
    value = text;
  });

  function commit(ev: KeyboardEvent | FocusEvent) {
    if (ev instanceof KeyboardEvent && ev.key === "Escape") {
      value = text;
      editing = false;
      return;
    }
    if (ev instanceof KeyboardEvent && ev.key !== "Enter") return;
    if (ev instanceof KeyboardEvent) ev.preventDefault();
    editing = false;
    if (value.trim() && value !== text) onreplace(value.trim());
  }

  // Heuristic: a tag containing "character" — server doesn't currently expose
  // a flag for character vs general tags in the listFrames payload, so we
  // treat anything that LOOKS like a character name as "important". This is
  // intentionally simple for v1; server-driven labels are a Phase 2C polish.
  let isCharacter = $derived(text.toLowerCase().includes("character"));
</script>

{#if editing}
  <input
    bind:value
    onkeydown={commit}
    onblur={commit}
    autofocus
    onclick={(e) => e.stopPropagation()}
    class="px-2 py-0.5 text-[9.5px] rounded-full bg-accent-500 text-white shadow-[0_0_0_1.5px_rgba(199,210,254,1),0_0_12px_rgba(99,102,241,0.6)] outline-none w-24"
  />
{:else}
  <button
    type="button"
    onclick={(e) => { e.stopPropagation(); editing = true; }}
    class="px-1.5 py-0.5 text-[9.5px] rounded-full backdrop-blur-sm border border-white/5 transition-colors
      {isCharacter ? 'bg-amber2-500/85 text-amber-950 font-medium' : 'bg-white/15 text-white hover:bg-white/25'}"
  >
    {text}
  </button>
{/if}
