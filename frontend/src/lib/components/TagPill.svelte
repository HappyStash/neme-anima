<script lang="ts">
  type Props = {
    text: string;
    onreplace: (next: string) => void;
    /** Called when an existing pill is committed empty — parent should drop
     *  the tag entirely. If omitted, empty commits revert to the old text. */
    ondelete?: () => void;
    /** Called when a *placeholder* pill (created by the "+" button) is
     *  dismissed without typing — parent should remove the temporary slot. */
    oncancel?: () => void;
    /** Mount the pill directly in edit mode (used for the "+" placeholder). */
    startEditing?: boolean;
  };
  const {
    text,
    onreplace,
    ondelete,
    oncancel,
    startEditing = false,
  }: Props = $props();

  let editing = $state<boolean>(startEditing);
  let value = $state(text);
  let inputEl = $state<HTMLInputElement | null>(null);

  $effect(() => {
    // Sync the buffer to the canonical text whenever it changes from the
    // outside (e.g. after a save). Only do this when not actively editing
    // so we don't yank the user's in-progress edit out from under them.
    if (!editing) value = text;
  });

  // When we transition into edit mode, auto-select the existing text so the
  // user can immediately retype to replace, or hit Backspace + Enter to
  // delete the tag. For the placeholder pill (text === ""), there's nothing
  // to select — the cursor just sits at the empty input.
  $effect(() => {
    if (!editing) return;
    const el = inputEl;
    if (!el) return;
    queueMicrotask(() => {
      el.focus();
      if (el.value.length > 0) el.select();
    });
  });

  function commit(ev: KeyboardEvent | FocusEvent) {
    if (ev instanceof KeyboardEvent && ev.key === "Escape") {
      ev.preventDefault();
      // Escape on a placeholder = discard. Escape on an existing pill =
      // revert and exit edit mode without saving.
      if (text === "" && oncancel) {
        oncancel();
        return;
      }
      value = text;
      editing = false;
      return;
    }
    if (ev instanceof KeyboardEvent && ev.key !== "Enter") return;
    if (ev instanceof KeyboardEvent) ev.preventDefault();

    const trimmed = value.trim();
    editing = false;

    if (trimmed === "") {
      // Empty existing pill = delete. Empty placeholder = discard.
      if (text === "") {
        oncancel?.();
      } else {
        ondelete?.();
      }
      return;
    }
    if (trimmed !== text) onreplace(trimmed);
  }

  // Heuristic: a tag containing "character" — server doesn't currently expose
  // a flag for character vs general tags in the listFrames payload, so we
  // treat anything that LOOKS like a character name as "important". This is
  // intentionally simple for v1; server-driven labels are a Phase 2C polish.
  let isCharacter = $derived(text.toLowerCase().includes("character"));
</script>

{#if editing}
  <input
    bind:this={inputEl}
    bind:value
    onkeydown={commit}
    onblur={commit}
    onclick={(e) => e.stopPropagation()}
    placeholder={text === "" ? "new tag…" : ""}
    class="px-2 py-0.5 text-[9.5px] rounded-full bg-accent-500 text-white shadow-[0_0_0_1.5px_rgba(199,210,254,1),0_0_12px_rgba(99,102,241,0.6)] outline-none w-24 placeholder-white/60"
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
