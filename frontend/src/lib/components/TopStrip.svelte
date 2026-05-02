<script lang="ts">
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import ActionBar from "./ActionBar.svelte";
  import CharacterStrip from "./CharacterStrip.svelte";
  import DensitySlider from "./DensitySlider.svelte";
  import ProjectPills from "./ProjectPills.svelte";
  import QueuePill from "./QueuePill.svelte";
  import ViewTabs from "./ViewTabs.svelte";

  type Props = { onopenRegex: () => void; onopenCreate: () => void };
  const { onopenRegex, onopenCreate }: Props = $props();

  // Character filter chips appear in the top bar (next to project pills)
  // when the user is on the Frames tab and the project has more than one
  // character. Single-character workflows render the top bar exactly as
  // before this change so the legacy UX is preserved.
  let showCharacterFilter = $derived(
    viewStore.tab === "frames"
      && (projectsStore.active?.characters.length ?? 0) > 1,
  );

  /** Pseudo-chips in front of the real characters: "All" (no filter) and
   *  "Unsorted" (server sentinel for orphan rows). */
  const leadingChips = [
    { key: "all", label: "All" },
    { key: "unsorted", label: "Unsorted" },
  ];

  function selectCharacterFilter(key: string) {
    viewStore.characterFilter = key;
  }
</script>

<!-- Top bar uses flex-wrap so when the window narrows the children
     wrap onto additional rows instead of overflowing. The min-h is
     dropped (no longer a strict single-row constraint), and a small
     row gap keeps stacked rows from butting against each other. -->
<header class="sticky top-0 z-30 px-4 py-3 bg-ink-950/95 backdrop-blur-sm border-b border-ink-700">
  <div class="flex flex-wrap items-center gap-x-3 gap-y-2 bg-ink-950 border border-ink-700 rounded-xl px-4 py-2.5 shadow-md">
    <!-- Left cluster: brand dot + project pills + (when applicable) the
         character filter strip. Wrapped in its own flex container so the
         filter strip rides next to ProjectPills and breaks together when
         the row gets tight. -->
    <div class="flex flex-wrap items-center gap-x-3 gap-y-2">
      <div class="w-5 h-5 rounded-full gradient-accent shadow-[0_0_12px_rgba(129,140,248,0.4)] flex-shrink-0"></div>
      <ProjectPills {onopenCreate} />
      {#if showCharacterFilter}
        <!-- Small left padding (pl-2) separates the filter chips from the
             project pills' "+" button. The strip itself wraps internally
             via flex-wrap so a project with many characters splits
             cleanly. -->
        <div class="pl-2">
          <CharacterStrip
            leadingChips={leadingChips}
            activeKey={viewStore.characterFilter}
            onselect={selectCharacterFilter}
          />
        </div>
      {/if}
    </div>

    <!-- Spacer pushes the right cluster (action bar + view tabs + density
         + queue) to the far edge on wide screens. On narrow screens it
         collapses to zero width and the right cluster wraps to the next
         row underneath the project/filter cluster. -->
    <div class="flex-1 min-w-0"></div>

    <!-- Right cluster: action bar + view tabs + density + queue pill.
         Allowed to wrap as a unit so they don't get truncated mid-control. -->
    <div class="flex flex-wrap items-center gap-x-3 gap-y-2 justify-end">
      <ActionBar {onopenRegex} />
      <ViewTabs />
      <DensitySlider />
      <QueuePill />
    </div>
  </div>
</header>
