<script lang="ts">
  import ProjectPills from "./ProjectPills.svelte";
  import ViewTabs from "./ViewTabs.svelte";
  import QueuePill from "./QueuePill.svelte";
  import DensitySlider from "./DensitySlider.svelte";
  import ActionBar from "./ActionBar.svelte";

  type Props = { onopenRegex: () => void; onopenCreate: () => void };
  const { onopenRegex, onopenCreate }: Props = $props();
</script>

<header class="sticky top-0 z-30 px-4 py-3 bg-ink-950/95 backdrop-blur-sm border-b border-ink-700">
  <div class="flex items-center gap-3 bg-ink-950 border border-ink-700 rounded-xl px-4 py-2.5 shadow-md min-h-[3rem]">
    <div class="w-5 h-5 rounded-full gradient-accent shadow-[0_0_12px_rgba(129,140,248,0.4)] flex-shrink-0"></div>
    <ProjectPills {onopenCreate} />
    <div class="flex-1"></div>
    <ActionBar {onopenRegex} />
    <ViewTabs />
    <DensitySlider />
    <QueuePill />
  </div>
</header>
