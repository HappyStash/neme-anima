<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";

  // The threshold sections we expose. Pulled from src/neme_extractor/config.py.
  const SECTIONS: {
    key: string;
    label: string;
    fields: { name: string; type: "number" | "boolean"; placeholder: string }[];
  }[] = [
    { key: "scene", label: "Scene detection", fields: [
        { name: "threshold", type: "number", placeholder: "27.0" },
        { name: "min_scene_len_frames", type: "number", placeholder: "8" },
      ]},
    { key: "detect", label: "Detection", fields: [
        { name: "person_score_min", type: "number", placeholder: "0.35" },
        { name: "frame_stride", type: "number", placeholder: "4" },
        { name: "detect_faces", type: "boolean", placeholder: "false" },
      ]},
    { key: "track", label: "Tracking", fields: [
        { name: "track_thresh", type: "number", placeholder: "0.25" },
        { name: "match_thresh", type: "number", placeholder: "0.8" },
        { name: "min_tracklet_len", type: "number", placeholder: "3" },
      ]},
    { key: "identify", label: "Identification", fields: [
        { name: "body_max_distance_strict", type: "number", placeholder: "0.15" },
        { name: "body_max_distance_loose", type: "number", placeholder: "0.20" },
        { name: "sample_frames_per_tracklet", type: "number", placeholder: "5" },
      ]},
    { key: "frame_select", label: "Frame selection", fields: [
        { name: "candidate_cap", type: "number", placeholder: "20" },
        { name: "dedup_min_frame_gap", type: "number", placeholder: "4" },
        { name: "top_k_short", type: "number", placeholder: "1" },
        { name: "top_k_long", type: "number", placeholder: "3" },
      ]},
    { key: "crop", label: "Crop", fields: [
        { name: "longest_side", type: "number", placeholder: "1024" },
        { name: "pad_ratio", type: "number", placeholder: "0.10" },
      ]},
    { key: "tag", label: "Tagging", fields: [
        { name: "general_threshold", type: "number", placeholder: "0.35" },
        { name: "character_threshold", type: "number", placeholder: "0.85" },
      ]},
  ];

  let overrides = $state<Record<string, Record<string, unknown>>>(
    projectsStore.active?.thresholds_overrides ?? {},
  );

  $effect(() => {
    overrides = { ...(projectsStore.active?.thresholds_overrides ?? {}) };
  });

  function getValue(section: string, field: string): string {
    const v = overrides[section]?.[field];
    return v === undefined || v === null ? "" : String(v);
  }

  function setValue(section: string, field: string, raw: string, type: "number" | "boolean") {
    if (raw === "") {
      if (overrides[section]) delete overrides[section][field];
      if (overrides[section] && Object.keys(overrides[section]).length === 0) delete overrides[section];
      overrides = { ...overrides };
      return;
    }
    const value = type === "boolean" ? raw.toLowerCase() === "true" : Number(raw);
    overrides = {
      ...overrides,
      [section]: { ...(overrides[section] ?? {}), [field]: value },
    };
  }

  let saving = $state(false);
  let savedAt = $state<number | null>(null);

  let pauseBeforeTag = $state<boolean>(
    projectsStore.active?.pause_before_tag ?? true,
  );
  $effect(() => {
    pauseBeforeTag = projectsStore.active?.pause_before_tag ?? true;
  });

  async function save() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    saving = true;
    try {
      await api.patchProject(slug, {
        thresholds_overrides: overrides,
        pause_before_tag: pauseBeforeTag,
      });
      await projectsStore.load(slug);
      savedAt = Date.now();
    } finally {
      saving = false;
    }
  }

  function resetSection(section: string) {
    if (overrides[section]) {
      delete overrides[section];
      overrides = { ...overrides };
    }
  }

  function resetAll() {
    overrides = {};
  }
</script>

<div class="mt-4 max-w-3xl mx-auto">
  <div class="flex items-center justify-between mb-4">
    <h2 class="text-base font-semibold text-slate-200">Per-project settings</h2>
    <div class="flex gap-2 items-center">
      {#if savedAt}
        <span class="text-xs text-emerald-400">saved</span>
      {/if}
      <button type="button" onclick={resetAll} class="text-xs text-slate-500 hover:text-slate-300">Reset thresholds</button>
      <button
        type="button"
        onclick={save}
        disabled={saving}
        class="px-4 py-1.5 text-xs rounded gradient-accent text-white disabled:opacity-50"
      >{saving ? "Saving…" : "Save"}</button>
    </div>
  </div>

  <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
    <h3 class="text-sm font-medium text-slate-200 mb-3">Workflow</h3>
    <label class="flex items-start gap-3 cursor-pointer">
      <input
        type="checkbox"
        bind:checked={pauseBeforeTag}
        class="mt-0.5 w-4 h-4 rounded bg-ink-950 border-ink-700 accent-accent-500"
      />
      <span class="flex-1">
        <span class="block text-sm text-slate-200">Pause before tagging</span>
        <span class="block text-xs text-slate-500 mt-0.5">
          When on, the pipeline waits after writing kept frames so you can
          delete unwanted ones before they're tagged. Click the yellow
          ⏸ pill on a running pipeline to resume tagging. Off = the
          pipeline tags inline as it runs.
        </span>
      </span>
    </label>
  </div>

  {#each SECTIONS as section (section.key)}
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-medium text-slate-200">{section.label}</h3>
        <button type="button" onclick={() => resetSection(section.key)} class="text-xs text-slate-500 hover:text-slate-300">Reset</button>
      </div>
      <div class="grid grid-cols-2 gap-3">
        {#each section.fields as f}
          <label class="block">
            <span class="text-[10px] uppercase tracking-wide text-slate-500">{f.name}</span>
            <input
              value={getValue(section.key, f.name)}
              oninput={(e) => setValue(section.key, f.name, (e.target as HTMLInputElement).value, f.type)}
              placeholder={f.placeholder}
              class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-sm font-mono focus:outline-none focus:border-accent-500"
            />
          </label>
        {/each}
      </div>
    </div>
  {/each}
</div>
