<script lang="ts">
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { viewStore } from "$lib/stores/view.svelte";
  import CharacterStrip from "./CharacterStrip.svelte";
  import VideoRow from "./VideoRow.svelte";

  let importing = $state(false);
  let uploading = $state(false);
  let imageDragOver = $state(false);
  let fileInput: HTMLInputElement | undefined = $state();

  // Active character drives what refs appear in the project-level grid AND
  // what each per-video ref strip renders. Sourced from viewStore so the
  // selection survives navigating away and back to this tab.
  let activeCharacter = $derived(projectsStore.activeCharacter);
  let activeRefs = $derived(activeCharacter?.refs ?? []);

  async function pickVideoFolder() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const current = projectsStore.active?.source_root ?? "";
    const folder = prompt(
      "Paste an absolute folder path. All video files in it will be added (.mkv, .mp4, .webm, …).",
      current,
    );
    if (!folder) return;
    importing = true;
    try {
      await api.importSourcesFolder(slug, folder.trim());
      await projectsStore.load(slug);
    } finally {
      importing = false;
    }
  }

  async function reimport() {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    importing = true;
    try {
      await api.reimportSources(slug);
      await projectsStore.load(slug);
    } finally {
      importing = false;
    }
  }

  async function uploadImageFiles(files: File[]) {
    const slug = projectsStore.active?.slug;
    if (!slug || files.length === 0) return;
    const images = files.filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name));
    if (images.length === 0) {
      alert("No image files in the drop.");
      return;
    }
    uploading = true;
    try {
      // Refs are character-scoped: drops land on the currently-selected
      // character so the user's mental model ("I'm uploading Mio refs
      // because Mio is selected") matches reality. Falling back to the
      // first character covers projects that haven't migrated yet.
      const target =
        viewStore.activeCharacterSlug ||
        projectsStore.active?.characters[0]?.slug;
      await api.uploadRefs(slug, images, target);
      await projectsStore.load(slug);
    } finally {
      uploading = false;
    }
  }

  function handleImageDrop(ev: DragEvent) {
    ev.preventDefault();
    imageDragOver = false;
    const files = Array.from(ev.dataTransfer?.files ?? []);
    void uploadImageFiles(files);
  }

  function handleImagePick(ev: Event) {
    const input = ev.currentTarget as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    void uploadImageFiles(files);
    input.value = "";
  }

  async function removeRef(path: string) {
    const slug = projectsStore.active?.slug;
    if (!slug) return;
    const name = path.split("/").pop() ?? path;
    if (!confirm(`Remove reference image “${name}” from this project? The file will be deleted.`)) return;
    await api.removeRef(slug, path);
    await projectsStore.load(slug);
  }

  function selectCharacter(slug: string) {
    viewStore.activeCharacterSlug = slug;
  }

  let sourceRootShort = $derived.by(() => {
    const r = projectsStore.active?.source_root;
    if (!r) return "";
    const parts = r.split("/").filter(Boolean);
    return parts.slice(-2).join("/");
  });
</script>

<div class="mt-4">
  <div class="grid grid-cols-2 gap-3 mb-4">
    <!-- Videos: folder picker + reimport in one zone -->
    <div class="bg-ink-900 border-2 border-dashed border-ink-700 rounded-xl px-4 py-4 flex items-center gap-3.5 hover:border-accent-500 transition-all">
      <button
        type="button"
        onclick={pickVideoFolder}
        disabled={importing || !projectsStore.active}
        class="flex items-center gap-3.5 flex-1 text-left disabled:opacity-50"
      >
        <div class="w-9 h-9 rounded-lg gradient-accent flex items-center justify-center text-white text-lg shadow-[0_2px_12px_rgba(99,102,241,0.3)] flex-shrink-0">
          ▶
        </div>
        <div class="flex-1 min-w-0">
          <p class="text-slate-200 text-sm font-medium">{importing ? "Scanning folder…" : "Add videos from folder…"}</p>
          <p class="text-slate-500 text-xs mt-0.5 truncate">
            {#if projectsStore.active?.source_root}
              Last: <span class="font-mono">{sourceRootShort}</span>
            {:else}
              .mkv · .mp4 · .webm — files stay in place
            {/if}
          </p>
        </div>
      </button>
      {#if projectsStore.active?.source_root}
        <button
          type="button"
          onclick={reimport}
          disabled={importing}
          title="Re-scan {projectsStore.active.source_root} for missing videos"
          class="px-3 py-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-300 border border-ink-700 disabled:opacity-50 flex-shrink-0"
        >Reimport</button>
      {/if}
    </div>

    <!-- Images: drop or click to pick; bytes are uploaded into the project -->
    <button
      type="button"
      ondragover={(e) => { e.preventDefault(); imageDragOver = true; }}
      ondragleave={() => (imageDragOver = false)}
      ondrop={handleImageDrop}
      onclick={() => fileInput?.click()}
      disabled={uploading || !projectsStore.active}
      class="w-full text-left bg-ink-900 border-2 border-dashed rounded-xl px-4 py-4 transition-all flex items-center gap-3.5 disabled:opacity-50
        {imageDragOver ? 'border-accent-500 bg-ink-800 shadow-[0_0_24px_rgba(99,102,241,0.15)]' : 'border-ink-700 hover:border-accent-500'}"
    >
      <div class="w-9 h-9 rounded-lg gradient-accent flex items-center justify-center text-white text-lg shadow-[0_2px_12px_rgba(99,102,241,0.3)] flex-shrink-0">
        ◇
      </div>
      <div class="flex-1 min-w-0">
        <p class="text-slate-200 text-sm font-medium">{uploading ? "Uploading…" : "Drop reference images here"}</p>
        <p class="text-slate-500 text-xs mt-0.5">Files are copied into the project; click to pick.</p>
      </div>
    </button>
    <input
      bind:this={fileInput}
      type="file"
      accept="image/*"
      multiple
      class="hidden"
      onchange={handleImagePick}
    />
  </div>

  {#if projectsStore.active}
    <p class="text-[10px] uppercase text-slate-500 tracking-wide mb-2">
      characters — refs and per-video opt-outs are scoped to the active one
    </p>
    <div class="mb-4">
      <CharacterStrip
        editable
        activeKey={viewStore.activeCharacterSlug}
        onselect={selectCharacter}
      />
    </div>

    {#if activeRefs.length > 0 && activeCharacter}
      <p class="text-[10px] uppercase text-slate-500 tracking-wide mb-2">
        {activeCharacter.name}'s references — apply to every video by default
      </p>
      <div class="bg-ink-950 border border-ink-700 rounded-xl px-3 py-2.5 flex items-center gap-2 mb-4">
        <span class="text-[10px] uppercase text-slate-600 tracking-wide w-24">Refs ({activeRefs.length})</span>
        <div class="flex gap-1.5 flex-1 flex-wrap">
          {#each activeRefs as r (r.path)}
            <div class="relative group w-9 h-9">
              <img
                src={api.refImageUrl(projectsStore.active.slug, r.path)}
                alt={r.path.split('/').pop() ?? ''}
                title={r.path}
                loading="lazy"
                class="w-9 h-9 rounded object-cover bg-ink-800 border border-ink-700"
              />
              <button
                type="button"
                onclick={() => removeRef(r.path)}
                title="Remove reference"
                aria-label="Remove reference {r.path.split('/').pop()}"
                class="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-ink-900 border border-ink-700 text-slate-500 hover:text-red-400 hover:border-red-400/60 text-[9px] leading-none flex items-center justify-center opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
              >✕</button>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <p class="text-[10px] uppercase text-slate-500 tracking-wide mb-2">videos in this project</p>
    {#if projectsStore.active.sources.length === 0}
      <p class="text-slate-500 text-sm py-8 text-center">No videos yet. Pick a folder above.</p>
    {:else}
      {#each projectsStore.active.sources as s, i (s.path)}
        <VideoRow source={s} sourceIdx={i} />
      {/each}
    {/if}
  {/if}
</div>
