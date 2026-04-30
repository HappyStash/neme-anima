<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { trainingStore } from "$lib/stores/training.svelte";
  import type { TrainingConfig, TrainingPathCheck } from "$lib/types";

  // Style-vs-character preset overrides. The reference recipe targets
  // styles; the notes flag that character LoRAs benefit from a higher LR.
  const PRESETS: Record<string, Partial<TrainingConfig>> = {
    style: {
      learning_rate: 2e-5,
      gradient_accumulation_steps: 4,
      epochs: 40,
    },
    character: {
      learning_rate: 5e-5,
      gradient_accumulation_steps: 2,
      epochs: 60,
    },
  };

  type PathField = "diffusion_pipe_dir" | "dit_path" | "vae_path" | "llm_path";
  const PATH_FIELDS: { key: PathField; label: string; expect: "dir" | "file"; placeholder: string; hint: string }[] = [
    {
      key: "diffusion_pipe_dir",
      label: "diffusion-pipe directory",
      expect: "dir",
      placeholder: "/home/you/code/diffusion-pipe",
      hint: "Folder containing train.py (tdrussell/diffusion-pipe or compatible fork).",
    },
    {
      key: "dit_path",
      label: "Anima DiT (transformer) file",
      expect: "file",
      placeholder: "/data/models/anima-preview3-base.safetensors",
      hint: "anima-preview3-base.safetensors (or a successor checkpoint).",
    },
    {
      key: "vae_path",
      label: "Qwen image VAE",
      expect: "file",
      placeholder: "/data/models/qwen_image_vae.safetensors",
      hint: "qwen_image_vae.safetensors.",
    },
    {
      key: "llm_path",
      label: "Qwen 3 0.6B base text encoder",
      expect: "file",
      placeholder: "/data/models/qwen_3_06b_base.safetensors",
      hint: "qwen_3_06b_base.safetensors.",
    },
  ];

  const RESOLUTION_PRESETS: { key: string; label: string; values: number[] }[] = [
    { key: "512", label: "512", values: [512] },
    { key: "512+1024", label: "512 + 1024 (recommended)", values: [512, 1024] },
    { key: "512+1024+1536", label: "512 + 1024 + 1536 (max detail)", values: [512, 1024, 1536] },
    { key: "1024", label: "1024 only", values: [1024] },
  ];

  let pollHandle: ReturnType<typeof setInterval> | null = null;

  // ---- per-path live debounce -----------------------------------------
  // We debounce the check-path POST so the user doesn't fire an HTTP
  // request on every keystroke. The check-path endpoint is cheap, but
  // 60 round-trips while pasting a long path is still wasteful.
  const debounceTimers: Partial<Record<PathField, ReturnType<typeof setTimeout>>> = {};
  let pathChecks = $state<Record<PathField, TrainingPathCheck | null>>({
    diffusion_pipe_dir: null, dit_path: null, vae_path: null, llm_path: null,
  });

  // Local edit buffers for path inputs so the user can type freely; we
  // only commit (PATCH) on blur or Save.
  let pathDraft = $state<Record<PathField, string>>({
    diffusion_pipe_dir: "", dit_path: "", vae_path: "", llm_path: "",
  });
  let pathDirty = $state<Record<PathField, boolean>>({
    diffusion_pipe_dir: false, dit_path: false, vae_path: false, llm_path: false,
  });

  // Trigger the initial load + start polling whenever the active project
  // changes. Use a derived value to drive the effect so tab switches don't
  // double-fetch.
  let activeSlug = $derived(projectsStore.active?.slug ?? null);

  $effect(() => {
    const slug = activeSlug;
    if (!slug) return;
    trainingStore.setProject(slug).then(() => {
      const cfg = trainingStore.configResp?.config;
      if (cfg) {
        pathDraft = {
          diffusion_pipe_dir: cfg.diffusion_pipe_dir,
          dit_path: cfg.dit_path,
          vae_path: cfg.vae_path,
          llm_path: cfg.llm_path,
        };
        pathDirty = { diffusion_pipe_dir: false, dit_path: false, vae_path: false, llm_path: false };
      }
      const pc = trainingStore.configResp?.path_checks;
      if (pc) pathChecks = { ...pathChecks, ...pc };
    });
  });

  onMount(() => {
    // Light polling backstops the WebSocket — covers the case where the
    // user opens the tab after a run already started and missed events.
    pollHandle = setInterval(async () => {
      if (!trainingStore.slug) return;
      try {
        trainingStore.status = await api.getTrainingStatus(trainingStore.slug);
      } catch {
        // ignore — poll resumes next tick
      }
    }, 5000);
  });

  onDestroy(() => {
    if (pollHandle) clearInterval(pollHandle);
    for (const t of Object.values(debounceTimers)) {
      if (t) clearTimeout(t);
    }
  });

  function schedulePathCheck(field: PathField, value: string) {
    pathDraft = { ...pathDraft, [field]: value };
    pathDirty = { ...pathDirty, [field]: true };
    if (debounceTimers[field]) clearTimeout(debounceTimers[field]!);
    debounceTimers[field] = setTimeout(async () => {
      try {
        const res = await trainingStore.checkPath(field, value);
        pathChecks = { ...pathChecks, [field]: res };
      } catch (e) {
        pathChecks = {
          ...pathChecks,
          [field]: { path: value, exists: false, is_file: false, is_dir: false, error: String(e) },
        };
      }
    }, 300);
  }

  async function commitPath(field: PathField) {
    if (!pathDirty[field]) return;
    await trainingStore.patch({ [field]: pathDraft[field] } as Partial<TrainingConfig>);
    pathDirty = { ...pathDirty, [field]: false };
    const pc = trainingStore.configResp?.path_checks;
    if (pc) pathChecks = { ...pathChecks, ...pc };
  }

  async function patchField<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) {
    await trainingStore.patch({ [key]: value } as Partial<TrainingConfig>);
  }

  function applyPreset(preset: string) {
    const overrides = PRESETS[preset];
    if (!overrides) return;
    trainingStore.patch({ ...overrides, preset } as Partial<TrainingConfig>);
  }

  function pickResolutionPreset(values: number[]) {
    trainingStore.patch({ resolutions: values });
  }

  function fmtBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
  }

  let cfg = $derived(trainingStore.configResp?.config ?? null);
  let problems = $derived(trainingStore.configResp?.problems ?? []);
  let status = $derived(trainingStore.status);
  let runs = $derived(trainingStore.runs);
  let preview = $derived(trainingStore.preview);
  let log = $derived(trainingStore.log);
  let runState = $derived(status?.state ?? null);
  let isRunning = $derived(!!status?.running);
  let canStart = $derived(!isRunning && problems.length === 0 && !!cfg);
  let canResume = $derived(
    !isRunning &&
    runs.length > 0 &&
    !!runs[0]?.latest_checkpoint &&
    problems.length === 0,
  );

  let busy = $state(false);
  async function doStart() {
    busy = true;
    try { await trainingStore.start(); }
    catch (e) { alert(`Start failed: ${e}`); }
    finally { busy = false; }
  }
  async function doStop() {
    busy = true;
    try { await trainingStore.stop(); }
    catch (e) { alert(`Stop failed: ${e}`); }
    finally { busy = false; }
  }
  async function doResume() {
    busy = true;
    try { await trainingStore.resume(); }
    catch (e) { alert(`Resume failed: ${e}`); }
    finally { busy = false; }
  }

  // ---- run/checkpoint browser -----------------------------------------
  let expandedRun = $state<string | null>(null);
  let runCheckpoints = $state<Record<string, Awaited<ReturnType<typeof api.listTrainingCheckpoints>>>>({});
  async function toggleRun(name: string) {
    if (expandedRun === name) { expandedRun = null; return; }
    expandedRun = name;
    if (!runCheckpoints[name] && trainingStore.slug) {
      runCheckpoints[name] = await api.listTrainingCheckpoints(trainingStore.slug, name);
    }
  }
  async function refreshExpandedRun() {
    if (expandedRun && trainingStore.slug) {
      runCheckpoints[expandedRun] = await api.listTrainingCheckpoints(trainingStore.slug, expandedRun);
    }
  }
  async function deleteCheckpoint(runName: string, ckptName: string) {
    if (!confirm(`Delete checkpoint ${ckptName}?`)) return;
    await trainingStore.deleteCheckpoint(runName, ckptName);
    await refreshExpandedRun();
  }
  async function deleteRun(runName: string) {
    if (!confirm(`Delete the entire run ${runName} and all its checkpoints?`)) return;
    await trainingStore.deleteRun(runName);
    if (expandedRun === runName) expandedRun = null;
  }
  async function resumeFromCheckpoint(runName: string, ckptName: string) {
    busy = true;
    try {
      await trainingStore.resume({ run_dir_name: runName, resume_from_checkpoint: ckptName });
    } catch (e) {
      alert(`Resume failed: ${e}`);
    } finally {
      busy = false;
    }
  }

  // Auto-scroll the log panel.
  let logPanel: HTMLDivElement | null = $state(null);
  $effect(() => {
    void log.length;
    if (logPanel) logPanel.scrollTop = logPanel.scrollHeight;
  });

  function pathBadge(check: TrainingPathCheck | null): { ok: boolean; text: string } {
    if (!check) return { ok: false, text: "not checked" };
    if (check.error) return { ok: false, text: check.error };
    return { ok: true, text: check.is_dir ? "directory" : "file" };
  }
</script>

<div class="mt-4 max-w-4xl mx-auto">
  <div class="flex items-center justify-between mb-4">
    <div>
      <h2 class="text-base font-semibold text-slate-200">LoRA training</h2>
      <p class="text-xs text-slate-500 mt-0.5">
        Train an Anima LoRA on this project's kept frames using
        <code class="text-slate-400">tdrussell/diffusion-pipe</code>.
        See <code class="text-slate-400">docs/anima-lora-training-notes.md</code>
        for the recipe these defaults follow.
      </p>
    </div>

    <!-- top-right control row -->
    <div class="flex items-center gap-2">
      {#if isRunning}
        <button
          type="button"
          onclick={doStop}
          disabled={busy}
          class="px-4 py-1.5 text-xs rounded bg-red-700 hover:bg-red-600 text-white disabled:opacity-50"
        >Stop</button>
      {:else}
        <button
          type="button"
          onclick={doResume}
          disabled={!canResume || busy}
          class="px-3 py-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-100 border border-ink-700 disabled:opacity-40 disabled:cursor-not-allowed"
          title={canResume ? "Resume from the most recent checkpoint" : "No prior checkpoint to resume from"}
        >Resume from last</button>
        <button
          type="button"
          onclick={doStart}
          disabled={!canStart || busy}
          class="px-4 py-1.5 text-xs rounded gradient-accent text-white disabled:opacity-40 disabled:cursor-not-allowed"
          title={canStart ? "Start a new training run" : (problems[0] ?? "Provide all model paths first")}
        >{busy ? "Starting…" : "Start training"}</button>
      {/if}
    </div>
  </div>

  {#if !cfg}
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 text-sm text-slate-400">
      Loading training config…
    </div>
  {:else}

    <!-- ============ run status banner ============ -->
    {#if runState}
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-medium text-slate-200">Current run</h3>
          <span class="text-xs px-2 py-0.5 rounded
            {runState.status === 'running' ? 'bg-emerald-900/50 text-emerald-300' :
             runState.status === 'starting' || runState.status === 'stopping' ? 'bg-amber-900/50 text-amber-300' :
             runState.status === 'failed' ? 'bg-red-900/50 text-red-300' :
             'bg-slate-800 text-slate-400'}">
            {runState.status}
          </span>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div>
            <span class="text-slate-500 uppercase tracking-wide text-[10px]">run</span>
            <div class="font-mono text-slate-300 truncate">{runState.run_name}</div>
          </div>
          <div>
            <span class="text-slate-500 uppercase tracking-wide text-[10px]">epoch · step</span>
            <div class="font-mono text-slate-300">
              {runState.epoch ?? "—"} · {runState.step ?? "—"}
            </div>
          </div>
          <div>
            <span class="text-slate-500 uppercase tracking-wide text-[10px]">loss</span>
            <div class="font-mono text-slate-300">{runState.loss?.toFixed(4) ?? "—"}</div>
          </div>
          <div>
            <span class="text-slate-500 uppercase tracking-wide text-[10px]">started</span>
            <div class="font-mono text-slate-300">{runState.started_at ? new Date(runState.started_at).toLocaleString() : "—"}</div>
          </div>
        </div>
        {#if runState.resumed_from}
          <div class="mt-2 text-xs text-slate-500">
            Resumed from <span class="font-mono text-slate-300">{runState.resumed_from}</span>.
          </div>
        {/if}
        {#if runState.error}
          <div class="mt-2 text-xs text-red-400 break-words">{runState.error}</div>
        {/if}
        {#if runState.last_log_line}
          <div class="mt-2 text-[11px] font-mono text-slate-500 truncate">{runState.last_log_line}</div>
        {/if}
      </div>
    {/if}

    {#if problems.length > 0}
      <div class="bg-amber-900/30 border border-amber-800 rounded-xl p-3 mb-3 text-xs text-amber-200">
        <div class="font-medium mb-1">Cannot start a run yet:</div>
        <ul class="list-disc list-inside space-y-0.5">
          {#each problems as p}
            <li>{p}</li>
          {/each}
        </ul>
      </div>
    {/if}

    <!-- ============ paths ============ -->
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <h3 class="text-sm font-medium text-slate-200 mb-1">Trainer paths</h3>
      <p class="text-xs text-slate-500 mb-3">
        Point at locally-installed assets so we don't have to download the
        Anima weights for you. Each path is checked as you type — green
        means the file or directory exists.
      </p>
      <div class="grid grid-cols-1 gap-3">
        {#each PATH_FIELDS as f (f.key)}
          {@const check = pathChecks[f.key]}
          {@const badge = pathBadge(check)}
          <label class="block">
            <div class="flex items-center justify-between mb-1">
              <span class="text-[10px] uppercase tracking-wide text-slate-500">{f.label}</span>
              <span class="text-[10px] {badge.ok ? 'text-emerald-400' : 'text-red-400'}">
                {badge.ok ? "✓" : "✗"} {badge.text}
              </span>
            </div>
            <div class="flex gap-2">
              <input
                value={pathDraft[f.key]}
                oninput={(e) => schedulePathCheck(f.key, (e.target as HTMLInputElement).value)}
                onblur={() => commitPath(f.key)}
                placeholder={f.placeholder}
                class="flex-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-xs font-mono focus:outline-none focus:border-accent-500"
              />
            </div>
            <span class="block text-[10px] text-slate-600 mt-1">{f.hint}</span>
          </label>
        {/each}
      </div>

      <details class="mt-4">
        <summary class="text-xs text-slate-400 cursor-pointer hover:text-slate-200">
          Advanced: launcher command override
        </summary>
        <p class="text-[11px] text-slate-500 mt-2 mb-1">
          Default: <code class="text-slate-400">deepspeed --num_gpus=1 train.py --deepspeed --config &lcub;config&rcub;</code>.
          Override only if you need a custom launcher or wrapper script. Use <code class="text-slate-400">&lcub;config&rcub;</code> as a placeholder for the run TOML path.
        </p>
        <input
          value={cfg.launcher_override}
          onchange={(e) => patchField("launcher_override", (e.target as HTMLInputElement).value)}
          placeholder="(empty = built-in default)"
          class="w-full px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-xs font-mono focus:outline-none focus:border-accent-500"
        />
      </details>
    </div>

    <!-- ============ preset & dataset ============ -->
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-medium text-slate-200">Preset &amp; dataset</h3>
        <div class="flex gap-1">
          {#each ["style", "character"] as p}
            <button
              type="button"
              onclick={() => applyPreset(p)}
              class="px-3 py-1 text-xs rounded {cfg.preset === p ? 'bg-accent-700 text-white' : 'bg-ink-800 text-slate-300 hover:bg-ink-700'}"
            >{p}</button>
          {/each}
        </div>
      </div>
      <p class="text-[11px] text-slate-500 mb-3">
        <strong>style</strong> = recipe defaults (lr 2e-5, grad-accum 4, 40 epochs).
        <strong>character</strong> bumps lr to 5e-5 and epochs to 60 — community
        reports indicate 2e-5 is too low for character identity.
      </p>

      {#if preview}
        <div class="grid grid-cols-3 gap-3 mb-3 text-xs">
          <div class="bg-ink-950 rounded p-2">
            <div class="text-[10px] text-slate-500 uppercase">total images</div>
            <div class="font-mono text-slate-200">{preview.total_images}</div>
          </div>
          <div class="bg-ink-950 rounded p-2">
            <div class="text-[10px] text-slate-500 uppercase">with tags</div>
            <div class="font-mono text-slate-200">{preview.with_tags}</div>
          </div>
          <div class="bg-ink-950 rounded p-2">
            <div class="text-[10px] text-slate-500 uppercase">with NL desc.</div>
            <div class="font-mono text-slate-200">{preview.with_descriptions}</div>
          </div>
        </div>

        {#if preview.samples.length > 0}
          <details>
            <summary class="text-xs text-slate-400 cursor-pointer hover:text-slate-200">
              Caption preview (first {preview.samples.length})
            </summary>
            <ul class="mt-2 space-y-2 text-[11px] font-mono text-slate-400">
              {#each preview.samples as s}
                <li class="bg-ink-950 rounded p-2">
                  <div class="text-slate-300 truncate">{s.filename}</div>
                  <div class="text-slate-500 mt-1 break-words">{s.rendered}</div>
                </li>
              {/each}
            </ul>
          </details>
        {:else}
          <p class="text-[11px] text-slate-500">No frames yet — extract some from the Frames tab first.</p>
        {/if}
      {/if}
    </div>

    <!-- ============ adapter / optimizer ============ -->
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <h3 class="text-sm font-medium text-slate-200 mb-3">Adapter &amp; optimizer</h3>
      <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">rank</span>
          <input
            type="number" min="1" step="1" value={cfg.rank}
            onchange={(e) => patchField("rank", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">alpha (kohya only)</span>
          <input
            type="number" min="1" step="1" value={cfg.alpha}
            onchange={(e) => patchField("alpha", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">learning_rate</span>
          <input
            type="number" step="any" value={cfg.learning_rate}
            onchange={(e) => patchField("learning_rate", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">weight_decay</span>
          <input
            type="number" step="any" value={cfg.weight_decay}
            onchange={(e) => patchField("weight_decay", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">eps</span>
          <input
            type="number" step="any" value={cfg.eps}
            onchange={(e) => patchField("eps", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">warmup_steps</span>
          <input
            type="number" min="0" step="1" value={cfg.warmup_steps}
            onchange={(e) => patchField("warmup_steps", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">gradient_clipping</span>
          <input
            type="number" step="any" value={cfg.gradient_clipping}
            onchange={(e) => patchField("gradient_clipping", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">betas (β₁,β₂)</span>
          <input
            value={cfg.optimizer_betas.join(",")}
            onchange={(e) => {
              const parts = (e.target as HTMLInputElement).value.split(",").map(s => Number(s.trim())).filter(n => !Number.isNaN(n));
              if (parts.length === 2) patchField("optimizer_betas", parts);
            }}
            placeholder="0.9, 0.99"
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
      </div>
    </div>

    <!-- ============ batching / resolution ============ -->
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <h3 class="text-sm font-medium text-slate-200 mb-3">Batching &amp; resolution</h3>
      <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs mb-3">
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">micro_batch_size</span>
          <input
            type="number" min="1" step="1" value={cfg.micro_batch_size}
            onchange={(e) => patchField("micro_batch_size", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">grad_accum_steps</span>
          <input
            type="number" min="1" step="1" value={cfg.gradient_accumulation_steps}
            onchange={(e) => patchField("gradient_accumulation_steps", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <div class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">effective batch</span>
          <div class="mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono text-slate-400">
            {cfg.micro_batch_size * cfg.gradient_accumulation_steps}
          </div>
        </div>
      </div>

      <div class="mb-3">
        <span class="text-[10px] uppercase tracking-wide text-slate-500">resolution buckets</span>
        <div class="flex flex-wrap gap-1 mt-1">
          {#each RESOLUTION_PRESETS as p (p.key)}
            <button
              type="button"
              onclick={() => pickResolutionPreset(p.values)}
              class="px-3 py-1 text-xs rounded {cfg.resolutions.join(",") === p.values.join(",") ? 'bg-accent-700 text-white' : 'bg-ink-800 text-slate-300 hover:bg-ink-700'}"
            >{p.label}</button>
          {/each}
        </div>
      </div>

      <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <label class="flex items-center gap-2">
          <input
            type="checkbox" checked={cfg.enable_ar_bucket}
            onchange={(e) => patchField("enable_ar_bucket", (e.target as HTMLInputElement).checked)}
            class="w-4 h-4 rounded bg-ink-950 border-ink-700 accent-accent-500"
          />
          <span class="text-slate-300">enable_ar_bucket</span>
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">min_ar</span>
          <input
            type="number" step="any" value={cfg.min_ar}
            onchange={(e) => patchField("min_ar", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">max_ar</span>
          <input
            type="number" step="any" value={cfg.max_ar}
            onchange={(e) => patchField("max_ar", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">num_ar_buckets</span>
          <input
            type="number" min="1" step="1" value={cfg.num_ar_buckets}
            onchange={(e) => patchField("num_ar_buckets", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
      </div>
    </div>

    <!-- ============ Anima specifics + duration ============ -->
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <h3 class="text-sm font-medium text-slate-200 mb-3">Schedule &amp; Anima specifics</h3>
      <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">epochs</span>
          <input
            type="number" min="1" step="1" value={cfg.epochs}
            onchange={(e) => patchField("epochs", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">eval_every_n_epochs</span>
          <input
            type="number" min="1" step="1" value={cfg.eval_every_n_epochs}
            onchange={(e) => patchField("eval_every_n_epochs", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">save_every_n_epochs</span>
          <input
            type="number" min="1" step="1" value={cfg.save_every_n_epochs}
            onchange={(e) => patchField("save_every_n_epochs", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">sigmoid_scale</span>
          <input
            type="number" step="any" value={cfg.sigmoid_scale}
            onchange={(e) => patchField("sigmoid_scale", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block col-span-2">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">llm_adapter_lr (keep at 0)</span>
          <input
            type="number" step="any" value={cfg.llm_adapter_lr}
            onchange={(e) => patchField("llm_adapter_lr", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono {cfg.llm_adapter_lr !== 0 ? 'border-amber-700' : ''}"
          />
          {#if cfg.llm_adapter_lr !== 0}
            <span class="block text-[10px] text-amber-400 mt-1">
              ⚠ Non-zero llm_adapter_lr causes "style dilution" on Anima — recommended value is 0.
            </span>
          {/if}
        </label>
      </div>
    </div>

    <!-- ============ captioning ============ -->
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <h3 class="text-sm font-medium text-slate-200 mb-3">Captioning</h3>
      <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">caption_mode</span>
          <select
            value={cfg.caption_mode}
            onchange={(e) => patchField("caption_mode", (e.target as HTMLSelectElement).value)}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          >
            <option value="tags">tags only</option>
            <option value="nl">natural language only</option>
            <option value="mixed">mixed (recommended)</option>
          </select>
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">tag_dropout %</span>
          <input
            type="number" min="0" max="100" step="1" value={cfg.tag_dropout_pct}
            onchange={(e) => patchField("tag_dropout_pct", Number((e.target as HTMLInputElement).value))}
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-slate-500">trigger_token (optional)</span>
          <input
            value={cfg.trigger_token}
            onchange={(e) => patchField("trigger_token", (e.target as HTMLInputElement).value)}
            placeholder="e.g. mychar"
            class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono"
          />
        </label>
      </div>
    </div>

    <!-- ============ checkpoint retention + run history ============ -->
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-medium text-slate-200">Runs &amp; checkpoints</h3>
        <button
          type="button"
          onclick={() => trainingStore.refreshRuns()}
          class="text-xs text-slate-500 hover:text-slate-300"
        >Refresh</button>
      </div>

      <label class="block text-xs mb-3">
        <span class="text-[10px] uppercase tracking-wide text-slate-500">keep last N checkpoints (0 = keep all)</span>
        <input
          type="number" min="0" step="1" value={cfg.keep_last_n_checkpoints}
          onchange={(e) => patchField("keep_last_n_checkpoints", Number((e.target as HTMLInputElement).value))}
          class="w-full mt-1 px-3 py-1.5 bg-ink-950 border border-ink-700 rounded font-mono max-w-[12rem]"
        />
        <span class="block text-[10px] text-slate-600 mt-1">
          Older checkpoints are pruned at the end of each run. Default
          (<code class="text-slate-400">0</code>) keeps every checkpoint.
        </span>
      </label>

      {#if runs.length === 0}
        <p class="text-xs text-slate-500">No runs yet. Click <em>Start training</em> above to launch one.</p>
      {:else}
        <ul class="space-y-2 text-xs">
          {#each runs as r (r.name)}
            <li class="bg-ink-950 border border-ink-700 rounded">
              <div class="flex items-center justify-between p-2">
                <button
                  type="button"
                  onclick={() => toggleRun(r.name)}
                  class="flex-1 text-left flex items-center gap-2"
                >
                  <span class="text-slate-500 text-[10px]">{expandedRun === r.name ? "▼" : "▶"}</span>
                  <span class="font-mono text-slate-200">{r.name}</span>
                  <span class="text-slate-500">{r.checkpoints} checkpoint{r.checkpoints === 1 ? "" : "s"}</span>
                  {#if r.latest_checkpoint}
                    <span class="text-slate-500">latest: <span class="font-mono text-slate-300">{r.latest_checkpoint}</span></span>
                  {/if}
                </button>
                <button
                  type="button"
                  onclick={() => deleteRun(r.name)}
                  class="text-[10px] text-slate-500 hover:text-red-400 ml-2"
                  title="Delete this run"
                >del run</button>
              </div>
              {#if expandedRun === r.name && runCheckpoints[r.name]}
                <div class="border-t border-ink-700 p-2">
                  {#if runCheckpoints[r.name].checkpoints.length === 0}
                    <p class="text-slate-500">No checkpoints saved in this run yet.</p>
                  {:else}
                    <ul class="space-y-1">
                      {#each runCheckpoints[r.name].checkpoints as cp (cp.name)}
                        <li class="flex items-center justify-between gap-2 px-2 py-1 hover:bg-ink-900 rounded">
                          <span class="font-mono text-slate-300">{cp.name}</span>
                          <span class="text-[10px] text-slate-500">
                            {cp.epoch !== null ? `epoch ${cp.epoch}` : `step ${cp.step}`}
                            · {fmtBytes(cp.size_bytes)}
                            · {new Date(cp.modified_at).toLocaleString()}
                          </span>
                          <div class="flex gap-2">
                            <button
                              type="button"
                              disabled={isRunning || busy}
                              onclick={() => resumeFromCheckpoint(r.name, cp.name)}
                              class="text-[10px] text-accent-400 hover:text-accent-300 disabled:opacity-40"
                            >resume</button>
                            <button
                              type="button"
                              disabled={isRunning}
                              onclick={() => deleteCheckpoint(r.name, cp.name)}
                              class="text-[10px] text-slate-500 hover:text-red-400 disabled:opacity-40"
                            >delete</button>
                          </div>
                        </li>
                      {/each}
                    </ul>
                  {/if}
                </div>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}
    </div>

    <!-- ============ live log ============ -->
    <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-sm font-medium text-slate-200">Trainer log</h3>
        <div class="flex gap-2">
          <button
            type="button"
            onclick={() => trainingStore.refreshLog()}
            class="text-xs text-slate-500 hover:text-slate-300"
          >Reload</button>
        </div>
      </div>
      <div
        bind:this={logPanel}
        class="bg-black/60 border border-ink-800 rounded p-2 h-64 overflow-y-auto font-mono text-[11px] leading-snug"
      >
        {#if log.length === 0}
          <span class="text-slate-600">No log lines yet.</span>
        {:else}
          {#each log as l, i (i)}
            <div class="{l.stream === 'stderr' ? 'text-amber-300' : 'text-slate-400'}">{l.line}</div>
          {/each}
        {/if}
      </div>
    </div>
  {/if}
</div>
