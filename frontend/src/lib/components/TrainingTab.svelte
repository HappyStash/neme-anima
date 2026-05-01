<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as api from "$lib/api";
  import { projectsStore } from "$lib/stores/projects.svelte";
  import { trainingStore } from "$lib/stores/training.svelte";
  import type { TrainingConfig, TrainingPathCheck, TrainingRun } from "$lib/types";

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

  // Sub-tab state. Persisted on the component, not the store, since the
  // user's choice is per-session.
  type SubTab = "run" | "dataset" | "settings";
  let subtab = $state<SubTab>("run");

  async function copyPath(path: string) {
    try {
      await navigator.clipboard.writeText(path);
    } catch (e) {
      // Clipboard API can fail outside secure contexts — fall back to a prompt.
      window.prompt("Copy path:", path);
    }
  }

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

  // Periodic poll: status + runs + expanded run's checkpoints. Acts as a
  // backstop for the WebSocket and ensures freshly-saved checkpoints
  // appear without the user clicking anything.
  let pollHandle: ReturnType<typeof setInterval> | null = null;
  onMount(() => {
    pollHandle = setInterval(async () => {
      if (!trainingStore.slug) return;
      try {
        const [st] = await Promise.all([
          api.getTrainingStatus(trainingStore.slug),
          trainingStore.refreshRuns(),
        ]);
        trainingStore.status = st;
        if (expandedRun) {
          // Refresh the open run's checkpoints in-place.
          runCheckpoints[expandedRun] = await api.listTrainingCheckpoints(
            trainingStore.slug, expandedRun,
          );
        }
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

  function fmtElapsed(startIso: string, endIso?: string | null): string {
    const t0 = new Date(startIso).getTime();
    const t1 = endIso ? new Date(endIso).getTime() : Date.now();
    const sec = Math.max(0, Math.round((t1 - t0) / 1000));
    if (sec < 60) return `${sec}s`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    if (m < 60) return `${m}m ${s}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
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
  // The first run that has a resumable DeepSpeed state. Used to enable
  // the global "Continue last run" affordance in the header.
  let resumableRun = $derived<TrainingRun | null>(
    runs.find((r) => !!r.resumable_subdir) ?? null,
  );
  // A run is only worth resuming if cfg.epochs is higher than what's
  // already been trained — otherwise diffusion-pipe would still grind out
  // one more epoch and save it. Returns true when there is real work left.
  function hasRoomToTrain(r: TrainingRun | null): boolean {
    if (!r || r.latest_epoch == null) return true; // unknown → allow
    if (!cfg) return false;
    return cfg.epochs > r.latest_epoch;
  }
  let canResume = $derived(
    !isRunning
      && !!resumableRun
      && problems.length === 0
      && hasRoomToTrain(resumableRun),
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
  async function continueRun(runName?: string) {
    busy = true;
    try {
      await trainingStore.resume(runName ? { run_dir_name: runName } : {});
      // Switch to the Run sub-tab so the user immediately sees what's happening.
      subtab = "run";
    } catch (e) {
      alert(`Continue failed: ${e}`);
    } finally {
      busy = false;
    }
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
  async function deleteCheckpoint(runName: string, ckptName: string, subdir: string) {
    if (!confirm(`Delete checkpoint ${subdir ? subdir + "/" : ""}${ckptName}?`)) return;
    await trainingStore.deleteCheckpoint(runName, ckptName, subdir);
    await refreshExpandedRun();
  }
  async function deleteRun(runName: string) {
    if (!confirm(`Delete the entire run ${runName} and all its checkpoints?`)) return;
    await trainingStore.deleteRun(runName);
    if (expandedRun === runName) expandedRun = null;
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

  // Color theme keyed off run status. Returns CSS class fragments so we
  // can drive both the badge and the surrounding card border in lockstep.
  function statusTheme(s: string): { tone: string; pill: string; bar: string; border: string; label: string } {
    switch (s) {
      case "running":
        return {
          tone: "emerald",
          pill: "bg-emerald-900/50 text-emerald-300 border border-emerald-800",
          bar: "bg-emerald-500",
          border: "border-emerald-800/60",
          label: "Running",
        };
      case "starting":
        return {
          tone: "amber",
          pill: "bg-amber-900/50 text-amber-300 border border-amber-800",
          bar: "bg-amber-500",
          border: "border-amber-800/60",
          label: "Starting…",
        };
      case "stopping":
        return {
          tone: "amber",
          pill: "bg-amber-900/50 text-amber-300 border border-amber-800",
          bar: "bg-amber-500",
          border: "border-amber-800/60",
          label: "Stopping…",
        };
      case "stopped":
        return {
          tone: "slate",
          pill: "bg-slate-800 text-slate-300 border border-slate-700",
          bar: "bg-slate-500",
          border: "border-slate-700",
          label: "Stopped",
        };
      case "finished":
        return {
          tone: "emerald",
          pill: "bg-emerald-900/60 text-emerald-200 border border-emerald-700",
          bar: "bg-emerald-500",
          border: "border-emerald-800/60",
          label: "✓ Finished",
        };
      case "failed":
        return {
          tone: "red",
          pill: "bg-red-900/60 text-red-200 border border-red-700",
          bar: "bg-red-500",
          border: "border-red-800/60",
          label: "✗ Failed",
        };
      default:
        return {
          tone: "slate",
          pill: "bg-slate-800 text-slate-400 border border-slate-700",
          bar: "bg-slate-500",
          border: "border-ink-700",
          label: s,
        };
    }
  }

  // Compute (epoch%, label) for the current-run progress bar.
  function progressInfo(rs: NonNullable<typeof runState>): { pct: number; label: string } {
    const total = rs.total_epochs ?? 0;
    const cur = rs.epoch ?? 0;
    if (rs.status === "finished") return { pct: 100, label: total ? `${total} / ${total} epochs` : "completed" };
    if (!total) return { pct: 0, label: cur ? `epoch ${cur}` : "preparing…" };
    const pct = Math.max(0, Math.min(100, Math.round((cur / total) * 100)));
    return { pct, label: `${cur} / ${total} epochs` };
  }
</script>

<div class="mt-4 max-w-4xl mx-auto">
  <div class="flex items-center justify-between mb-4 gap-4">
    <h2 class="text-base font-semibold text-slate-200">LoRA training</h2>

    <!-- top-right control row -->
    <div class="flex items-center gap-2 shrink-0">
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
          onclick={() => continueRun(resumableRun?.name)}
          disabled={!canResume || busy}
          class="px-3 py-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-slate-100 border border-ink-700 disabled:opacity-40 disabled:cursor-not-allowed"
          title={canResume
            ? `Continue ${resumableRun?.name} from epoch ${resumableRun?.latest_epoch ?? "?"}`
            : !resumableRun
              ? "No prior run with a resumable state"
              : !hasRoomToTrain(resumableRun)
                ? `Already at epoch ${resumableRun?.latest_epoch} / ${cfg?.epochs} — raise 'epochs' in Settings to continue`
                : (problems[0] ?? "Cannot continue right now")}
        >Continue last</button>
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
    <!-- ============ sub-tabs ============ -->
    <div class="flex gap-1 mb-3 border-b border-ink-800">
      {#each [{ k: "run", label: "Run" }, { k: "dataset", label: "Dataset" }, { k: "settings", label: "Settings" }] as t}
        <button
          type="button"
          onclick={() => (subtab = t.k as SubTab)}
          class="px-4 py-2 text-xs font-medium border-b-2 -mb-px transition-colors
            {subtab === t.k
              ? 'border-accent-500 text-slate-100'
              : 'border-transparent text-slate-500 hover:text-slate-300'}"
        >{t.label}{t.k === "settings" && problems.length > 0 ? ` (${problems.length})` : ""}</button>
      {/each}
    </div>

    {#if subtab === "run"}
      <!-- ====================================================== -->
      <!-- ==================== RUN SUB-TAB ===================== -->
      <!-- ====================================================== -->

      {#if problems.length > 0}
        <div class="bg-amber-900/30 border border-amber-800 rounded-xl p-3 mb-3 text-xs text-amber-200">
          <div class="font-medium mb-1">Cannot start a run yet:</div>
          <ul class="list-disc list-inside space-y-0.5">
            {#each problems as p}
              <li>{p}</li>
            {/each}
          </ul>
          <button
            type="button"
            onclick={() => (subtab = "settings")}
            class="mt-2 text-amber-300 hover:text-amber-100 underline text-[11px]"
          >Open Settings →</button>
        </div>
      {/if}

      <!-- ============ Current run card ============ -->
      {#if runState}
        {@const theme = statusTheme(runState.status)}
        {@const prog = progressInfo(runState)}
        <div class="bg-ink-900 border-2 {theme.border} rounded-xl p-4 mb-4">
          <div class="flex items-start justify-between gap-3 mb-3">
            <div class="min-w-0 flex-1">
              <div class="flex items-center gap-2 mb-1">
                <span class="text-[10px] uppercase tracking-wide text-slate-500">Current run</span>
                <span class="text-[10px] font-mono text-slate-400 truncate">{runState.run_name}</span>
              </div>
              <div class="flex items-center gap-3 flex-wrap">
                <span class="text-sm px-2.5 py-0.5 rounded-full {theme.pill}">
                  {theme.label}
                </span>
                {#if runState.resumed_from}
                  <span class="text-[11px] text-slate-500">
                    resumed from <span class="font-mono text-slate-400">{runState.resumed_from}</span>
                  </span>
                {/if}
              </div>
            </div>
            <div class="text-right text-[11px] text-slate-500 shrink-0">
              {#if runState.finished_at}
                <div>
                  finished {new Date(runState.finished_at).toLocaleString()}
                </div>
                <div>
                  ran for {fmtElapsed(runState.started_at, runState.finished_at)}
                </div>
              {:else}
                <div>
                  started {new Date(runState.started_at).toLocaleString()}
                </div>
                <div>
                  elapsed {fmtElapsed(runState.started_at)}
                </div>
              {/if}
            </div>
          </div>

          <!-- Progress bar -->
          <div class="mb-3">
            <div class="flex items-center justify-between text-[11px] mb-1">
              <span class="text-slate-400 font-mono">{prog.label}</span>
              <span class="text-slate-300 font-mono">{prog.pct}%</span>
            </div>
            <div class="w-full h-2 bg-ink-950 rounded-full overflow-hidden">
              <div
                class="h-full {theme.bar} transition-all duration-500"
                style="width: {prog.pct}%"
              ></div>
            </div>
          </div>

          <!-- Live numerics -->
          <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
            <div class="bg-ink-950 rounded p-2">
              <div class="text-[10px] text-slate-500 uppercase tracking-wide">step</div>
              <div class="font-mono text-slate-200">{runState.step ?? "—"}</div>
            </div>
            <div class="bg-ink-950 rounded p-2">
              <div class="text-[10px] text-slate-500 uppercase tracking-wide">loss</div>
              <div class="font-mono text-slate-200">{runState.loss != null ? runState.loss.toFixed(4) : "—"}</div>
            </div>
            <div class="bg-ink-950 rounded p-2">
              <div class="text-[10px] text-slate-500 uppercase tracking-wide">epoch</div>
              <div class="font-mono text-slate-200">
                {runState.epoch ?? "—"}{runState.total_epochs ? ` / ${runState.total_epochs}` : ""}
              </div>
            </div>
          </div>

          {#if runState.error}
            <div class="mt-3 px-3 py-2 bg-red-950/40 border border-red-900 rounded text-xs text-red-300 break-words">
              {runState.error}
            </div>
          {/if}
          {#if runState.last_log_line}
            <div class="mt-2 text-[11px] font-mono text-slate-500 truncate">{runState.last_log_line}</div>
          {/if}
        </div>
      {/if}

      <!-- ============ Run history ============ -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-4">
        <h3 class="text-sm font-medium text-slate-200 mb-3">Run history</h3>

        {#if runs.length === 0}
          <p class="text-xs text-slate-500">
            No runs yet. Click <em>Start training</em> above to launch one.
          </p>
        {:else}
          <ul class="space-y-2 text-xs">
            {#each runs as r (r.name)}
              {@const expanded = expandedRun === r.name}
              {@const cur = runState && runState.run_name === r.name}
              {@const cps = runCheckpoints[r.name]}
              <li class="bg-ink-950 border border-ink-800 rounded-lg overflow-hidden">
                <!-- Run header row -->
                <div class="flex items-center gap-2 p-2.5">
                  <button
                    type="button"
                    onclick={() => toggleRun(r.name)}
                    class="flex-1 text-left flex items-center gap-2 min-w-0"
                  >
                    <span class="text-slate-500 text-[10px] w-3">{expanded ? "▼" : "▶"}</span>
                    <span class="font-mono text-slate-200 truncate">{r.name}</span>
                    {#if cur}
                      {@const t = statusTheme(runState.status)}
                      <span class="text-[10px] px-1.5 py-0.5 rounded {t.pill}">{t.label}</span>
                    {/if}
                  </button>

                  <div class="flex items-center gap-3 text-[11px] text-slate-500 shrink-0">
                    {#if r.latest_epoch != null}
                      <span title="Highest saved epoch">
                        epoch <span class="text-slate-300 font-mono">{r.latest_epoch}</span>
                      </span>
                    {/if}
                    <span title="Number of epoch checkpoints saved">
                      {r.checkpoints} ckpt{r.checkpoints === 1 ? "" : "s"}
                    </span>
                    <span title="Total disk usage including DeepSpeed state">
                      {fmtBytes(r.total_size_bytes)}
                    </span>

                    {#if r.resumable_subdir && !isRunning}
                      {@const room = hasRoomToTrain(r)}
                      <button
                        type="button"
                        onclick={() => continueRun(r.name)}
                        disabled={busy || problems.length > 0 || !room}
                        class="px-2.5 py-1 text-[11px] rounded bg-accent-700 hover:bg-accent-600 text-white disabled:opacity-40 disabled:cursor-not-allowed"
                        title={room
                          ? "Continue training from this run's last saved state"
                          : `Already at epoch ${r.latest_epoch} / ${cfg?.epochs} — raise 'epochs' in Settings to continue`}
                      >Continue</button>
                    {:else if !r.resumable_subdir}
                      <span class="text-[10px] text-slate-600 italic" title="No DeepSpeed save — can't resume">
                        not resumable
                      </span>
                    {/if}

                    <button
                      type="button"
                      onclick={() => copyPath(r.path)}
                      class="w-6 h-6 inline-flex items-center justify-center rounded text-slate-400 hover:text-slate-100 hover:bg-ink-800"
                      title="Copy run path: {r.path}"
                      aria-label="Copy run path"
                    >
                      <svg viewBox="0 0 24 24" class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="9" y="9" width="11" height="11" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                      </svg>
                    </button>
                    <button
                      type="button"
                      onclick={() => deleteRun(r.name)}
                      disabled={cur && isRunning}
                      class="w-6 h-6 inline-flex items-center justify-center rounded text-red-400 hover:text-white hover:bg-red-700 disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-red-400"
                      title="Delete this run"
                      aria-label="Delete this run"
                    >
                      <svg viewBox="0 0 24 24" class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                        <path d="M18 6L6 18M6 6l12 12"/>
                      </svg>
                    </button>
                  </div>
                </div>

                {#if expanded}
                  <div class="border-t border-ink-800 px-2 py-2 bg-black/20">
                    {#if !cps}
                      <p class="text-slate-500 text-[11px] px-2">Loading…</p>
                    {:else if cps.checkpoints.length === 0}
                      <p class="text-slate-500 text-[11px] px-2">No checkpoints saved in this run yet.</p>
                    {:else}
                      <ul class="space-y-1">
                        {#each cps.checkpoints.slice().reverse() as cp (cp.subdir + "/" + cp.name)}
                          {@const isStep = cp.epoch == null && cp.step != null}
                          <li class="flex items-center justify-between gap-2 px-2 py-1 hover:bg-ink-900 rounded">
                            <div class="flex items-center gap-2 min-w-0">
                              {#if isStep}
                                <span class="text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-slate-800 text-slate-400" title="DeepSpeed pipeline state — used for resume">
                                  state
                                </span>
                              {:else}
                                <span class="text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-emerald-900/40 text-emerald-300" title="LoRA adapter output — usable for inference">
                                  lora
                                </span>
                              {/if}
                              <span class="font-mono text-slate-300 truncate">{cp.name}</span>
                            </div>
                            <div class="flex items-center gap-2 text-[10px] text-slate-500 shrink-0">
                              <span>{fmtBytes(cp.size_bytes)}</span>
                              <span>{new Date(cp.modified_at).toLocaleString()}</span>
                              <button
                                type="button"
                                onclick={() => copyPath(cp.path)}
                                class="w-5 h-5 inline-flex items-center justify-center rounded text-slate-400 hover:text-slate-100 hover:bg-ink-800"
                                title="Copy checkpoint path: {cp.path}"
                                aria-label="Copy checkpoint path"
                              >
                                <svg viewBox="0 0 24 24" class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                  <rect x="9" y="9" width="11" height="11" rx="2" ry="2"/>
                                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                                </svg>
                              </button>
                              <button
                                type="button"
                                disabled={isRunning}
                                onclick={() => deleteCheckpoint(r.name, cp.name, cp.subdir)}
                                class="w-5 h-5 inline-flex items-center justify-center rounded text-red-400 hover:text-white hover:bg-red-700 disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-red-400"
                                title="Delete this checkpoint"
                                aria-label="Delete this checkpoint"
                              >
                                <svg viewBox="0 0 24 24" class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                                  <path d="M18 6L6 18M6 6l12 12"/>
                                </svg>
                              </button>
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

      <!-- ============ Live log ============ -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-4">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-medium text-slate-200">Trainer log</h3>
          <button
            type="button"
            onclick={() => trainingStore.refreshLog()}
            class="text-xs text-slate-500 hover:text-slate-300"
          >Reload</button>
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

    {:else if subtab === "dataset"}
      <!-- ====================================================== -->
      <!-- ================= DATASET SUB-TAB ==================== -->
      <!-- ====================================================== -->

      {#if preview}
        <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
          <h3 class="text-sm font-medium text-slate-200 mb-3">Dataset summary</h3>
          <div class="grid grid-cols-3 gap-3 mb-3 text-xs">
            <div class="bg-ink-950 rounded p-2">
              <div class="text-[10px] text-slate-500 uppercase tracking-wide">total images</div>
              <div class="font-mono text-slate-200">{preview.total_images}</div>
            </div>
            <div class="bg-ink-950 rounded p-2">
              <div class="text-[10px] text-slate-500 uppercase tracking-wide">with tags</div>
              <div class="font-mono text-slate-200">{preview.with_tags}</div>
            </div>
            <div class="bg-ink-950 rounded p-2">
              <div class="text-[10px] text-slate-500 uppercase tracking-wide">with NL desc.</div>
              <div class="font-mono text-slate-200">{preview.with_descriptions}</div>
            </div>
          </div>
          {#if preview.samples.length > 0}
            <details open>
              <summary class="text-xs text-slate-400 cursor-pointer hover:text-slate-200 mb-2">
                Caption preview (first {preview.samples.length})
              </summary>
              <ul class="space-y-2 text-[11px] font-mono text-slate-400">
                {#each preview.samples as s}
                  <li class="bg-ink-950 rounded p-2">
                    <div class="text-slate-300 truncate">{s.filename}</div>
                    <div class="text-slate-500 mt-1 break-words">{s.rendered}</div>
                  </li>
                {/each}
              </ul>
            </details>
          {:else}
            <p class="text-[11px] text-slate-500">
              No frames yet — extract some from the Frames tab first.
            </p>
          {/if}
        </div>
      {:else}
        <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 text-sm text-slate-400">
          Loading dataset preview…
        </div>
      {/if}

    {:else if subtab === "settings"}
      <!-- ====================================================== -->
      <!-- ================= SETTINGS SUB-TAB =================== -->
      <!-- ====================================================== -->

      <!-- Trainer paths -->
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
              <input
                value={pathDraft[f.key]}
                oninput={(e) => schedulePathCheck(f.key, (e.target as HTMLInputElement).value)}
                onblur={() => commitPath(f.key)}
                placeholder={f.placeholder}
                class="w-full px-3 py-1.5 bg-ink-950 border border-ink-700 rounded text-xs font-mono focus:outline-none focus:border-accent-500"
              />
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

      <!-- Preset -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-sm font-medium text-slate-200">Preset</h3>
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
        <p class="text-[11px] text-slate-500">
          <strong>style</strong> = recipe defaults (lr 2e-5, grad-accum 4, 40 epochs).
          <strong>character</strong> bumps lr to 5e-5 and epochs to 60 — community
          reports indicate 2e-5 is too low for character identity.
        </p>
      </div>

      <!-- Adapter / optimizer -->
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

      <!-- Batching / resolution -->
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

      <!-- Schedule + Anima specifics -->
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

      <!-- Captioning -->
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

      <!-- Retention -->
      <div class="bg-ink-900 border border-ink-700 rounded-xl p-4 mb-3">
        <h3 class="text-sm font-medium text-slate-200 mb-3">Checkpoint retention</h3>
        <label class="block text-xs">
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
      </div>
    {/if}
  {/if}
</div>
