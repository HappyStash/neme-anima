<script lang="ts">
  type Props = {
    title: string;
    subtitle: string;
    icon: string;
    onpaths: (paths: string[]) => void;
  };
  const { title, subtitle, icon, onpaths }: Props = $props();

  let dragging = $state(false);

  function fileToPath(f: File): string {
    // Browsers don't expose absolute paths for security. Best-effort: use the
    // File.path provided by Electron/webkitdirectory if available; fall back
    // to a "vfs://" sentinel that the user must replace by typing the real path.
    const anyF = f as unknown as { path?: string };
    return anyF.path ?? `vfs://${f.name}`;
  }

  function handleDrop(ev: DragEvent) {
    ev.preventDefault();
    dragging = false;
    const files = Array.from(ev.dataTransfer?.files ?? []);
    if (files.length === 0) return;
    onpaths(files.map(fileToPath));
  }

  function handleManual() {
    const input = prompt("Paste an absolute path (one per line):");
    if (!input) return;
    onpaths(input.split("\n").map((s) => s.trim()).filter(Boolean));
  }
</script>

<button
  type="button"
  ondragover={(e) => { e.preventDefault(); dragging = true; }}
  ondragleave={() => (dragging = false)}
  ondrop={handleDrop}
  onclick={handleManual}
  class="w-full text-left bg-ink-900 border-2 border-dashed rounded-xl px-4 py-4 transition-all flex items-center gap-3.5
    {dragging ? 'border-accent-500 bg-ink-800 shadow-[0_0_24px_rgba(99,102,241,0.15)]' : 'border-ink-700 hover:border-accent-500'}"
>
  <div class="w-9 h-9 rounded-lg gradient-accent flex items-center justify-center text-white text-lg shadow-[0_2px_12px_rgba(99,102,241,0.3)] flex-shrink-0">
    {icon}
  </div>
  <div>
    <p class="text-slate-200 text-sm font-medium">{title}</p>
    <p class="text-slate-500 text-xs mt-0.5">{subtitle}</p>
  </div>
</button>
