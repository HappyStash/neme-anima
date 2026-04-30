<script lang="ts">
  type Props = {
    title: string;
    subtitle: string;
    icon: string;
    onpaths: (paths: string[]) => void;
  };
  const { title, subtitle, icon, onpaths }: Props = $props();

  let dragging = $state(false);

  function fileDotPath(f: File): string | null {
    // Electron and a few embedded webviews expose File.path. Browsers don't.
    const anyF = f as unknown as { path?: unknown };
    return typeof anyF.path === "string" && anyF.path ? anyF.path : null;
  }

  function pathsFromUriList(dt: DataTransfer): string[] {
    // Drag-and-drop from File Explorer / Finder usually populates `text/uri-list`
    // with `file://` URIs. The server resolves these (incl. Windows-on-WSL).
    const raw = dt.getData("text/uri-list");
    if (!raw) return [];
    return raw
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter((s) => s && !s.startsWith("#"));
  }

  function pathsFromTextPlain(dt: DataTransfer): string[] {
    // Some sources (Windows Explorer in particular) drop the absolute path
    // as plain text — one path per line.
    const raw = dt.getData("text/plain");
    if (!raw) return [];
    const lines = raw.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
    // Heuristic: looks like a path — drive letter, leading slash, or file:// URI.
    return lines.filter((l) => /^([A-Za-z]:[\\/]|\/|file:\/\/)/.test(l));
  }

  function promptForMissing(filenames: string[]): string[] {
    const hint = filenames.join("\n");
    const input = prompt(
      "Your browser hid the file paths for these drops. Paste each absolute path " +
        "(one per line, in the same order). Windows paths and file:// URIs are accepted.\n\n" +
        "Files:\n" + hint,
      hint,
    );
    if (!input) return [];
    return input.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  }

  function handleDrop(ev: DragEvent) {
    ev.preventDefault();
    dragging = false;
    const dt = ev.dataTransfer;
    if (!dt) return;

    // 1. Best source: real paths exposed via File.path (Electron only).
    const files = Array.from(dt.files ?? []);
    const direct = files.map(fileDotPath).filter((p): p is string => !!p);
    if (direct.length > 0 && direct.length === files.length) {
      onpaths(direct);
      return;
    }

    // 2. Next best: URIs / plain-text paths from the drag payload.
    const fromDt = pathsFromUriList(dt);
    if (fromDt.length > 0) {
      onpaths(fromDt);
      return;
    }
    const fromText = pathsFromTextPlain(dt);
    if (fromText.length > 0) {
      onpaths(fromText);
      return;
    }

    // 3. Fallback: the browser only gave us File handles with no path. Ask the user.
    if (files.length > 0) {
      const supplied = promptForMissing(files.map((f) => f.name));
      if (supplied.length > 0) onpaths(supplied);
    }
  }

  function handleManual() {
    const input = prompt(
      "Paste one absolute path per line. Windows paths (C:\\…) and file:// URIs are accepted.",
    );
    if (!input) return;
    onpaths(input.split(/\r?\n/).map((s) => s.trim()).filter(Boolean));
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
