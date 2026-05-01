export type Tab = "sources" | "frames" | "training" | "settings";

class ViewStore {
  tab = $state<Tab>("frames");
  density = $state<number>(7); // columns in the frame grid (3-12)
  sourceFilter = $state<string | null>(null);
  // Server-side tag filter for the frames grid. Whitespace-separated
  // substrings; tokens prefixed with `~` negate. Bound to the search input
  // in the top bar (debounced) and read by FramesTab to refresh the list.
  tagQuery = $state<string>("");

  setDensity(n: number) {
    this.density = Math.max(3, Math.min(12, n));
  }
}

export const viewStore = new ViewStore();
