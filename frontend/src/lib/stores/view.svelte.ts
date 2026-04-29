export type Tab = "sources" | "frames" | "settings";

class ViewStore {
  tab = $state<Tab>("frames");
  density = $state<number>(7); // columns in the frame grid (3-12)
  sourceFilter = $state<string | null>(null);

  setDensity(n: number) {
    this.density = Math.max(3, Math.min(12, n));
  }
}

export const viewStore = new ViewStore();
