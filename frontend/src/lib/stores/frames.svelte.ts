import * as api from "$lib/api";
import { SelectionModel } from "$lib/selection";
import type { FrameRecord, ServerEvent } from "$lib/types";

class FramesStore {
  items = $state<FrameRecord[]>([]);
  loading = $state(false);
  selection = new SelectionModel();
  selectionVersion = $state(0); // bump to force reactivity for selection changes

  async refresh(slug: string, opts: { source?: string } = {}) {
    this.loading = true;
    try {
      const page = await api.listFrames(slug, opts);
      this.items = page.items;
    } finally {
      this.loading = false;
    }
  }

  ingest(event: ServerEvent) {
    if (event.type === "job.frame") {
      // A frame was just written by a running job — we'd ideally splice it in,
      // but the simpler invariant is: refresh on job.done. For now, no-op.
    }
  }

  click(index: number, mods: { shift: boolean; ctrl: boolean }) {
    this.selection.click(this.items.map(i => i.filename), index, mods);
    this.selectionVersion++;
  }

  selectAll() {
    this.selection.selectAll(this.items.map(i => i.filename));
    this.selectionVersion++;
  }

  clear() {
    this.selection.clear();
    this.selectionVersion++;
  }

  selectedFilenames(): string[] {
    return [...this.selection.selected()];
  }

  removeLocal(filenames: string[]) {
    const set = new Set(filenames);
    this.items = this.items.filter(i => !set.has(i.filename));
    this.selection.remove(set);
    this.selectionVersion++;
  }
}

export const framesStore = new FramesStore();
