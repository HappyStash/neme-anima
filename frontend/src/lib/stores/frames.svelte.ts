import * as api from "$lib/api";
import { SelectionModel } from "$lib/selection";
import type { FrameRecord, ServerEvent } from "$lib/types";

class FramesStore {
  items = $state<FrameRecord[]>([]);
  loading = $state(false);
  selection = new SelectionModel();
  selectionVersion = $state(0); // bump to force reactivity for selection changes
  // Per-filename counter bumped each time an LLM description finishes for a
  // frame. FrameThumb watches its own filename's value and runs the badge-pop
  // animation on every tick. We use a counter (not a boolean) so re-describing
  // an already-described frame still triggers the animation as feedback.
  describedVersion = $state<Map<string, number>>(new Map());

  async refresh(slug: string, opts: { source?: string } = {}) {
    this.loading = true;
    try {
      const page = await api.listFrames(slug, opts);
      this.items = page.items;
      // Drop describedVersion on refresh — a different filter or project
      // could reuse filenames coincidentally, and we never want a stale
      // bump from a previous view to fire animations on a fresh tile.
      this.describedVersion = new Map();
    } finally {
      this.loading = false;
    }
  }

  /** Mark a frame as freshly described: flip its has_description and bump the
   *  per-filename counter so FrameThumb runs its pop animation. */
  markDescribed(filename: string) {
    const idx = this.items.findIndex((i) => i.filename === filename);
    if (idx >= 0 && !this.items[idx].has_description) {
      this.items[idx] = { ...this.items[idx], has_description: true };
    }
    const next = new Map(this.describedVersion);
    next.set(filename, (next.get(filename) ?? 0) + 1);
    this.describedVersion = next;
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
