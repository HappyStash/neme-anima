import * as api from "$lib/api";
import { SelectionModel } from "$lib/selection";
import type { FrameRecord, ServerEvent } from "$lib/types";

class FramesStore {
  items = $state<FrameRecord[]>([]);
  // Unfiltered count for the current source/kept_only view — used by the
  // top-bar count badge to render "X / total" when a tag query is active.
  totalInView = $state<number>(0);
  loading = $state(false);
  selection = new SelectionModel();
  selectionVersion = $state(0); // bump to force reactivity for selection changes
  // Per-filename counter bumped each time an LLM description finishes for a
  // frame. FrameThumb watches its own filename's value and runs the badge-pop
  // animation on every tick. We use a counter (not a boolean) so re-describing
  // an already-described frame still triggers the animation as feedback.
  describedVersion = $state<Map<string, number>>(new Map());
  // Per-filename counter bumped each time a WD14 retag finishes for a frame.
  // FrameThumb watches its own filename's value to invalidate its cached
  // tagText so the next hover renders the freshly-tagged line. Same shape as
  // describedVersion — a counter (not a boolean) so repeated retags keep
  // invalidating the cache.
  retaggedVersion = $state<Map<string, number>>(new Map());
  // Filenames currently in flight for a bulk re-tag / re-describe action. The
  // ActionBar populates this with the whole selection up-front so frames that
  // haven't been reached yet still show a spinner (queued state), and clears
  // each one as the per-frame call resolves. FrameThumb reads this to render
  // a centered spinner overlay and absorb clicks while busy.
  processing = $state<Set<string>>(new Set());

  async refresh(
    slug: string,
    opts: { source?: string; query?: string } = {},
  ) {
    this.loading = true;
    try {
      const page = await api.listFrames(slug, opts);
      this.items = page.items;
      this.totalInView = page.total;
      // Drop per-filename version maps and the in-flight set on refresh — a
      // different filter or project could reuse filenames coincidentally, and
      // we never want a stale bump or a stranded spinner to leak into a fresh
      // view.
      this.describedVersion = new Map();
      this.retaggedVersion = new Map();
      this.processing = new Set();
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

  /** Bump a frame's retag counter so FrameThumb invalidates its tag cache. */
  markRetagged(filename: string) {
    const next = new Map(this.retaggedVersion);
    next.set(filename, (next.get(filename) ?? 0) + 1);
    this.retaggedVersion = next;
  }

  /** Add filenames to the in-flight set so their tiles show a spinner. */
  markProcessing(filenames: Iterable<string>) {
    const next = new Set(this.processing);
    for (const f of filenames) next.add(f);
    this.processing = next;
  }

  /** Remove a single filename from the in-flight set (per-frame finish). */
  unmarkProcessing(filename: string) {
    if (!this.processing.has(filename)) return;
    const next = new Set(this.processing);
    next.delete(filename);
    this.processing = next;
  }

  /** Drop filenames from the selection without removing the underlying rows.
   *  Used by bulk-retag flows to clear each frame from selection as soon as
   *  it finishes processing. */
  deselect(filenames: Iterable<string>) {
    this.selection.remove(filenames);
    this.selectionVersion++;
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
