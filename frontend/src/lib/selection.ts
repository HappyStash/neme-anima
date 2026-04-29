/** Pure selection model. Stateless w.r.t the items array — caller passes
 * the current ordered list on every click so shift-click ranges are computed
 * against what the user actually sees. */
export class SelectionModel {
  private _selected = new Set<string>();
  private _anchor: number | null = null;

  selected(): Set<string> {
    return new Set(this._selected);
  }

  anchor(): number | null {
    return this._anchor;
  }

  count(): number {
    return this._selected.size;
  }

  has(item: string): boolean {
    return this._selected.has(item);
  }

  click(
    items: readonly string[],
    index: number,
    mods: { shift: boolean; ctrl: boolean },
  ): void {
    const target = items[index];
    if (target === undefined) return;

    if (mods.shift && this._anchor !== null) {
      const from = Math.min(this._anchor, index);
      const to = Math.max(this._anchor, index);
      const next = new Set<string>();
      for (let i = from; i <= to; i++) next.add(items[i]);
      this._selected = next;
      // Anchor unchanged on shift-click.
      return;
    }

    if (mods.ctrl) {
      if (this._selected.has(target)) this._selected.delete(target);
      else this._selected.add(target);
      this._anchor = index;
      return;
    }

    this._selected = new Set([target]);
    this._anchor = index;
  }

  selectAll(items: readonly string[]): void {
    this._selected = new Set(items);
  }

  clear(): void {
    this._selected.clear();
  }

  remove(items: Iterable<string>): void {
    for (const i of items) this._selected.delete(i);
  }
}
