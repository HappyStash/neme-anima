import { describe, expect, it } from "vitest";
import { SelectionModel } from "../src/lib/selection";

describe("SelectionModel", () => {
  const items = ["a", "b", "c", "d", "e", "f"];

  it("bare click selects only that item and updates anchor", () => {
    const sel = new SelectionModel();
    sel.click(items, 2, { shift: false, ctrl: false });
    expect(sel.selected()).toEqual(new Set(["c"]));
    expect(sel.anchor()).toBe(2);
  });

  it("shift-click extends from anchor to target", () => {
    const sel = new SelectionModel();
    sel.click(items, 1, { shift: false, ctrl: false }); // anchor at b
    sel.click(items, 4, { shift: true, ctrl: false });
    expect(sel.selected()).toEqual(new Set(["b", "c", "d", "e"]));
    expect(sel.anchor()).toBe(1); // anchor unchanged on shift-click
  });

  it("ctrl-click toggles a single item without changing others", () => {
    const sel = new SelectionModel();
    sel.click(items, 0, { shift: false, ctrl: false });
    sel.click(items, 3, { shift: false, ctrl: true });
    expect(sel.selected()).toEqual(new Set(["a", "d"]));
    sel.click(items, 0, { shift: false, ctrl: true });
    expect(sel.selected()).toEqual(new Set(["d"]));
  });

  it("selectAll picks every item", () => {
    const sel = new SelectionModel();
    sel.selectAll(items);
    expect(sel.selected()).toEqual(new Set(items));
  });

  it("clear empties the selection but keeps the anchor", () => {
    const sel = new SelectionModel();
    sel.click(items, 2, { shift: false, ctrl: false });
    sel.clear();
    expect(sel.selected()).toEqual(new Set());
    // Anchor preserved so a later shift-click works.
    expect(sel.anchor()).toBe(2);
  });

  it("count reflects size", () => {
    const sel = new SelectionModel();
    expect(sel.count()).toBe(0);
    sel.selectAll(items);
    expect(sel.count()).toBe(items.length);
  });
});
